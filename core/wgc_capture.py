# -*- coding: utf-8 -*-
"""Windows.Graphics.Capture (WGC) GPU捕获：raw COM实现
帧纹理由帧池在本项目D3D设备上创建（同设备对象拷贝实测可用），
CopySubresourceRegion到管线常驻纹理 -> CUDA映射 -> 预处理内核 -> TRT。
IID全部提取自本机 C:\\Windows\\System32\\WinMetadata\\Windows.Graphics.winmd（运行时权威）。"""
import ctypes as ct
import ctypes.wintypes as wt
import uuid
import struct
import numpy as np

from core.dxgi_capture import (D3D11_TEXTURE2D_DESC, DXGI_SAMPLE_DESC,
                               D3D11_MAPPED_SUBRESOURCE, DXGI_FORMAT_B8G8R8A8_UNORM,
                               D3D11_USAGE_STAGING, D3D11_CPU_ACCESS_READ)
from ctypes import c_void_p, byref, POINTER, HRESULT

IIG = lambda s: ct.create_string_buffer(uuid.UUID(s).bytes_le)
IID_IActivationFactory = "{00000035-0000-0000-C000-000000000046}"
IID_IGraphicsCaptureItem = "{79C3F95B-31F7-4EC2-A464-632EF5D30760}"
IID_ItemStatics2 = "{3B92ACC9-E584-5862-BF5C-9C316C6D2DBB}"
IID_FramePoolStatics2 = "{589B103F-6BBC-5DF5-A991-02E28B3B66D5}"
IID_FramePoolStatics = "{7784056A-67AA-4D53-AE54-1088D5A8CA21}"
IID_FramePool = "{24EB6D22-1975-422E-82E7-780DBD8DDF24}"
IID_Frame = "{FA50C623-38DA-4B32-ACF3-FA9734AD800E}"
IID_Session = "{814E42A9-F70F-4AD7-939B-FDDCC6EB880D}"
IID_Session3 = "{F2CDD966-22AE-5EA1-9596-3A289344C3BE}"   # 本机winmd解析（put_IsBorderRequired=slot7）
IID_IClosable = "{30D5A829-7FA4-4026-83BB-D75BAE4EA99E}"
IID_Device = "{A37624AB-8D5F-4650-9D3E-9EAE3D9BC670}"
IID_DxgiAccess = "{AF86E2E0-B12D-4C6A-9C5A-D7AA65101E90}"   # 新版Windows的IDirect3DDxgiInterfaceAccess（暴力搜索+GetInterface实测）
IID_ID3D11Texture2D = "{6F15AAF2-D208-4B89-9B93-1B6A3D1C3F3F}"
IID_IDXGIDevice = "{54EC77FA-1377-44E6-8C32-88FD5F44C84C}"

DXGI_ERROR_WAIT_TIMEOUT = 0x887A0027
SIZE_POOL = 3   # 帧池缓冲数


class SizeInt32(ct.Structure):
    _fields_ = [("Width", ct.c_int32), ("Height", ct.c_int32)]


def _vcall(addr, slot, restype, *args, argtypes=None):
    """WinRT/COM虚表直调：IUnknown(0-2)+IInspectable(3-5)+方法(6+)"""
    vtbl = ct.cast(c_void_p(addr), POINTER(c_void_p)).contents.value
    fp = ct.cast(c_void_p(vtbl), POINTER(c_void_p))[slot]
    if argtypes is None:
        argtypes = (c_void_p,) + tuple(type(a) for a in args)
    fn = ct.WINFUNCTYPE(restype, *argtypes)(fp)
    return fn(c_void_p(addr), *args)


def _qi(addr, iid):
    out = c_void_p()
    rc = _vcall(addr, 0, ct.c_ulong, IIG(iid), byref(out),
                argtypes=(c_void_p, ct.c_char_p, POINTER(c_void_p)))
    rc &= 0xFFFFFFFF
    if rc != 0:
        raise OSError(f"QI {iid} 失败: 0x{rc:08X}")
    return out.value


def _rel(addr):
    if addr:
        try:
            _vcall(addr, 2, ct.c_ulong, argtypes=(c_void_p,))
        except OSError:
            pass


class WGCCaptureError(RuntimeError):
    pass


class WGCCapture:
    """WGC捕获器：与GPUPipeline共用同一D3D设备，帧直接GPU侧进管线纹理"""

    def __init__(self, pipeline, monitor_name=None):
        self.pipeline = pipeline
        self.monitor_name = monitor_name
        self.w = 0
        self.h = 0
        self.ok = False
        self._pool = None
        self._session = None
        self._item = None
        self._device = None
        self.staging = 0
        self.stage_size = None
        self.last_stage = "-"

    def _activation_factory(self, class_name):
        combase = ct.windll.combase
        h = c_void_p()
        combase.WindowsCreateString.argtypes = [wt.LPCWSTR, ct.c_int, POINTER(c_void_p)]
        combase.WindowsCreateString.restype = ct.c_long
        rc = combase.WindowsCreateString(class_name, len(class_name), byref(h))
        if rc != 0:
            raise WGCCaptureError(f"WindowsCreateString失败: 0x{rc:08X}")
        f = c_void_p()
        combase.RoGetActivationFactory.argtypes = [c_void_p, ct.c_char_p, POINTER(c_void_p)]
        combase.RoGetActivationFactory.restype = ct.c_long
        rc = combase.RoGetActivationFactory(h, IIG(IID_IActivationFactory), byref(f))
        if rc != 0:
            raise WGCCaptureError(f"激活工厂 {class_name} 失败: 0x{rc & 0xFFFFFFFF:08X}")
        return f.value

    def open(self, monitor_rect):
        """monitor_rect: 主捕获显示器的RECT（决定帧池尺寸）"""
        try:
            import comtypes
            try:
                comtypes.CoInitializeEx(comtypes.COINIT_APARTMENTTHREADED)
            except Exception:
                pass
        except Exception:
            pass
        w = monitor_rect.right - monitor_rect.left
        h = monitor_rect.bottom - monitor_rect.top
        if w <= 0 or h <= 0:
            raise WGCCaptureError("显示器矩形无效")

        # 1. IDirect3DDevice：包装本项目D3D设备（帧池将在同一设备上建纹理）
        d3d11 = ct.WinDLL("d3d11.dll")
        CreateDirect3D11DeviceFromDXGIDevice = d3d11.CreateDirect3D11DeviceFromDXGIDevice
        CreateDirect3D11DeviceFromDXGIDevice.argtypes = [c_void_p, POINTER(c_void_p)]
        CreateDirect3D11DeviceFromDXGIDevice.restype = ct.c_long
        dxgi_dev = _qi(self.pipeline.d3d_device, IID_IDXGIDevice)
        insp = c_void_p()
        try:
            rc = CreateDirect3D11DeviceFromDXGIDevice(c_void_p(dxgi_dev), byref(insp))
        finally:
            _rel(dxgi_dev)
        if rc != 0 or not insp.value:
            raise WGCCaptureError(f"CreateDirect3D11DeviceFromDXGIDevice失败: 0x{rc & 0xFFFFFFFF:08X}")
        self._device = _qi(insp.value, IID_Device)
        _rel(insp.value)

        # 1.5 挂钩设备CreateTexture2D：捕获帧池创建的缓冲纹理指针
        #     （本机系统包装层的surface QI/GetInterface不可用，改由钩子直接拿到真纹理）
        self._install_tex_hook()

        # 2. GraphicsCaptureItem：Statics2.TryCreateFromMonitorId（本机winmd实测存在）
        f = self._activation_factory("Windows.Graphics.Capture.GraphicsCaptureItem")
        statics2 = _qi(f, IID_ItemStatics2)
        _rel(f)
        try:
            user32 = ct.windll.user32
            pt = wt.POINT((monitor_rect.left + monitor_rect.right) // 2,
                          (monitor_rect.top + monitor_rect.bottom) // 2)
            hmon = user32.MonitorFromPoint(pt, 1)   # MONITOR_DEFAULTTONEAREST
            if not hmon:
                raise WGCCaptureError("MonitorFromPoint失败")
            item = c_void_p()
            rc = _vcall(statics2, 7, ct.c_ulong, ct.c_int32(hmon & 0xFFFFFFFF), byref(item),
                        argtypes=(c_void_p, ct.c_int32, POINTER(c_void_p)))
            rc &= 0xFFFFFFFF
            if rc != 0 or not item.value:
                raise WGCCaptureError(f"TryCreateFromDisplayId失败: 0x{rc:08X}")
            self._item = item.value
        finally:
            _rel(statics2)

        # 3. 帧池：Statics2.CreateFreeThreaded（winmd实测：Statics仅Create，Statics2仅CreateFreeThreaded）
        sf = self._activation_factory("Windows.Graphics.Capture.Direct3D11CaptureFramePool")
        statics2 = _qi(sf, IID_FramePoolStatics2)
        _rel(sf)
        try:
            pool = c_void_p()
            size = SizeInt32(w, h)
            rc = _vcall(statics2, 6, ct.c_long, c_void_p(self._device), ct.c_int(87),
                        ct.c_int(SIZE_POOL), size, byref(pool),
                        argtypes=(c_void_p, c_void_p, ct.c_int, ct.c_int, SizeInt32, POINTER(c_void_p)))
            if rc != 0 or not pool.value:
                raise WGCCaptureError(f"CreateFreeThreaded失败: 0x{rc & 0xFFFFFFFF:08X}")
            self._pool = _qi(pool.value, IID_FramePool)
            _rel(pool.value)
        finally:
            _rel(statics2)

        # 4. 会话并启动（FramePool: 6 Recreate, 7 TryGetNextFrame, 8/9 事件, 10 CreateCaptureSession）
        session = c_void_p()
        rc = _vcall(self._pool, 10, ct.c_long, c_void_p(self._item), byref(session),
                    argtypes=(c_void_p, c_void_p, POINTER(c_void_p)))
        if rc != 0 or not session.value:
            raise WGCCaptureError(f"CreateCaptureSession失败: 0x{rc & 0xFFFFFFFF:08X}")
        self._session = _qi(session.value, IID_Session)
        _rel(session.value)
        rc = _vcall(self._session, 6, ct.c_long, argtypes=(c_void_p,))   # StartCapture
        if rc != 0:
            raise WGCCaptureError(f"StartCapture失败: 0x{rc & 0xFFFFFFFF:08X}")

        # 去掉系统捕获提示黄边：QI IGraphicsCaptureSession3 → put_IsBorderRequired(false)
        # （Win10 21H2+/Win11 支持；系统未授权时返回拒绝——静默忽略，黄边只是屏幕
        #   视觉提示，不出现在捕获帧内容里，不影响检测）
        try:
            s3 = _qi(self._session, IID_Session3)
            try:
                rc = _vcall(s3, 7, ct.c_long, ct.c_int(0),
                            argtypes=(c_void_p, ct.c_int))   # put_IsBorderRequired(false)
                if rc != 0:
                    print(f"[捕获] 黄边关闭被系统拒绝: 0x{rc & 0xFFFFFFFF:08X}（不影响检测）")
                else:
                    print("[捕获] 捕获提示黄边已关闭")
            finally:
                _rel(s3)
        except Exception:
            pass   # 旧系统无Session3接口：黄边为系统行为，忽略
        # 池缓冲通常在CreateFreeThreaded时已建齐；若hook未捕获到（延迟创建的系统），
        # 保留hook由grab_gpu在首次成功后懒卸——staging已被回调按Usage过滤不受影响
        if len(self._pool_texs) >= 1:
            self._uninstall_tex_hook()

        # 5. 预览staging（回调按Usage过滤，hook未卸时staging也不会被误捕获）
        self._create_staging()
        self.w, self.h = w, h
        self.ok = True
        print(f"[捕获] WGC就绪：帧池纹理捕获 {len(self._pool_texs)} 个"
              f"{'' if len(self._pool_texs) else '（0个！hook保留懒卸）'}")

    # ---------- CreateTexture2D钩子：捕获帧池缓冲纹理（真设备对象，健康家族） ----------
    def _install_tex_hook(self):
        """替换设备虚表slot5(CreateTexture2D)为回调：记录帧池创建的所有纹理。
        仅在本进程内生效（虚表为同类设备共享，设备创建后WGC建立前安装）。"""
        k32 = ct.windll.kernel32
        k32.VirtualProtect.argtypes = [c_void_p, ct.c_size_t, wt.DWORD, POINTER(wt.DWORD)]
        k32.VirtualProtect.restype = wt.BOOL
        dev = self.pipeline.d3d_device
        vtbl = ct.cast(c_void_p(dev), POINTER(c_void_p)).contents.value
        self._vtbl_addr = vtbl
        slot_addr = c_void_p(vtbl + 5 * 8)
        self._orig_create_tex = ct.cast(
            c_void_p(vtbl + 5 * 8), POINTER(c_void_p)).contents.value
        self._pool_texs = []            # 捕获到的真纹理指针
        self._pool_tex_set = set()
        self._tex_paths = {}            # surface/frame包装 → 池纹理指针的偏移路径缓存（分键）

        def _on_create_tex(this, desc, data, out):
            fn = ct.WINFUNCTYPE(ct.c_long, c_void_p, POINTER(D3D11_TEXTURE2D_DESC),
                                c_void_p, POINTER(c_void_p))(self._orig_create_tex)
            rc = fn(this, desc, data, out)
            try:
                if rc == 0 and desc:
                    d = desc[0]
                    # 只捕获DEFAULT用途纹理（帧池缓冲实测Usage=0）；staging(3)等排除——
                    # 允许hook在池缓冲延迟创建时保留存活，预览staging不会被误捕获
                    # （pipeline.tex创建于attach()，必在hook安装前，不会被误捕）
                    if d.Usage == 0:
                        v = out[0]   # POINTER(c_void_p)取元素直接得到int
                        if v:
                            if v not in self._pool_tex_set:
                                self._pool_tex_set.add(v)
                                self._pool_texs.append(v)
            except Exception:
                pass
            return rc

        self._tex_hook_cb = ct.WINFUNCTYPE(
            ct.c_long, c_void_p, POINTER(D3D11_TEXTURE2D_DESC),
            c_void_p, POINTER(c_void_p))(_on_create_tex)
        self._on_create_tex_impl = _on_create_tex   # 保活闭包

        old = wt.DWORD(0)
        k32.VirtualProtect(slot_addr, 8, 0x40, byref(old))
        hp = c_void_p(ct.cast(self._tex_hook_cb, c_void_p).value)
        ct.memmove(slot_addr, ct.byref(hp), ct.sizeof(c_void_p))
        k32.VirtualProtect(slot_addr, 8, old, byref(old))
        self._hooked = True

    def _uninstall_tex_hook(self):
        if not getattr(self, "_hooked", False):
            return
        k32 = ct.windll.kernel32
        k32.VirtualProtect.argtypes = [c_void_p, ct.c_size_t, wt.DWORD, POINTER(wt.DWORD)]
        slot_addr = c_void_p(self._vtbl_addr + 5 * 8)
        old = wt.DWORD(0)
        if k32.VirtualProtect(slot_addr, 8, 0x40, byref(old)):
            orig = c_void_p(self._orig_create_tex)
            ct.memmove(slot_addr, ct.byref(orig), ct.sizeof(c_void_p))
            k32.VirtualProtect(slot_addr, 8, old, byref(old))
        self._hooked = False

    def _create_staging(self):
        desc = D3D11_TEXTURE2D_DESC(Width=self.pipeline._w, Height=self.pipeline._h,
                                    MipLevels=1, ArraySize=1,
                                    Format=DXGI_FORMAT_B8G8R8A8_UNORM,
                                    SampleDesc=DXGI_SAMPLE_DESC(1, 0),
                                    Usage=D3D11_USAGE_STAGING, BindFlags=0,
                                    CPUAccessFlags=D3D11_CPU_ACCESS_READ, MiscFlags=0)
        out = c_void_p()
        rc = _vcall(self.pipeline.d3d_device, 5, ct.c_long, byref(desc), None, byref(out),
                    argtypes=(c_void_p, POINTER(D3D11_TEXTURE2D_DESC), c_void_p, POINTER(c_void_p)))
        if rc != 0 or not out.value:
            raise WGCCaptureError(f"staging创建失败: 0x{rc & 0xFFFFFFFF:08X}")
        self.staging = out.value
        self.stage_size = (self.pipeline._w, self.pipeline._h)

    def _locate_pool_tex(self, base, path_key=0):
        """在包装对象内存中定位当前帧的池纹理指针（两级间接+路径缓存）
        path_key：surface/frame包装布局可能不同，各自独立缓存路径。
        所有内存读取均通过string_at（AV会转为可捕获异常，无硬崩风险）"""
        k32 = ct.windll.kernel32

        class MBI(ct.Structure):
            _fields_ = [("BaseAddress", c_void_p), ("AllocationBase", c_void_p),
                        ("AllocationProtect", wt.DWORD), ("__a", wt.DWORD),
                        ("RegionSize", ct.c_size_t), ("State", wt.DWORD),
                        ("Protect", wt.DWORD), ("Type", wt.DWORD), ("__b", wt.DWORD)]

        def read_ptr(addr):
            """安全读8字节指针，返回int或None（AV转异常被捕获）"""
            try:
                return struct.unpack_from("<Q", ct.string_at(addr, 8), 0)[0]
            except Exception:
                return None

        def readable(addr, size):
            """校验[addr, addr+size)完整落在同一已提交区域内（防跨页AV）"""
            mbi = MBI()
            if not k32.VirtualQuery(c_void_p(addr), byref(mbi), ct.sizeof(mbi)):
                return False
            if mbi.State != 0x1000:   # MEM_COMMIT
                return False
            return addr + size <= (mbi.BaseAddress or 0) + mbi.RegionSize

        def safe_read_ptr(addr):
            if not readable(addr, 8):
                return None
            return read_ptr(addr)

        # 1) 缓存路径直读（全部走安全读）。
        #    off2==0 表示一级路径：base+off1 处直接存的就是纹理指针，
        #    必须直接读出比对——绝不能再解引用（旧实现把纹理指针当地址二次
        #    读取=读到vtable，永远不在池集合里 → 缓存100%失效，
        #    每帧退化成两次全内存扫描，grab被拖到12ms+，帧率腰斩）
        path = getattr(self, "_tex_paths", {}).get(path_key)
        if path:
            off1, off2 = path
            try:
                if off2 == 0:
                    tex = safe_read_ptr(base + off1)
                else:
                    inner = safe_read_ptr(base + off1)
                    tex = safe_read_ptr(inner + off2) if inner else None
                if tex in self._pool_tex_set:
                    return tex
            except Exception:
                pass
            self._tex_paths[path_key] = None

        # 2) 全扫描（两级，全部走安全读）
        self._scan_seen = set()

        def scan(addr, depth):
            if addr in self._scan_seen:
                return None
            self._scan_seen.add(addr)
            if not readable(addr, 0x300):
                return None
            try:
                buf = ct.string_at(addr, 0x300)
            except Exception:
                return None
            for off in range(0, 0x300 - 8, 8):
                v = int.from_bytes(buf[off:off + 8], "little")
                if v in self._pool_tex_set:
                    return ("direct", addr, off, v)
                if depth > 0 and 0x10000 < v < 0x7FFFFFFF0000:
                    r = scan(v, depth - 1)
                    if r:
                        return r
            return None

        r = scan(base, 2)
        if r:
            kind, addr, off, tex = r
            if kind == "direct":
                self._tex_paths[path_key] = (off, 0)   # 一级路径
            else:
                # 记录二级路径：addr为base+off1处的inner，off为inner内偏移
                for off1 in range(0, 0x300 - 8, 8):
                    v1 = safe_read_ptr(base + off1)
                    if v1 == addr:
                        self._tex_paths[path_key] = (off1, off)
                        break
            return tex
        return None

    def grab_gpu(self):
        """取新帧 -> 在包装对象内存中定位当前帧的池纹理 -> GPU侧拷贝到管线常驻纹理。
        返回True=有新帧"""
        if not self.ok:
            return False
        frame = c_void_p()
        surface = None
        tex = None
        try:
            rc = _vcall(self._pool, 7, ct.c_long, byref(frame),   # TryGetNextFrame
                        argtypes=(c_void_p, POINTER(c_void_p)))
            rc &= 0xFFFFFFFF
            self.last_grab_rc = rc
            if rc != 0 or not frame.value:
                if rc != 0:
                    self.last_stage = "trynext_fail"   # 无新帧是S_OK+null；rc非零=池异常
                return False
            self.last_stage = "get_surface"
            surface = c_void_p()
            rc = _vcall(frame.value, 6, ct.c_long, byref(surface),   # get_Surface
                        argtypes=(c_void_p, POINTER(c_void_p)))
            if rc != 0 or not surface.value:
                return False
            self.last_stage = "locate_tex"
            # 在surface/frame包装对象内存中定位当前帧的池纹理指针（纯读取零风险）。
            # 本机系统包装层的QI/GetInterface不可用（返回自身/IIDs数组），因此
            # 通过设备CreateTexture2D钩子直接拿到帧池缓冲纹理（真设备对象）。
            # 包装层可能二级间接引用纹理（包装→内部→纹理），支持两级扫描+路径缓存。
            tex = None
            # frame包装在前：实测frame+0直接持有纹理指针（缓存命中<1μs）；
            # surface包装在本机扫不到纹理（每次全扫描~9ms白烧），仅作兜底
            for path_key, base in enumerate((frame.value, surface.value)):
                tex = self._locate_pool_tex(base, path_key)
                if tex:
                    break
            if not tex:
                return False
            self.last_stage = "copy"
            dw, dh = self.w, self.h
            pw, ph = self.pipeline._w, self.pipeline._h
            if dw == pw and dh == ph:
                _vcall(self.pipeline.d3d_ctx, 47, None, c_void_p(self.pipeline.tex),
                       c_void_p(tex),
                       argtypes=(c_void_p, c_void_p, c_void_p))   # CopyResource
            else:
                # 帧池尺寸≠管线尺寸：中心裁剪
                bx = max(0, min((dw - pw) // 2, dw - pw))
                by = max(0, min((dh - ph) // 2, dh - ph))

                class BOX(ct.Structure):
                    _fields_ = [("left", wt.UINT), ("top", wt.UINT),
                                ("front", wt.UINT), ("right", wt.UINT),
                                ("bottom", wt.UINT), ("back", wt.UINT)]
                box = BOX(bx, by, 0, bx + pw, by + ph, 1)
                _vcall(self.pipeline.d3d_ctx, 46, None, c_void_p(self.pipeline.tex),
                       wt.UINT(0), wt.UINT(0), wt.UINT(0), wt.UINT(0),
                       c_void_p(tex), wt.UINT(0), byref(box),
                       argtypes=(c_void_p, c_void_p, wt.UINT, wt.UINT, wt.UINT,
                                 wt.UINT, c_void_p, wt.UINT, POINTER(BOX)))
            self.last_stage = "done"
            # 首次成功后懒卸hook（池缓冲延迟创建的系统在此时才捕获到纹理）
            if getattr(self, "_hooked", False):
                self._uninstall_tex_hook()
            # 帧释放放在成功路径（实测安全）；失败路径不释放（有限泄漏，池自动回收）
            if frame.value:
                try:
                    _rel(frame.value)
                except Exception:
                    pass
                frame.value = None
            return True
        except OSError as e:
            self.last_stage = f"error:{e}"
            return False
        finally:
            # 失败路径不释放frame（本机包装层对未消费帧的Release会硬崩溃）
            if surface and surface.value:
                try:
                    _rel(surface.value)
                except Exception:
                    pass

    def readback(self):
        """预览：管线纹理 -> staging -> BGR ndarray"""
        if not self.ok or not self.pipeline.tex:
            return None
        pw, ph = self.pipeline._w, self.pipeline._h
        _vcall(self.pipeline.d3d_ctx, 47, None, c_void_p(self.staging), c_void_p(self.pipeline.tex),
               argtypes=(c_void_p, c_void_p, c_void_p))   # CopyResource
        msub = D3D11_MAPPED_SUBRESOURCE()
        rc = _vcall(self.pipeline.d3d_ctx, 14, ct.c_ulong, c_void_p(self.staging), wt.UINT(0),
                    wt.UINT(1), wt.UINT(0), byref(msub),
                    argtypes=(c_void_p, c_void_p, wt.UINT, wt.UINT, wt.UINT,
                              POINTER(D3D11_MAPPED_SUBRESOURCE)))
        if rc != 0 or not msub.pData:
            return None
        try:
            buf = ct.string_at(msub.pData, ph * msub.RowPitch)
            arr = np.frombuffer(buf, np.uint8).reshape(ph, msub.RowPitch // 4, 4)
            return np.ascontiguousarray(arr[:, :pw, :3])
        finally:
            _vcall(self.pipeline.d3d_ctx, 15, None, c_void_p(self.staging), wt.UINT(0), wt.UINT(0),
                   argtypes=(c_void_p, c_void_p, wt.UINT, wt.UINT))

    def close(self):
        self.ok = False
        self._uninstall_tex_hook()
        # Close走IClosable（winmd：Session/FramePool本体无Close方法）
        for obj in (self._session, self._pool):
            if not obj:
                continue
            try:
                closable = _qi(obj, IID_IClosable)
                _vcall(closable, 6, ct.c_long, argtypes=(c_void_p,))   # Close
                _rel(closable)
            except Exception:
                pass
            _rel(obj)
        self._session = None
        self._pool = None
        if self._item:
            _rel(self._item)
            self._item = None
        if self._device:
            _rel(self._device)
            self._device = None
        if self.staging:
            _rel(self.staging)
            self.staging = 0
