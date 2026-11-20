# -*- coding: utf-8 -*-
"""DXGI 桌面复制引擎 GPU 捕获（手动COM实现，raw指针+手动引用计数）
两种模式：
  GPU直通零拷贝（ext_device传入，gpu_mode=True）：
    Acquire -> CopySubresourceRegion到GPUPipeline常驻纹理 -> Release，全程不落CPU内存
  CPU读回（兼容/预览）：staging纹理Map回传numpy
所有 COM 调用通过虚表槽位手动调用（已验证槽位），避免 comtypes wrapper 崩溃。"""
import ctypes
import ctypes as ct
import ctypes.wintypes as wt
from ctypes import POINTER, byref, WINFUNCTYPE, c_void_p
import uuid
import numpy as np
import comtypes
from comtypes import HRESULT

D3D11_CREATE_DEVICE_BGRA_SUPPORT = 0x20
D3D11_USAGE_STAGING = 3
D3D11_CPU_ACCESS_READ = 0x20000
DXGI_FORMAT_B8G8R8A8_UNORM = 87
DXGI_ERROR_WAIT_TIMEOUT = 0x887A0027
IID_IDXGIDevice = uuid.UUID("{54EC77FA-1377-44E6-8C32-88FD5F44C84C}").bytes_le
IID_IDXGIAdapter = uuid.UUID("{2411E7E1-12AC-4CCF-BD14-253E83A1E38B}").bytes_le
IID_IDXGIOutput = uuid.UUID("{AE02EEDB-C735-4690-8D52-5A8DC20213AA}").bytes_le
IID_IDXGIOutput1 = uuid.UUID("{00CDDEA8-939B-4B83-A340-A685226666CC}").bytes_le
IID_IDXGIOutputDuplication = uuid.UUID("{191CFAC3-A341-470D-B26E-A864F37C3683}").bytes_le
IID_ID3D11Texture2D = uuid.UUID("{6F15AAF2-D208-4E89-9AB4-489535D34F9C}").bytes_le
IID_IDXGISurface = uuid.UUID("{CAFCB56C-6AC3-4889-BF47-9E23BBD260EC}").bytes_le


class DXGI_SAMPLE_DESC(ct.Structure):
    _fields_ = [("Count", wt.UINT), ("Quality", wt.UINT)]
class D3D11_BOX(ct.Structure):
    _fields_ = [("left", wt.UINT), ("top", wt.UINT), ("front", wt.UINT),
                ("right", wt.UINT), ("bottom", wt.UINT), ("back", wt.UINT)]
class D3D11_TEXTURE2D_DESC(ct.Structure):
    _fields_ = [("Width", wt.UINT), ("Height", wt.UINT), ("MipLevels", wt.UINT),
                ("ArraySize", wt.UINT), ("Format", wt.UINT), ("SampleDesc", DXGI_SAMPLE_DESC),
                ("Usage", wt.UINT), ("BindFlags", wt.UINT),
                ("CPUAccessFlags", wt.UINT), ("MiscFlags", wt.UINT)]
class D3D11_MAPPED_SUBRESOURCE(ct.Structure):
    _fields_ = [("pData", ct.c_void_p), ("RowPitch", wt.UINT), ("DepthPitch", wt.UINT)]
class DXGI_OUTPUT_DESC(ct.Structure):
    _fields_ = [("DeviceName", wt.WCHAR * 32), ("DesktopCoordinates", wt.RECT),
                ("AttachedToDesktop", wt.BOOL), ("Rotation", wt.UINT), ("Monitor", wt.HMONITOR)]
class DXGI_OUTDUPL_POINTER_POSITION(ctypes.Structure):
    _fields_ = [("Position", wt.POINT), ("Visible", wt.BOOL)]
class DXGI_OUTDUPL_FRAME_INFO(ctypes.Structure):
    _fields_ = [("LastPresentTime", wt.LARGE_INTEGER), ("LastMouseUpdateTime", wt.LARGE_INTEGER),
                ("AccumulatedFrames", wt.UINT), ("RectsCoalesced", wt.BOOL),
                ("ProtectedContentMaskedOut", wt.BOOL), ("PointerPosition", DXGI_OUTDUPL_POINTER_POSITION),
                ("TotalMetadataBufferSize", wt.UINT), ("PointerShapeBufferSize", wt.UINT)]
class DXGI_OUTDUPL_DESC(ctypes.Structure):
    _fields_ = [("Width", wt.UINT), ("Height", wt.UINT), ("RefreshNum", wt.UINT),
                ("RefreshDen", wt.UINT), ("Format", wt.UINT), ("Scaling", wt.UINT),
                ("Rotation", wt.UINT), ("RotationDup", wt.UINT),
                ("DesktopImageInSystemMemory", wt.UINT)]
class DXGI_MAPPED_RECT(ctypes.Structure):
    _fields_ = [("Pitch", wt.INT), ("pBits", c_void_p)]
def _new_guid(iid):
    return ctypes.create_string_buffer(iid)
class DXGICapture:
    """桌面复制捕获器。ext_device传入时为零拷贝模式（grab_gpu），否则CPU回退模式(grab)"""
    def __init__(self, region_w, region_h, device_name=None,
                 ext_device=None, ext_ctx=None, pipeline=None):
        self.w, self.h = region_w, region_h
        self.device_name = device_name
        self.ox = 0
        self.oy = 0
        self.ok = False
        self.gpu_mode = ext_device is not None
        self.pipeline = pipeline
        self.ext_device = ext_device
        self.ext_ctx = ext_ctx
        self.device = 0
        self.ctx = 0
        self.dup = 0
        self.staging = 0
        self.surface = 0
        self.stage_size = None
        self.dup_size = (region_w, region_h)   # 桌面全屏尺寸（dup GetDesc 缓存）
        self.output_desc = None
        self.region = {"left": 0, "top": 0, "width": region_w, "height": region_h}
        self._base = {"left": 0, "top": 0, "width": 0, "height": 0}
        self._fn_cache = {}
        self.last_hr = 0
        self.last_stage = "-"
        self._open()

    # ---------- COM虚表直调 ----------
    def _fn(self, addr, slot, restype, *argtypes):
        if restype is HRESULT:
            restype = ct.c_ulong
        key = (addr, slot, restype, argtypes)
        fn = self._fn_cache.get(key)
        if fn is None:
            vtbl = ctypes.cast(addr, POINTER(c_void_p)).contents.value
            fp = ctypes.cast(vtbl, POINTER(c_void_p))[slot]
            fn = WINFUNCTYPE(restype, c_void_p, *argtypes)(fp)
            self._fn_cache[key] = fn
        return fn
    @staticmethod
    def _addr(ptr):
        return ctypes.cast(ptr, POINTER(c_void_p)).contents.value
    def _release(self, addr):
        if addr:
            self._fn(addr, 2, HRESULT)(addr)
    def _qi(self, addr, iid):
        out = c_void_p()
        buf = _new_guid(iid)
        hr = self._fn(addr, 0, HRESULT, POINTER(ctypes.c_char), POINTER(c_void_p))(addr, buf, byref(out))
        if hr != 0:
            raise OSError(f"QueryInterface失败: 0x{hr & 0xFFFFFFFF:08X}")
        return out.value

    @staticmethod
    def _primary_monitor_name():
        user32 = ctypes.windll.user32
        pt = wt.POINT(0, 0)
        hmon = user32.MonitorFromPoint(pt, 1)
        class MONITORINFOEXW(ctypes.Structure):
            _fields_ = [("cbSize", wt.DWORD), ("rcMonitor", wt.RECT),
                        ("rcWork", wt.RECT), ("dwFlags", wt.DWORD),
                        ("szDevice", wt.WCHAR * 32)]
        miex = MONITORINFOEXW()
        miex.cbSize = ctypes.sizeof(miex)
        if user32.GetMonitorInfoW(hmon, ctypes.byref(miex)):
            return miex.szDevice
        return None

    def _open(self):
        try:
            comtypes.CoInitializeEx(comtypes.COINIT_APARTMENTTHREADED)
        except Exception:
            pass
        if self.gpu_mode:
            device = c_void_p(int(self.ext_device))
            ctx = c_void_p(int(self.ext_ctx))
            dxgi_dev = self._qi(int(self.ext_device), IID_IDXGIDevice)
            adapter = c_void_p()
            hr = self._fn(dxgi_dev, 7, HRESULT, POINTER(c_void_p))(dxgi_dev, byref(adapter))
            self._release(dxgi_dev)
            if hr != 0:
                raise OSError("获取显卡适配器失败")
        else:
            d3d11 = ctypes.WinDLL("d3d11.dll")
            device = c_void_p()
            ctx = c_void_p()
            feature = wt.INT()
            hr = d3d11.D3D11CreateDevice(None, 1, None, D3D11_CREATE_DEVICE_BGRA_SUPPORT,
                                         None, 0, 7, byref(device), byref(feature), byref(ctx))
            if hr != 0:
                raise OSError(f"D3D11设备创建失败: 0x{hr & 0xFFFFFFFF:08X}")
            self.device, self.ctx = device.value, ctx.value
            dxgi_dev = self._qi(self.device, IID_IDXGIDevice)
            adapter = c_void_p()
            hr = self._fn(dxgi_dev, 7, HRESULT, POINTER(c_void_p))(dxgi_dev, byref(adapter))
            self._release(dxgi_dev)
            if hr != 0:
                raise OSError("获取显卡适配器失败")
        self.device, self.ctx = device.value, ctx.value

        primary_name = self._primary_monitor_name()
        desc = DXGI_OUTPUT_DESC()
        output = 0
        i = 0
        while True:
            out = c_void_p()
            hr = self._fn(adapter.value, 7, HRESULT, wt.UINT, POINTER(c_void_p))(
                adapter.value, i, byref(out))
            if hr != 0:
                break
            hr = self._fn(out.value, 7, HRESULT, POINTER(DXGI_OUTPUT_DESC))(out.value, byref(desc))
            if hr == 0 and desc.AttachedToDesktop:
                matched = ((self.device_name and desc.DeviceName == self.device_name) or
                           (not self.device_name and
                            (primary_name is None or desc.DeviceName == primary_name)))
                if matched:
                    output = self._qi(out.value, IID_IDXGIOutput1)
                    self._release(out.value)
                    break
            self._release(out.value)
            i += 1
        if not output:
            raise OSError("未找到可用的显示输出")
        self.output_desc = dict(left=desc.DesktopCoordinates.left, top=desc.DesktopCoordinates.top,
                                width=desc.DesktopCoordinates.right - desc.DesktopCoordinates.left,
                                height=desc.DesktopCoordinates.bottom - desc.DesktopCoordinates.top)
        dup = c_void_p()
        hr = self._fn(output, 22, HRESULT, c_void_p, POINTER(c_void_p))(output, self.device, byref(dup))
        self._release(output)
        if hr != 0:
            raise OSError(f"桌面复制初始化失败: 0x{hr & 0xFFFFFFFF:08X}")
        self.dup = dup.value
        # 桌面全屏尺寸（替代对帧资源QI取desc——本机帧资源QI不可用）
        dd = DXGI_OUTDUPL_DESC()
        hr = self._fn(self.dup, 7, HRESULT, POINTER(DXGI_OUTDUPL_DESC))(
            c_void_p(self.dup), byref(dd))
        if hr != 0 or dd.Width == 0:
            raise OSError(f"桌面复制描述获取失败: 0x{hr & 0xFFFFFFFF:08X}")
        self.dup_size = (dd.Width, dd.Height)
        # staging纹理（preview读回用；CPU模式下为主读回通道）
        self._create_staging(min(self.w, 65536), min(self.h, 65536))
        self.ok = True

    def _create_staging(self, w, h):
        desc = D3D11_TEXTURE2D_DESC(Width=w, Height=h, MipLevels=1, ArraySize=1,
                                    Format=DXGI_FORMAT_B8G8R8A8_UNORM,
                                    SampleDesc=DXGI_SAMPLE_DESC(1, 0),
                                    Usage=D3D11_USAGE_STAGING, BindFlags=0,
                                    CPUAccessFlags=D3D11_CPU_ACCESS_READ, MiscFlags=0)
        out = c_void_p()
        hr = self._fn(self.device, 3 + 2, HRESULT, POINTER(D3D11_TEXTURE2D_DESC), c_void_p,
                      POINTER(c_void_p))(self.device, byref(desc), None, byref(out))
        if hr != 0 or not out.value:
            raise OSError(f"暂存纹理创建失败: 0x{hr & 0xFFFFFFFF:08X}")
        self.staging = out.value
        self.stage_size = (w, h)
    def _release_staging(self):
        self._release(self.staging)
        self.staging = 0
        self.stage_size = None
    def _reopen(self):
        try:
            self.close()
        except Exception:
            pass
        try:
            self._open()
        except Exception:
            self.ok = False

    def _acquire(self):
        """AcquireNextFrame + QI帧资源为ID3D11Texture2D。返回 (tex, resource) 或 None（无新帧）"""
        self.last_hr = 0
        self.last_stage = "acquire"
        dup = self.dup
        info = DXGI_OUTDUPL_FRAME_INFO()
        resource = c_void_p()
        hr = self._fn(dup, 8, HRESULT, wt.UINT, POINTER(DXGI_OUTDUPL_FRAME_INFO),
                      POINTER(c_void_p))(c_void_p(dup), 0, byref(info), byref(resource))
        self.last_hr = hr
        if hr != 0 or not resource.value:
            self._release(resource.value)
            if hr != DXGI_ERROR_WAIT_TIMEOUT:
                self._reopen()
            return None
        try:
            self.last_stage = "qi_texture"
            tex = c_void_p()
            hr = self._fn(resource.value, 0, HRESULT, POINTER(ctypes.c_char),
                          POINTER(c_void_p))(resource.value, _new_guid(IID_ID3D11Texture2D), byref(tex))
            if hr != 0 or not tex.value:
                self.last_hr = hr
                return None
            self.last_stage = "device_removed"
            # GetDeviceRemovedReason = ID3D11Device槽39（0-2 IUnknown, 3-27 Create*, 28 OpenSharedResource,
            # 29 CheckFormatSupport, 33 CheckFeatureSupport, 34 GetPrivateData, 37 GetFeatureLevel, 39 GetDeviceRemovedReason）
            drr = self._fn(self.device, 39, HRESULT)(c_void_p(self.device))
            if drr != 0:
                self.last_hr = drr
                return None
            return tex.value, resource.value
        finally:
            if not tex.value:
                self._release(tex.value)
                self._release(resource.value)

    def _copy_box(self, src):
        """把src帧资源中心区域拷贝到staging（CPU读回用）"""
        dw, dh = self.dup_size
        sw, sh = min(self.w, dw), min(self.h, dh)
        if not self.staging or self.stage_size != (sw, sh):
            self._release_staging()
            self._create_staging(sw, sh)
        self.last_stage = "copy"
        bx = max(0, min((dw - sw) // 2 + self.ox, dw - sw))
        by = max(0, min((dh - sh) // 2 + self.oy, dh - sh))
        box = D3D11_BOX(left=bx, top=by, right=bx + sw, bottom=by + sh, front=0, back=1)
        self._fn(self.ctx, 46, None, c_void_p, wt.UINT, wt.UINT, wt.UINT,
                 wt.UINT, c_void_p, wt.UINT, POINTER(D3D11_BOX))(
            self.ctx, c_void_p(self.staging), 0, 0, 0, 0, c_void_p(src), 0, byref(box))

    def grab_gpu(self):
        """零拷贝：Acquire -> QI纹理 -> 拷贝中心区域到GPUPipeline常驻纹理 -> Release。
        返回 True=有新帧已进入管线纹理 / False=无新帧"""
        if not self.ok:
            return False
        if getattr(self, "neutered", False):
            return False
        tex = resource = None
        try:
            acq = self._acquire()
            if acq is None:
                return False
            tex, resource = acq
            dw, dh = self.dup_size
            self.last_stage = "copy_gpu"
            bx = max(0, min((dw - self.w) // 2 + self.ox, dw - self.w))
            by = max(0, min((dh - self.h) // 2 + self.oy, dh - self.h))
            box = D3D11_BOX(left=bx, top=by, right=bx + self.w, bottom=by + self.h, front=0, back=1)
            dst_tex = ct.c_void_p(self.pipeline.tex)
            self._fn(self.ctx, 46, None, c_void_p, wt.UINT, wt.UINT, wt.UINT,
                     wt.UINT, c_void_p, wt.UINT, POINTER(D3D11_BOX))(
                self.ctx, dst_tex, 0, 0, 0, 0, c_void_p(tex), 0, byref(box))
            self.last_stage = "release_frame"
            self._fn(self.dup, 14, None)(c_void_p(self.dup))
            return True
        except OSError as e:
            self.neutered = True
            self.last_stage = f"error:{e}"
            return False
        finally:
            if tex:
                try:
                    self._release(tex)
                except Exception:
                    pass
            if resource:
                try:
                    self._release(resource)
                except Exception:
                    pass

    def readback(self):
        """预览用：把管线常驻纹理回读到CPU（BGR ndarray）。仅在preview开启时调用"""
        if not self.ok or not self.pipeline or not self.pipeline.tex:
            return None
        self.last_stage = "copy_preview"
        self._fn(self.ctx, 47, None, c_void_p, c_void_p)(
            self.ctx, ct.c_void_p(self.staging), ct.c_void_p(self.pipeline.tex))
        self.last_stage = "map"
        msub = D3D11_MAPPED_SUBRESOURCE()
        rc = self._fn(self.ctx, 14, HRESULT, c_void_p, wt.UINT, wt.UINT, wt.UINT,
                      POINTER(D3D11_MAPPED_SUBRESOURCE))(
            self.ctx, ct.c_void_p(self.staging), 0, 1, 0, byref(msub))
        if rc != 0 or not msub.pData:
            return None
        try:
            self.last_stage = "read"
            h = self.stage_size[1]; w = self.stage_size[0]
            buf = ctypes.string_at(msub.pData, h * msub.RowPitch)
            arr = np.frombuffer(buf, np.uint8).reshape(h, msub.RowPitch // 4, 4)
            return np.ascontiguousarray(arr[:, :w, :3])
        finally:
            self.last_stage = "unmap"
            self._fn(self.ctx, 15, None, c_void_p, wt.UINT, wt.UINT)(
                self.ctx, ct.c_void_p(self.staging), 0, 0)

    def grab(self):
        """CPU读回模式（兼容保留）。返回BGR ndarray或None"""
        if not self.ok:
            return None
        acq = self._acquire()
        if acq is None:
            return None
        tex, resource = acq
        try:
            self._copy_box(tex)
            self.last_stage = "release_frame"
            self._fn(self.dup, 14, None)(c_void_p(self.dup))
            self.last_stage = "map"
            msub = D3D11_MAPPED_SUBRESOURCE()
            rc = self._fn(self.ctx, 14, HRESULT, c_void_p, wt.UINT, wt.UINT, wt.UINT,
                          POINTER(D3D11_MAPPED_SUBRESOURCE))(
                self.ctx, ct.c_void_p(self.staging), 0, 1, 0, byref(msub))
            if rc != 0 or not msub.pData:
                return None
            self.last_stage = "read"
            h = self.stage_size[1]; w = self.stage_size[0]
            buf = ctypes.string_at(msub.pData, h * msub.RowPitch)
            self.last_stage = "unmap"
            self._fn(self.ctx, 15, None, c_void_p, wt.UINT, wt.UINT)(
                self.ctx, ct.c_void_p(self.staging), 0, 0)
            self.last_stage = "done"
            arr = np.frombuffer(buf, np.uint8).reshape(h, msub.RowPitch // 4, 4)
            return np.ascontiguousarray(arr[:, :w, :3])
        finally:
            self._release(tex)
            self._release(resource)

    def close(self):
        self.ok = False
        self._release_staging()
        self._release(self.dup)
        self.dup = 0
        self._release(self.ctx)
        self.ctx = 0
        if not self.gpu_mode:
            self._release(self.device)
        self.device = 0
        self._fn_cache = {}
