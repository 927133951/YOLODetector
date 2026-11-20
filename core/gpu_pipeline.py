# -*- coding: utf-8 -*-
"""GPU直通零拷贝管线：D3D11纹理 -> CUDA -> NVRTC预处理 -> ORT IOBinding(TRT FP16)

时序契约（PoC实测验证）：
  prewarm()(cudaD3D11SetDirect3DDevice) 必须在 CUDA 主上下文被激活之前调用——
  即引擎初始化顺序必须是：管线预热 -> 加载ORT/TRT会话 -> 捕获线程注册纹理。

失败策略（用户决策）：零拷贝是唯一路径，本模块初始化失败直接抛错、拒绝启动，
绝不静默回退CPU。混合显卡（核显驱动屏幕）属于物理不可行配置，明确报错。
"""
import os
import sys
import glob
import ctypes as ct
import uuid

import numpy as np


def _prep_dll_dirs():
    """预载 pip nvidia wheels 的 DLL 目录（cudart/nvrtc/cublas/tensorrt）"""
    bases = [os.path.join(os.path.dirname(sys.executable), "Lib", "site-packages")]
    if getattr(sys, "frozen", False) and getattr(sys, "_MEIPASS", None):
        bases.insert(0, sys._MEIPASS)  # PyInstaller打包：DLL解压在_internal下
    dirs = []
    for base in bases:
        dirs += glob.glob(os.path.join(base, "nvidia", "*", "bin")) + \
            glob.glob(os.path.join(base, "nvidia", "cublas", "lib")) + \
            [os.path.join(base, "tensorrt_libs"), os.path.join(base, "tensorrt")]
    for d in dirs:
        if os.path.isdir(d):
            try:
                os.add_dll_directory(d)
                os.environ["PATH"] = d + os.pathsep + os.environ.get("PATH", "")
            except OSError:
                pass


_prep_dll_dirs()

IID_IDXGIFactory1 = uuid.UUID("{770aae78-f26f-4dba-a429-253e83a1e38b}").bytes_le
DXGI_FORMAT_B8G8R8A8_UNORM = 87
D3D11_USAGE_DEFAULT = 0
D3D11_CPU_ACCESS_READ = 0x20000


class GPUPipelineError(RuntimeError):
    """GPU直通管线不可用——按用户决策：报错拒绝启动，不回退CPU"""


class _D3DHelper:
    """COM vtable 直调辅助（模式与 dxgi_capture 一致，已生产验证）"""

    @staticmethod
    def vfn(addr, slot, restype, *argtypes):
        fp = ct.cast(ct.cast(addr, ct.POINTER(ct.c_void_p)).contents.value,
                     ct.POINTER(ct.c_void_p))[slot]
        return ct.WINFUNCTYPE(restype, ct.c_void_p, *argtypes)(fp)

    @staticmethod
    def qi(addr, iid):
        out = ct.c_void_p()
        rc = _D3DHelper.vfn(addr, 0, ct.c_long, ct.c_char_p,
                            ct.POINTER(ct.c_void_p))(addr, iid, ct.byref(out))
        if rc != 0:
            raise OSError(f"QueryInterface失败 0x{rc & 0xFFFFFFFF:08X}")
        return out

    @staticmethod
    def rel(addr):
        if addr:
            _D3DHelper.vfn(addr, 2, ct.c_long)(addr)


class TEXDESC(ct.Structure):
    _fields_ = [("Width", ct.c_uint), ("Height", ct.c_uint), ("MipLevels", ct.c_uint),
                ("ArraySize", ct.c_uint), ("Format", ct.c_uint),
                ("SampleDesc_Count", ct.c_uint), ("SampleDesc_Quality", ct.c_uint),
                ("Usage", ct.c_uint), ("BindFlags", ct.c_uint),
                ("CPUAccessFlags", ct.c_uint), ("MiscFlags", ct.c_uint)]


class GPUPipeline:
    """持有 D3D11 设备（创建于预热期）、常驻 BGRA 纹理与 CUDA 注册/预处理内核"""

    def __init__(self):
        self.ok = False
        self.d3d_device = None
        self.d3d_ctx = None
        self.cuda_ordinal = 0
        self.adapter_outputs = []    # 本适配器拥有的显示器设备名（用于显示器切换校验）
        self.blob_ptr = None         # 预处理输出（float32 NCHW RGB/255）显存指针
        self.mode_desc = "GPU直通管线 (GDI捕获+CUDA预处理)"
        self._cudart = None
        self._tex = None
        self._res = None
        self._registered = False
        self._w = 0
        self._h = 0
        self._stream = None
        self._kernel = None
        self._linear = None
        self._blob = None
        self._mapped = False
        self._launch = None
        self._lc = None
        self._rt = None

    # ---------- 预热（引擎主线程，必须先于ORT/TRT） ----------
    def prewarm(self, monitor=None):
        import ctypes.wintypes as wt
        try:
            import comtypes
            try:
                comtypes.CoInitializeEx(comtypes.COINIT_APARTMENTTHREADED)
            except Exception:
                pass
        except Exception:
            pass
        # 不用CreateDXGIFactory1（实测本机E_NOINTERFACE）——走引擎验证路径：
        # D3D11CreateDevice默认适配器 -> QI IDXGIDevice -> GetAdapter -> 枚举输出
        d3d11 = ct.WinDLL("d3d11.dll")
        device = ct.c_void_p(); ctx = ct.c_void_p()
        feature = wt.INT()
        hr = d3d11.D3D11CreateDevice(None, 1, None, 0x20, None, 0, 7,
                                     ct.byref(device), ct.byref(feature), ct.byref(ctx))
        if hr != 0:
            raise GPUPipelineError(f"D3D11CreateDevice失败: 0x{hr & 0xFFFFFFFF:08X}")

        def vfn(addr, slot, restype, *argtypes):
            fp = ct.cast(ct.cast(addr, ct.POINTER(ct.c_void_p)).contents.value,
                         ct.POINTER(ct.c_void_p))[slot]
            return ct.WINFUNCTYPE(restype, ct.c_void_p, *argtypes)(fp)

        IID_IDXGIDevice = uuid.UUID("{54ec77fa-1377-44e6-8c32-88fd5f44c84c}").bytes_le
        dxgi_dev = _D3DHelper.qi(device.value, IID_IDXGIDevice)
        adapter = ct.c_void_p()
        rc = vfn(dxgi_dev, 7, ct.c_long, ct.POINTER(ct.c_void_p))(dxgi_dev, ct.byref(adapter))
        _D3DHelper.rel(dxgi_dev)
        if rc != 0:
            raise GPUPipelineError("获取显卡适配器失败")

        class OUTDESC(ct.Structure):
            _fields_ = [("DeviceName", wt.WCHAR * 32), ("DesktopCoordinates", wt.RECT),
                        ("AttachedToDesktop", wt.BOOL), ("Rotation", wt.UINT),
                        ("Monitor", wt.HMONITOR)]
        target = monitor or None
        found = False
        i = 0
        while True:
            out = ct.c_void_p()
            rc = vfn(adapter.value, 7, ct.c_long, wt.UINT,
                     ct.POINTER(ct.c_void_p))(adapter.value, i, ct.byref(out))
            if rc != 0:
                break
            d = OUTDESC()
            if vfn(out, 7, ct.c_long, ct.POINTER(OUTDESC))(out, ct.byref(d)) == 0:
                self.adapter_outputs.append(d.DeviceName)
                if (target and d.DeviceName == target) or \
                        (not target and d.AttachedToDesktop and not found):
                    found = True
            _D3DHelper.rel(out)
            i += 1
        self.adapter_outputs = list(dict.fromkeys(self.adapter_outputs))
        if not found:
            raise GPUPipelineError(
                f"目标显示器 {monitor or '主显示器'} 不在本GPU适配器输出上"
                "（混合显卡/多适配器系统）——CUDA无法接管其他适配器的纹理。"
                "请将游戏设置在NVIDIA直连输出的显示器上")
        self.d3d_device, self.d3d_ctx = device.value, ctx.value

        # CUDA与D3D适配器关联（必须在CUDA主上下文激活之前）
        cudart = ct.CDLL("cudart64_12.dll")
        gdev = cudart.cudaD3D11GetDevice
        gdev.argtypes = [ct.POINTER(ct.c_int), ct.c_void_p]
        gdev.restype = ct.c_int
        ordinal = ct.c_int(-1)
        rc = gdev(ct.byref(ordinal), adapter.value)
        if rc != 0 or ordinal.value < 0:
            raise GPUPipelineError(
                "目标显示器适配器不是NVIDIA GPU（混合显卡/核显输出屏幕）——"
                "CUDA无法接管该适配器纹理。请将游戏设置在NVIDIA直连输出的显示器上")
        self.cuda_ordinal = ordinal.value
        sdd = cudart.cudaD3D11SetDirect3DDevice
        sdd.argtypes = [ct.c_void_p, ct.c_int]
        sdd.restype = ct.c_int
        rc = sdd(self.d3d_device, 0)
        if rc != 0:
            raise GPUPipelineError(f"cudaD3D11SetDirect3DDevice失败: cudaError {rc}")
        self._cudart = cudart
        self.ok = True

    # ---------- CUDA侧惰性初始化（主上下文激活后任意线程可调） ----------
    def _ensure_cuda(self):
        if self._stream is not None:
            return
        from cuda.core import Device, Program, ProgramOptions
        d = Device(self.cuda_ordinal)
        d.set_current()
        self._stream = d.create_stream()
        self._linear = d.allocate(self._w * self._h * 4, stream=self._stream)
        self._blob = d.allocate(3 * self._h * self._w * 4, stream=self._stream)
        self.blob_ptr = int(self._blob.handle)
        cc = getattr(d, "compute_capability", (8, 6))
        arch = f"compute_{cc[0] * 10 + cc[1]}" if isinstance(cc, (tuple, list)) else f"compute_{cc}"
        src = f'''
extern "C" __global__ void bgra_to_rgb_nchw(const uchar4* src, float* dst) {{
    const int W = {self._w}, H = {self._h}, pitchBytes = {self._w * 4};
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= W * H) return;
    int x = i % W, y = i / W;
    const uchar4* row = (const uchar4*)((const char*)src + y * pitchBytes);
    uchar4 p = row[x];
    dst[(0 * H + y) * W + x] = p.z / 255.0f;   // R
    dst[(1 * H + y) * W + x] = p.y / 255.0f;   // G
    dst[(2 * H + y) * W + x] = p.x / 255.0f;   // B
}}
'''
        prog = Program(src, code_type="c++", options=ProgramOptions(arch=arch))
        mod = prog.compile("ptx")
        self._kernel = mod.get_kernel("bgra_to_rgb_nchw")
        # 每帧热路径的Python引用全部预缓存（避免每帧import查找/LaunchConfig重建）
        from cuda.core import launch, LaunchConfig
        from cuda.bindings import runtime
        self._launch = launch
        self._lc = LaunchConfig(grid=(self._w * self._h) // 256, block=256)
        self._rt = runtime

    # ---------- 捕获线程：创建/注册常驻纹理 ----------
    def attach(self, w, h):
        if self._registered and self._w == w and self._h == h:
            return
        self._w, self._h = w, h          # 先于_ensure_cuda：缓冲尺寸依赖它
        self._ensure_cuda()
        if self._res:
            self._unregister()
        desc = TEXDESC(w, h, 1, 1, DXGI_FORMAT_B8G8R8A8_UNORM, 1, 0,
                       D3D11_USAGE_DEFAULT, 0, 0, 0)
        out = ct.c_void_p()
        rc = _D3DHelper.vfn(ct.c_void_p(self.d3d_device), 5, ct.c_long,
                            ct.POINTER(TEXDESC), ct.c_void_p, ct.POINTER(ct.c_void_p))(
            ct.c_void_p(self.d3d_device), ct.byref(desc), None, ct.byref(out))
        if rc != 0:
            raise GPUPipelineError(f"常驻纹理创建失败: 0x{rc & 0xFFFFFFFF:08X}")
        self._tex = out.value
        self.tex = out.value      # 公共别名（dxgi_capture.readback使用）
        res = ct.c_void_p()
        rc = self._cudart.cudaGraphicsD3D11RegisterResource(
            ct.byref(res), ct.c_void_p(self._tex), 0)
        if rc != 0:
            raise GPUPipelineError(f"CUDA纹理注册失败: cudaError {rc}")
        self._res = res.value
        self._w, self._h = w, h
        self._registered = True

    # ---------- 捕获线程：GDI帧 -> UpdateSubresource写入常驻纹理 ----------
    def update_frame(self, bgra):
        """把BGRA帧写入常驻纹理（UpdateSubresource，实测可用路径；
        本机桌面复制帧资源拷贝损坏，故采用此组合）"""
        if not self._registered:
            raise GPUPipelineError("常驻纹理未注册")
        h, w = self._h, self._w
        if bgra.shape[0] != h or bgra.shape[1] != w:
            raise GPUPipelineError(f"帧尺寸不匹配: {bgra.shape[1]}x{bgra.shape[0]} != {w}x{h}")
        src = np.ascontiguousarray(bgra)
        rc = _D3DHelper.vfn(ct.c_void_p(self.d3d_ctx), 48, None,
                            ct.c_void_p, ct.c_uint, ct.c_void_p,
                            ct.c_void_p, ct.c_uint, ct.c_uint)(
            ct.c_void_p(self.d3d_ctx), ct.c_void_p(self._tex), 0, None,
            ct.c_void_p(src.ctypes.data), w * 4, w * h * 4)
        if rc not in (None, 0):
            raise GPUPipelineError(f"UpdateSubresource失败: 0x{(rc or 0) & 0xFFFFFFFF:08X}")

    # ---------- 检测线程：映射+线性化+预处理，返回blob显存指针 ----------
    def stage(self):
        if not self._registered:
            raise GPUPipelineError("常驻纹理未注册")
        cudart = self._cudart
        rc = cudart.cudaGraphicsMapResources(
            1, ct.byref(ct.c_void_p(self._res)), None)
        if rc != 0:
            raise GPUPipelineError(f"纹理映射失败: cudaError {rc}")
        self._mapped = True
        try:
            arr = ct.c_void_p()
            rc = cudart.cudaGraphicsSubResourceGetMappedArray(
                ct.byref(arr), ct.c_void_p(self._res), 0, 0)
            if rc != 0:
                raise GPUPipelineError(f"获取映射数组失败: cudaError {rc}")
            rc = self._rt.cudaMemcpy2DFromArray(int(self._linear.handle), self._w * 4,
                                                int(arr.value), 0, 0,
                                                self._w * 4, self._h,
                                                self._rt.cudaMemcpyKind.cudaMemcpyDeviceToDevice)[0]
            if rc != 0:
                raise GPUPipelineError(f"GPU线性化拷贝失败: cudaError {rc}")
            self._launch(self._stream, self._lc, self._kernel, self._linear, self._blob)
            self._rt.cudaStreamSynchronize(int(self._stream.handle))
        finally:
            cudart.cudaGraphicsUnmapResources(1, ct.byref(ct.c_void_p(self._res)), None)
            self._mapped = False
        return self.blob_ptr

    def _unregister(self):
        rc = self._cudart.cudaGraphicsUnregisterResource(ct.c_void_p(self._res))
        self._res = None
        self._registered = False
        return rc

    def close(self):
        if self._res:
            try:
                self._unregister()
            except Exception:
                pass
        if self._tex:
            _D3DHelper.rel(ct.c_void_p(self._tex))
            self._tex = None
            self.tex = None

