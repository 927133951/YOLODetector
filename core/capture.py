# -*- coding: utf-8 -*-
"""屏幕捕获模块：GPU直通管线（capture_method: wgc / dxgi）
帧纹理直接进入GPUPipeline常驻纹理 -> CUDA映射预处理 -> TRT推理，全程无CPU图像拷贝"""
import ctypes
from ctypes import wintypes, POINTER, WINFUNCTYPE, byref
import numpy as np
import mss
from core.dxgi_capture import DXGICapture


def list_monitors():
    """枚举显示器，返回 [(设备名, 是否主屏)]，用于GUI选择捕获显示器"""
    user32 = ctypes.windll.user32

    class MONITORINFOEXW(ctypes.Structure):
        _fields_ = [("cbSize", wintypes.DWORD), ("rcMonitor", wintypes.RECT),
                    ("rcWork", wintypes.RECT), ("dwFlags", wintypes.DWORD),
                    ("szDevice", wintypes.WCHAR * 32)]
    try:
        primary = user32.MonitorFromPoint(wintypes.POINT(0, 0), 1)
    except Exception:
        primary = None
    res = []

    def _cb(hmon, hdc, lprect, lparam):
        mi = MONITORINFOEXW()
        mi.cbSize = ctypes.sizeof(mi)
        if user32.GetMonitorInfoW(hmon, byref(mi)):
            res.append((mi.szDevice, hmon == primary))
        return True
    CB = WINFUNCTYPE(wintypes.BOOL, wintypes.HMONITOR, wintypes.HDC,
                     POINTER(wintypes.RECT), wintypes.LPARAM)
    try:
        user32.EnumDisplayMonitors(None, None, CB(_cb), 0)
    except Exception:
        pass
    return res or [(r"\\.\DISPLAY1", True)]


class ScreenCapture:
    """指定显示器(默认主屏)的偏移中心区域捕获
    capture_method（pipeline模式下）：
      wgc  = Windows Graphics Capture（推荐）
      dxgi = DXGI Desktop Duplication
    帧纹理均GPU直通进入管线常驻纹理，无CPU图像拷贝。"""
    def __init__(self, size=320, offset=(0, 0), monitor=None, pipeline=None,
                 capture_method="wgc"):
        self.size = size
        self.offset = [int(offset[0]), int(offset[1])]
        self.monitor = monitor
        self.pipeline = pipeline
        self.capture_method = capture_method
        self.gpu_mode = False
        self.mode_desc = "-"
        self.broken = None     # 捕获异常原因（None=正常）
        self.region = {"left": 0, "top": 0, "width": size, "height": size}
        self._base = {"left": 0, "top": 0, "width": 0, "height": 0}
        self.dxgi = None
        self.sct = None
        self.wgc = None

    def _apply_region(self):
        cx = self._base["left"] + self._base["width"] // 2 + self.offset[0]
        cy = self._base["top"] + self._base["height"] // 2 + self.offset[1]
        self.region = {"left": cx - self.size // 2, "top": cy - self.size // 2,
                       "width": self.size, "height": self.size}

    def _monitor_rect(self, device_name):
        """按设备名取显示器RECT与HMONITOR，None则取主屏"""
        user32 = ctypes.windll.user32

        class MONITORINFOEXW(ctypes.Structure):
            _fields_ = [("cbSize", wintypes.DWORD), ("rcMonitor", wintypes.RECT),
                        ("rcWork", wintypes.RECT), ("dwFlags", wintypes.DWORD),
                        ("szDevice", wintypes.WCHAR * 32)]
        want = device_name
        result = []

        def _cb(hmon, hdc, lprect, lparam):
            mi = MONITORINFOEXW()
            mi.cbSize = ctypes.sizeof(mi)
            if user32.GetMonitorInfoW(hmon, byref(mi)):
                result.append((mi.szDevice, mi.rcMonitor, hmon))
            return True
        CB = WINFUNCTYPE(wintypes.BOOL, wintypes.HMONITOR, wintypes.HDC,
                         POINTER(wintypes.RECT), wintypes.LPARAM)
        try:
            user32.EnumDisplayMonitors(None, None, CB(_cb), 0)
        except Exception:
            pass
        try:
            primary = user32.MonitorFromPoint(wintypes.POINT(0, 0), 1)
        except Exception:
            primary = None
        for name, rect, hmon in result:
            if want and name == want:
                return rect, hmon
        for name, rect, hmon in result:
            if hmon == primary:
                return rect, hmon
        if result:
            return result[0][1], result[0][2]
        return None, None

    def open(self):
        """在捕获线程中调用。pipeline模式失败直接抛错（不回退CPU）"""
        if self.pipeline is not None:
            rect, hmon = self._monitor_rect(self.monitor)
            if rect is None:
                raise RuntimeError(f"找不到显示器 {self.monitor or '主屏'}")
            self._base = {"left": rect.left, "top": rect.top,
                          "width": rect.right - rect.left,
                          "height": rect.bottom - rect.top}
            self._apply_region()
            self.pipeline.attach(self.size, self.size)
            self.gpu_mode = True
            method = self.capture_method
            if method == "wgc":
                from core.wgc_capture import WGCCapture
                self.wgc = WGCCapture(self.pipeline, self.monitor)
                self.wgc.open(rect)
                self.mode_desc = "GPU直通管线 (WGC+CUDA)"
            elif method == "dxgi":
                self.dxgi = DXGICapture(self.size, self.size, device_name=self.monitor,
                                        ext_device=self.pipeline.d3d_device,
                                        ext_ctx=self.pipeline.d3d_ctx,
                                        pipeline=self.pipeline)
                self.mode_desc = "GPU直通管线 (DXGI+CUDA)"
            else:
                raise RuntimeError(f"未知捕获方式: {method}")
            print(f"[捕获] {self.mode_desc} 就绪，区域：中心{self.size}x{self.size} "
                  f"偏移({self.offset[0]},{self.offset[1]})")
            return True
        try:
            self.dxgi = DXGICapture(self.size, self.size, device_name=self.monitor)
            self.dxgi.ox, self.dxgi.oy = self.offset
            self._base = dict(self.dxgi.output_desc)
            self.mode_desc = "GPU (DXGI桌面复制)"
            print(f"[捕获] GPU桌面复制引擎就绪，区域：中心{self.size}x{self.size} "
                  f"偏移({self.offset[0]},{self.offset[1]})")
            self._apply_region()
            return True
        except Exception as e:
            print(f"[捕获] GPU捕获不可用({e})，回退mss软件捕获")
            self.dxgi = None
            with mss.mss() as s:
                mon = s.monitors[1] if len(s.monitors) > 1 else s.monitors[0]
            self._base = {"left": mon["left"], "top": mon["top"],
                          "width": mon["width"], "height": mon["height"]}
            self.mode_desc = "CPU回退 (mss)"
            self._apply_region()
            return False

    def set_offset(self, dx, dy):
        """实时调整准星偏移（不重建捕获器）"""
        self.offset = [int(dx), int(dy)]
        if self.dxgi is not None:
            self.dxgi.ox, self.dxgi.oy = self.offset
        self._apply_region()

    def grab_gpu(self):
        """GPU管线路径：按捕获方式取帧并送入管线常驻纹理。返回True=有新帧"""
        m = self.capture_method
        if m == "wgc" and self.wgc is not None:
            ok = self.wgc.grab_gpu()
            if getattr(self.wgc, "neutered", False) and self.broken is None:
                self.broken = "WGC帧纹理访问异常"
            self._check_silent_fail(ok, "wgc")
            return ok
        if m == "dxgi" and self.dxgi is not None:
            ok = self.dxgi.grab_gpu()
            if getattr(self.dxgi, "neutered", False) and self.broken is None:
                self.broken = "DXGI帧资源访问异常"
            self._check_silent_fail(ok, "dxgi")
            return ok
        return False

    def _check_silent_fail(self, ok, kind):
        """静默失败检测：无新帧(超时)是正常现象不算失败；但"帧纹理定位/QI持续
        失败"意味着抓帧链路坏死——不置broken会变成推理黑画面空转、预览全黑。
        连续200次(约1秒)定位失败→broken→引擎自动互切另一捕获方式"""
        cap = self.wgc if kind == "wgc" else self.dxgi
        if ok:
            self._loc_fail = 0
            return
        stage = str(getattr(cap, "last_stage", ""))
        if stage == "locate_tex" or stage.startswith("error") or stage == "trynext_fail":
            self._loc_fail = getattr(self, "_loc_fail", 0) + 1
            if self._loc_fail >= 200 and self.broken is None:
                pool_n = len(getattr(self.wgc, "_pool_texs", []) or [])
                self.broken = (f"{kind.upper()}帧纹理持续定位失败"
                               f"(last={stage}, 次数={self._loc_fail}, 池纹理={pool_n})")
        elif kind == "dxgi" and stage == "qi_texture":
            self._loc_fail = getattr(self, "_loc_fail", 0) + 1
            if self._loc_fail >= 200 and self.broken is None:
                self.broken = f"DXGI帧资源QI持续失败(次数={self._loc_fail})"
        else:
            self._loc_fail = 0

    def readback(self):
        """预览用：返回最近一帧BGR"""
        m = self.capture_method
        if m == "wgc" and self.wgc is not None:
            return self.wgc.readback()
        if m == "dxgi" and self.dxgi is not None:
            return self.dxgi.readback()
        return None

    def grab(self):
        if self.dxgi is not None:
            return self.dxgi.grab()
        if self.sct is None:
            self.sct = mss.mss()
        raw = self.sct.grab(self.region)
        return np.ascontiguousarray(np.array(raw)[:, :, :3])

    def close(self):
        if self.wgc is not None:
            try:
                self.wgc.close()
            except Exception:
                pass
            self.wgc = None
        if self.dxgi is not None:
            self.dxgi.close()
            self.dxgi = None
        if self.sct is not None:
            self.sct.close()
            self.sct = None
