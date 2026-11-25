# -*- coding: utf-8 -*-
"""主控引擎：GPU捕获-推理线程 + 500Hz控制线程 + win32预览线程 + 状态机(IDLE/TRACKING/LOCKED)"""
import os
import time
import ctypes
import threading
import cv2
import numpy as np
from core.config import MODEL_PATH, LOGITECH_DLL, DEFAULTS, load_settings, save_settings
from core.capture import ScreenCapture, list_monitors
from core import capture_guard
from core.detector import Detector
from core.mouse_backend import (SendInputBackend, LogitechBackend,
                                key_pressed, foreground_title)
def _query_refresh_hz(monitor=None):
    """查询指定显示器刷新率（monitor如'\\\\.\\DISPLAY2'，None=主显示器）。
    旧实现用GetDC(0)只查主屏——游戏跑在副屏时上限显示错误（如副屏120Hz显示成主屏60Hz）"""
    try:
        import ctypes
        from ctypes import wintypes
        user32 = ctypes.windll.user32

        class DEVMODEW(ctypes.Structure):
            # 必须定义完整结构：EnumDisplaySettingsW 会写满整个DEVMODE，
            # 缺字段=越界写栈内存 → 偶发0xC0000005闪退（已实测踩坑）
            _fields_ = [("dmDeviceName", wintypes.WCHAR * 32),
                        ("dmSpecVersion", wintypes.WORD), ("dmDriverVersion", wintypes.WORD),
                        ("dmSize", wintypes.WORD), ("dmDriverExtra", wintypes.WORD),
                        ("dmFields", wintypes.DWORD),
                        ("dmPositionX", wintypes.LONG), ("dmPositionY", wintypes.LONG),
                        ("dmDisplayOrientation", wintypes.DWORD),
                        ("dmDisplayFixedOutput", wintypes.DWORD),
                        ("dmColorShort", wintypes.SHORT), ("dmBitsPerPel", wintypes.SHORT),
                        ("dmPelsWidth", wintypes.DWORD), ("dmPelsHeight", wintypes.DWORD),
                        ("dmDisplayFlags", wintypes.DWORD), ("dmDisplayFrequency", wintypes.DWORD),
                        ("dmICMMethod", wintypes.DWORD), ("dmICMIntent", wintypes.DWORD),
                        ("dmMediaType", wintypes.DWORD), ("dmDitherType", wintypes.DWORD),
                        ("dmReserved1", wintypes.DWORD), ("dmReserved2", wintypes.DWORD),
                        ("dmPanningWidth", wintypes.DWORD), ("dmPanningHeight", wintypes.DWORD)]

        class DISPLAY_DEVICEW(ctypes.Structure):
            _fields_ = [("cb", wintypes.DWORD), ("DeviceName", wintypes.WCHAR * 32),
                        ("DeviceString", wintypes.WCHAR * 128), ("StateFlags", wintypes.DWORD),
                        ("DeviceID", wintypes.WCHAR * 128), ("DeviceKey", wintypes.WCHAR * 128)]

        def _freq(dev_name):
            dm = DEVMODEW()
            dm.dmSize = ctypes.sizeof(DEVMODEW)
            # ENUM_CURRENT_SETTINGS = -1
            if user32.EnumDisplaySettingsW(dev_name, -1, ctypes.byref(dm)):
                hz = int(dm.dmDisplayFrequency)
                return hz if hz > 1 else 0
            return 0

        if monitor:
            for i in range(16):
                dd = DISPLAY_DEVICEW()
                dd.cb = ctypes.sizeof(DISPLAY_DEVICEW)
                if not user32.EnumDisplayDevicesW(None, i, ctypes.byref(dd), 0):
                    break
                if dd.DeviceName == monitor:
                    hz = _freq(dd.DeviceName)
                    if hz:
                        return hz
                    break
        hz = _freq(None)   # 回退：主显示器
        return hz if hz > 1 else 60
    except Exception:
        return 60
class Engine:
    def __init__(self, log_fn=print):
        self.log = log_fn
        self.cfg = dict(DEFAULTS)
        self.cfg["model_path"] = MODEL_PATH
        self.cfg["dll_path"] = LOGITECH_DLL
        saved = load_settings()
        self.cfg.update({k: v for k, v in saved.items() if k in self.cfg})
        if isinstance(self.cfg.get("selected_classes"), list):
            self.cfg["selected_classes"] = set(int(i) for i in self.cfg["selected_classes"])
        self.lock = threading.Lock()
        self.running = False
        self.preview_on = False
        self._pv_hwnd = 0            # 预览窗口句柄（namedWindow后按标题获取）
        self._preview_hidden = False # 切换键控制的窗口显隐状态
        self._hide_prev = False      # 显隐切换键边沿检测
        self.detector = None
        self.detector_ready = False
        self.model_error = None
        self.model_info = {}
        self.backend = None
        self.region = {"left": 0, "top": 0, "width": 0, "height": 0}  # 捕获区域（虚拟屏幕坐标，中心=准星）
        self._move_fail_t = 0.0  # 鼠标发送失败日志节流
        self._diag_t = 0.0       # 诊断日志节流
        self._save_timer = None  # 配置防抖落盘定时器
        self.aim_enabled = True  # 跟随辅助总开关（快捷键toggle切换）
        self._aim_vk_prev = False
        self._cap_tried = set()  # 本轮推理已尝试的捕获方式（异常时自动互切）
        self.threads = []
        self.preview_thread = None
        self._exit = threading.Event()
        self._cap_closed = threading.Event()
        self._cap_closed.set()
        # GPU直通零拷贝管线：必须先于ORT/TRT激活主上下文完成D3D-CUDA关联
        # 失败策略（用户决策）：直接报错拒绝启动，绝不回退CPU
        from core.gpu_pipeline import GPUPipeline
        self.pipeline = None
        self._pipeline_err = None
        try:
            self.pipeline = GPUPipeline()
            self.pipeline.prewarm(self.cfg.get("monitor"))
            self.log(f"[GPU] 直通零拷贝管线就绪（适配器CUDA设备{self.pipeline.cuda_ordinal}）")
        except Exception as e:
            self.pipeline = None
            self._pipeline_err = str(e)
            self.log(f"[GPU] 管线初始化失败：{e}")
        threading.Thread(target=self._capture_worker, daemon=True).start()
        threading.Thread(target=self._control_worker, daemon=True).start()
        self.log("[初始化] 工作线程已启动")
        if self._pipeline_err:
            self.state = "GPU管线初始化失败"
        else:
            self.state = "模型加载中"
        self.fps = 0.0
        self._last_tick = time.perf_counter()
        self.infer_ms = 0.0
        self.num_dets = 0
        self._last_dets = []     # 最近一次新帧的检测结果（重复帧纹理未变=同输出，复用）
        self._last_ms = 0.0
        self.target_err = 0.0
        self.lock_flag = False
        self.boxes = []
        self.frame = None
        self.frame_seq = 0   # 预览帧版本号：预览线程只在真正新帧时重绘（防无效高频重绘闪烁）
        self.provider = "-"
        self.capture_mode = "-"
        self._cap = None
        self.lock_time_total = 0.0
        self.start_ts = 0.0
        # 刷新率只在单线程阶段查询一次并缓存——TRT加载期间并发调用
        # EnumDisplaySettingsW会与preload_cuda_dlls加载的NVIDIA运行库冲突 → 0xC0000005闪退
        self._refresh_hz = _query_refresh_hz(self.cfg.get("monitor"))
        self.log(f"[初始化] 刷新率查询完成：{self._refresh_hz}Hz")
        # 显示器列表同样必须在TRT加载开始前枚举并缓存——GUI晚于模型加载线程
        # 初始化，若GUI在TRT会话创建期间并发调用EnumDisplayMonitors会触发
        # 0xC0000005原生崩溃（提权实例4次崩溃均复现于此竞态）
        self.monitors = list_monitors()
        if self._pipeline_err is None:
            self._load_model()
        self.log("[初始化] 模型加载线程已启动")
    def _load_model(self):
        def work():
            try:
                self.detector = Detector(self.cfg["model_path"], self.cfg["conf"],
                                         self.cfg.get("nms_iou", 0.45),
                                         pipeline=self.pipeline)
                self.provider = self.detector.provider
                self.model_info = {"names": self.detector.names, "family": self.detector.family}
                self._sync_capture_to_model()
                self.state = "空闲"
                self.log(f"[推理] 模型就绪：{self.detector.family}，类别：{self._names_str()}")
            except Exception as e:
                self.model_error = str(e)
                self.state = "模型加载失败"
                self.log(f"[推理] 模型加载失败：{e}")
            finally:
                self.detector_ready = True
                # GC遍历本应用的原生互操作对象（ORT/TRT会话、CUDA管线、
                # WinRT/D3D捕获包装）会原生崩溃。实测证据：faulthandler连续
                # 10+次记录 access violation 发生在GC遍历中
                # （python313.dll+0x15ce2，栈标记"Garbage-collecting"），
                # 另一次崩溃于 nvinfer_10.dll（GC回收垃圾会话→析构与推理
                # 线程竞态）。旧方案"加载完freeze再enable"不够：freeze只保护
                # 冻结时刻已存在的对象，之后新建的对象每次分代回收都会被
                # 遍历——推理中重建捕获（切显示器/捕获自动互切/开停推理）
                # 留下的旧包装即毒对象（17:12崩溃：切显示器2分钟后GC崩溃）。
                # 结论：GC永久禁用，进程生命周期内不再enable。引用计数已
                # 覆盖绝大部分回收；成环垃圾极少且量级可忽略。
                try:
                    import gc
                    gc.freeze()
                except Exception:
                    pass
        threading.Thread(target=work, daemon=True).start()
    def _sync_capture_to_model(self):
        """捕获区域自动匹配模型输入尺寸（模型按该尺度训练，其他尺寸会尺度失配）"""
        size = int(getattr(self.detector, "input_size", 0) or 0)
        if size and int(self.cfg.get("capture_size", 0)) != size:
            self.cfg["capture_size"] = size
            self._save()
            self.log(f"[捕获] 区域大小已自动匹配模型输入：{size}×{size}")
    def _save(self):
        """快照后写盘：遍历在锁内完成，文件IO在锁外——避免与cfg修改竞态"""
        try:
            with self.lock:
                snap = {k: (sorted(v) if isinstance(v, set) else v)
                        for k, v in self.cfg.items()
                        if k in DEFAULTS or k == "model_path"}
            save_settings(snap)
        except Exception:
            pass
    def _schedule_save(self):
        """防抖落盘：连续调参合并为0.5秒后一次写盘"""
        try:
            if self._save_timer is not None:
                self._save_timer.cancel()
        except Exception:
            pass
        self._save_timer = threading.Timer(0.5, self._save)
        self._save_timer.daemon = True
        self._save_timer.start()
    def flush_settings(self):
        """立即落盘待保存配置（退出前调用）"""
        try:
            if self._save_timer is not None:
                self._save_timer.cancel()
        except Exception:
            pass
        self._save_timer = None
        self._save()
    def load_model(self, path):
        """切换加载用户选择的ONNX模型（后台线程），完成后更新类别信息"""
        if self.running:
            self.log("[推理] 请先停止推理再切换模型")
            return
        if not os.path.isfile(path):
            self.log("[推理] 模型文件不存在")
            return
        self.state = "模型加载中"
        self.detector_ready = False
        self.model_error = None
        def work():
            try:
                det = Detector(path, self.cfg["conf"], self.cfg.get("nms_iou", 0.45))
                self.detector = det
                self.provider = det.provider
                self.model_info = {"names": det.names, "family": det.family}
                self.cfg["model_path"] = path
                self._sync_capture_to_model()
                self.state = "空闲"
                self.log(f"[推理] 模型已切换：{det.family}，类别：{self._names_str()}")
            except Exception as e:
                self.model_error = str(e)
                self.state = "模型加载失败"
                self.log(f"[推理] 模型加载失败：{e}")
            finally:
                self.detector_ready = True
                # 同上：GC永久禁用（遍历原生互操作对象=原生崩溃），不再enable
                try:
                    import gc
                    gc.freeze()
                except Exception:
                    pass
        threading.Thread(target=work, daemon=True).start()
    def _names_str(self):
        if not self.model_info.get("names"):
            return "-"
        return ", ".join(f"{k}.{v}" for k, v in sorted(self.model_info["names"].items()))
    def start(self):
        if self.running:
            return
        # 模型未就绪立即返回（不再阻塞GUI线程等待60秒），按钮由GUI置灰
        if self._pipeline_err:
            self.state = "GPU管线初始化失败"
            self.log(f"[引擎] 无法启动：{self._pipeline_err}")
            return
        if self.model_error:
            self.state = "模型加载失败"
            self.log(f"[引擎] 无法启动：{self.model_error}")
            return
        if self.detector is None or not self.detector_ready:
            self.state = "模型加载中"
            self.log("[引擎] 模型仍在加载，就绪后再点「开始推理」")
            return
        try:
            self._init_backend()
        except Exception as e:
            self.state = "启动失败"
            self.log(f"[引擎] 启动失败：{e}")
            return
        self.running = True
        self.state = "运行中"
        self.start_ts = time.perf_counter()
        self.lock_time_total = 0.0
        self.log("[引擎] 推理已启动，长按热键时启用跟随辅助（AimYolo原生算法：每帧直移目标中心）")
    def stop(self):
        if not self.running:
            return
        self.running = False
        self._cap_closed.wait(timeout=5.0)
        self.state = "空闲"
        self.boxes = []
        self.frame = None
        self.lock_flag = False
        # 注：此处禁止gc.collect()——刚关闭的捕获包装（WinRT/D3D互操作）
        # 是GC毒对象，遍历=原生崩溃（GC已永久禁用，见main.py/_load_model）
        self.log("[引擎] 推理已停止")
    def _init_backend(self):
        mode = self.cfg["backend"]
        if self.backend is not None and self.backend.name == ("罗技驱动" if mode == "logitech" else "SendInput"):
            return
        if mode == "logitech":
            dll = self.cfg["dll_path"]
            if not os.path.isfile(dll):
                self.log("[鼠标] 未找到 MouseControl.dll，已回退到 SendInput 后端")
                self.backend = SendInputBackend()
            else:
                self.backend = LogitechBackend(dll)
        else:
            self.backend = SendInputBackend()
    def toggle_preview(self, on):
        if on and not self.preview_on:
            self.preview_on = True
            self._preview_hidden = False   # 窗口重建默认可见
            self.preview_thread = threading.Thread(target=self._preview_loop, daemon=True)
            self.preview_thread.start()
            self.log("[预览] win32预览窗口已开启")
        elif not on and self.preview_on:
            self.preview_on = False
            if self.preview_thread is not None:
                self.preview_thread.join(timeout=2.0)
                self.preview_thread = None
            self._pv_hwnd = 0
            self._preview_hidden = False
            self.log("[预览] 预览窗口已关闭")
    def _toggle_preview_visible(self):
        """切换键（默认Tab）显隐预览窗口：隐藏=SW_HIDE；恢复=SW_SHOWNA
        （不激活窗口，不抢游戏焦点）"""
        hwnd = getattr(self, "_pv_hwnd", 0)
        if not hwnd:
            return
        try:
            self._preview_hidden = not self._preview_hidden
            ctypes.windll.user32.ShowWindow(hwnd, 0 if self._preview_hidden else 8)
            self.log("[预览] 窗口已隐藏（再按切换键恢复显示）" if self._preview_hidden
                     else "[预览] 窗口已恢复显示")
        except Exception:
            pass
    def set_classes(self, ids):
        with self.lock:
            self.cfg["selected_classes"] = set(int(i) for i in ids)
    def get_selected_classes(self):
        with self.lock:
            sel = self.cfg.get("selected_classes")
            return None if sel is None else set(sel)
    @staticmethod
    def _boost_current_thread(pin=0):
        """提升当前线程优先级；多核(≥8逻辑核)时绑定到高位逻辑核——
        游戏通常占用低位核，错开调度竞争降低推理/控制延迟抖动"""
        try:
            k32 = ctypes.windll.kernel32
            k32.SetThreadPriority(k32.GetCurrentThread(), 2)   # THREAD_PRIORITY_HIGHEST
            n = os.cpu_count() or 0
            if pin and n >= 8:
                k32.SetThreadAffinityMask(k32.GetCurrentThread(), 1 << (n - 1 - pin))
        except Exception:
            pass
    def _capture_worker(self):
        """常驻捕获线程：整个进程生命周期固定一个COM apartment，避免跨线程COM崩溃"""
        self._boost_current_thread(pin=0)
        cap = None
        while not self._exit.is_set():
            if not self.running:
                if cap is not None:
                    cap.close()
                    cap = None
                self._cap_closed.set()
                time.sleep(0.05)
                continue
            if cap is not None and (cap.monitor != self.cfg.get("monitor")
                                    or cap.size != int(self.cfg.get("capture_size", 448))
                                    or cap.capture_method != str(self.cfg.get("capture_method", "wgc"))):
                cap.close()
                cap = None
                self._cap = None
                self._cap_tried = set()   # 用户主动切换方式：重置已尝试集合
                time.sleep(0.05)
                continue
            if cap is None:
                try:
                    method = str(self.cfg.get("capture_method", "wgc"))
                    tried = getattr(self, "_cap_tried", None) or set()
                    if method in tried and len(tried) < 2:
                        # 首选方式已在本轮被证实异常，切换另一种（WGC<->DXGI）
                        method = "dxgi" if method == "wgc" else "wgc"
                        self.log(f"[捕获] 自动切换捕获方式: {method}")
                    cap = ScreenCapture(size=self.cfg["capture_size"],
                                        offset=(int(self.cfg.get("crosshair_dx", 0)),
                                                int(self.cfg.get("crosshair_dy", 0))),
                                        monitor=self.cfg.get("monitor"),
                                        pipeline=self.pipeline,
                                        capture_method=method)
                    cap.open()
                    self.capture_mode = cap.mode_desc
                    self._cap = cap
                    self._cap_closed.clear()
                except Exception as e:
                    self.log(f"[捕获] GPU捕获初始化失败：{e}，0.2秒后重试")
                    cap = None
                    time.sleep(0.2)
                    continue
            if cap.broken:
                tried = getattr(self, "_cap_tried", None)
                if tried is None:
                    tried = set()
                tried.add(cap.capture_method)
                self._cap_tried = tried
                self.log(f"[捕获] {cap.broken}（{cap.capture_method}）")
                cap.close()
                cap = None
                self._cap = None
                if len(tried) >= 2:
                    self.log("[捕获] WGC与DXGI均在本机异常——系统捕获层不可用，已停止")
                    self._force_stop()
                    return
                time.sleep(0.3)
                continue
            if getattr(cap, "dxgi", None) is not None and not cap.dxgi.ok:
                cap.close()
                cap = None
                time.sleep(0.2)
                continue
            try:
                self._capture_once(cap)
            except Exception as e:
                self.log(f"[引擎] 捕获线程异常：{e}")
                time.sleep(0.05)
        if cap is not None:
            cap.close()
        self._cap = None
    def _capture_once(self, cap):
        t0 = time.perf_counter()
        new_frame = False
        if cap.gpu_mode:
            # GPU直通零拷贝。有新帧→拷进管线纹理再推理；无新帧→对当前纹理
            # 继续推理（与C++版一致：帧率=推理速率，不被屏幕刷新率锁死）。
            # 重复帧画面未变，不参与跟随测量（见下），只维持检测循环全速运转。
            new_frame = cap.grab_gpu()
            if new_frame:
                self.region = dict(cap.region)
            elif not getattr(self.pipeline, "tex", None):
                time.sleep(0.002)
                return
            try:
                # 重复帧纹理未变，blob内容不变——直接推理（detector每帧自行stage）
                dets, ms = self.detector.infer_gpu(self.pipeline)
            except Exception as e:
                self.log(f"[GPU] 管线故障：{e}——推理已停止")
                self._force_stop()
                return
            # 预览回读：新帧即回读，但限频60Hz——readback的CopyResource+Map会
            # 强制GPU同步，捕获率240Hz时逐帧回读会在热循环里叠加同步开销。
            # 发布率即预览有效帧率（单层节流语义不变），60fps预览足够流畅。
            # 无新帧500ms兜底（抓帧链路坏死时预览不永久黑屏）
            pb_now = time.perf_counter()
            if self.preview_on and ((new_frame and pb_now - getattr(self, "_pb_t", 0.0) >= 0.0155)
                                    or pb_now - getattr(self, "_pb_t", 0.0) > 0.5):
                self._pb_t = pb_now
                frame = cap.readback()
            else:
                frame = None
        else:
            # 按用户决策：GPU直通是唯一路径，不处于零拷贝模式即拒绝（不回退CPU）
            self.log("[GPU] 捕获未处于零拷贝模式——拒绝推理（不回退CPU）")
            self._force_stop()
            return
        sel = self.cfg.get("selected_classes")
        if sel is not None:
            dets = [d for d in dets if d[5] in sel]
        # —— AimYolo原生跟随算法（github.com/Aa-bN/AimYolo）+ 可调参数 ——
        #    每个新检测帧选目标（最近中心/最高置信度），跟随点=框内Y偏移位置。
        #    鼠标速度1.0=SendInput绝对坐标直达（AimYolo原生，无需光标位置）；
        #    <1.0或罗技后端=按光标当前位置每帧走差值百分比（连续帧几何收敛，
        #    等效平滑）。FOV限制选目标范围；死区防框抖动微颤。
        now = time.perf_counter()
        if new_frame:
            rx, ry = self.region["left"], self.region["top"]
            w_, h_ = self.region["width"], self.region["height"]
            avk = int(self.cfg.get("vk", 0x06) or 0)
            aiming = (bool(avk) and self.aim_enabled
                      and key_pressed(avk)
                      and self._window_ok())
            if dets:
                fov = float(self.cfg.get("fov_radius", 0.0) or 0.0)
                if fov > 0:
                    dets = [d for d in dets
                            if ((d[0] + d[2]) / 2.0 - w_ / 2.0) ** 2
                            + ((d[1] + d[3]) / 2.0 - h_ / 2.0) ** 2 <= fov * fov]
            if dets:
                if self.cfg.get("strategy", "最近中心") == "最高置信度":
                    best = max(dets, key=lambda d: d[4])
                else:
                    # 目标锁定迟滞：上一帧跟随的目标仍在（有框几何中心距其<80px）
                    # 则继续跟它，不再每帧按"离准星最近"重选——敌人+敌人头双框
                    # 同屏时两者中心差几十px，按最近重选=每帧在头/身框间跳变=Y轴狂震。
                    prev = getattr(self, "_lock_tgt", None)
                    best = None
                    if prev is not None:
                        cand = min(dets, key=lambda d:
                                   ((d[0] + d[2]) / 2.0 - prev[0]) ** 2
                                   + ((d[1] + d[3]) / 2.0 - prev[1]) ** 2)
                        if ((cand[0] + cand[2]) / 2.0 - prev[0]) ** 2 \
                                + ((cand[1] + cand[3]) / 2.0 - prev[1]) ** 2 <= 80 ** 2:
                            best = cand
                    if best is None:
                        best = min(dets, key=lambda d:
                                   ((d[0] + d[2]) / 2.0 - w_ / 2.0) ** 2
                                   + ((d[1] + d[3]) / 2.0 - h_ / 2.0) ** 2)
                # 跟随点Y偏移：-0.5=框顶 0=中心 +0.5=框底
                off = max(-0.5, min(0.5, float(self.cfg.get("aim_offset_y", 0.0) or 0.0)))
                cx0 = (best[0] + best[2]) / 2.0
                cy0 = (best[1] + best[3]) / 2.0
                # —— 移动目标预测：锁定目标的帧间位移→速度EMA→瞄准点前移 ——
                # 目的：消检测+画面显示固有延迟（目标横移时瞄点落后真实位置）
                prev = getattr(self, "_lock_tgt", None)
                lead = max(0.0, min(120.0, float(self.cfg.get("aim_lead", 0.0) or 0.0)))
                dtf = now - getattr(self, "_lock_t", 0.0)
                if lead > 0 and prev is not None and 0.0 < dtf < 0.5:
                    ivx, ivy = (cx0 - prev[0]) / dtf, (cy0 - prev[1]) / dtf
                    # 帧间速度>3000px/s=框切换/检测跳变噪声，不采纳
                    if ivx * ivx + ivy * ivy < 3000.0 ** 2:
                        v = getattr(self, "_tgt_v", None)
                        a = 0.35 if v is None else 0.3    # 首帧直接采纳，其后EMA平滑
                        self._tgt_v = (ivx, ivy) if v is None else (
                            v[0] * (1 - a) + ivx * a, v[1] * (1 - a) + ivy * a)
                else:
                    self._tgt_v = None   # 目标刚切换/间隔超时：速度清零重建
                self._lock_t = now
                vx, vy = getattr(self, "_tgt_v", None) or (0.0, 0.0)
                bx = cx0 + vx * lead / 1000.0
                by = cy0 + vy * lead / 1000.0 + (best[3] - best[1]) * off
                self._lock_tgt = (cx0, cy0)  # 迟滞用：几何中心
                ex, ey = bx - w_ / 2.0, by - h_ / 2.0
                dist = (ex * ex + ey * ey) ** 0.5
                self.target_err = dist
                self.lock_flag = aiming and dist < float(self.cfg.get("lock_radius", 15.0))
                self.state = ("跟随已关闭" if not self.aim_enabled
                              else "已跟随" if self.lock_flag
                              else "跟随中" if aiming else "搜索目标")
                if aiming and dist >= float(self.cfg.get("deadzone", 2.0)):
                    # 鼠标速度：1.0=每帧直满；<1.0=每帧朝目标走剩余误差的百分比
                    spd = min(1.0, max(0.05, float(self.cfg.get("aim_speed", 1.0))))
                    # 视角增益K：鼠标移动1px时目标在屏幕上反向移动的像素数
                    # （由游戏灵敏度/FOV/分辨率决定）。步长=误差/K：
                    # K=游戏真实增益时每帧一步到位（spd=1.0零过冲完美吸附）
                    k_gain = max(0.1, float(self.cfg.get("sens_k", 1.0) or 1.0))
                    try:
                        if spd >= 0.999 and k_gain <= 1.001 \
                                and isinstance(self.backend, SendInputBackend):
                            self.backend.move_abs(rx + bx, ry + by)  # AimYolo原生绝对直达（仅K≈1低灵敏度稳）
                        else:
                            # 步长基准=目标相对准星(区域中心)的屏幕误差——
                            # 绝不读光标位置：raw input游戏会把OS光标推到屏幕
                            # 边缘，按它算差值=步长巨大方向错乱=吸附甩飞
                            dxm, dym = ex * spd / k_gain, ey * spd / k_gain
                            if self.cfg.get("invert_y"):
                                dym = -dym
                            # 亚像素残差累积：每帧<1px时直接int会截成0发不出
                            self._res_x = getattr(self, "_res_x", 0.0) + dxm
                            self._res_y = getattr(self, "_res_y", 0.0) + dym
                            sx, sy = int(self._res_x), int(self._res_y)
                            self._res_x -= sx
                            self._res_y -= sy
                            if sx or sy:
                                self.backend.move(sx, sy)
                    except Exception as e:
                        if now - self._move_fail_t > 2.0:
                            self._move_fail_t = now
                            self.log(f"[鼠标] 移动失败：{e}")
                if now - self._diag_t > 0.25:
                    self._diag_t = now
                    self.log(f"[跟随] 目标=({rx + bx:.0f},{ry + by:.0f}) "
                             f"距离={dist:.0f}px {'发射' if aiming else '待机'}")
            else:
                self.lock_flag = False
                self._lock_tgt = None  # 目标消失，下帧重新按最近中心选
                self.state = ("跟随已关闭" if not self.aim_enabled
                              else "搜索目标" if aiming else "运行中")
            dt = now - self._last_tick
            # 帧率显示=真实画面更新率(新帧率)：重复帧tick不计入
            self.fps = 0.9 * self.fps + 0.1 * (1.0 / max(min(dt, 0.5), 0.001))
            self._last_tick = now
        self.infer_ms = ms
        with self.lock:
            if new_frame:
                self.num_dets = len(dets)
                self.boxes = dets
            # 回读失败（Map偶发错）时保留上一帧：预览不出现黑屏跳变
            if self.preview_on and frame is not None:
                self.frame = frame
                self.frame_seq += 1
        # 检测帧率上限（默认240Hz）：仅作为新帧轮询节流——重复帧不推理（复用
        # 上次结果），实际推理速率=游戏画面更新率（≤刷新率）
        wait = 1.0 / max(30.0, float(self.cfg.get("detect_fps", 240))) - (time.perf_counter() - t0)
        if wait > 0:
            time.sleep(wait)

    def _control_worker(self):
        """辅助热键线程：紧急停止/跟随开关/预览显隐（跟随发射已并入检测帧循环）"""
        self._boost_current_thread(pin=1)
        while not self._exit.is_set():
            # 紧急停止热键：推理中按下立即终止（未设置/VK0不启用——
            # GetAsyncKeyState(0)恒返回按下态，不过滤会在启动瞬间误触发）
            pvk = int(self.cfg.get("panic_vk", 0x23) or 0)
            if self.running and pvk and key_pressed(pvk):
                self._force_stop()
                continue
            if not self.running:
                time.sleep(0.05)
                self._aim_vk_prev = False
                continue
            # —— 跟随辅助开关快捷键：按下切换（边沿触发，仅推理运行中生效）——
            with self.lock:
                avk = int(self.cfg.get("aim_enable_vk", 0) or 0)
            if avk:
                pa = key_pressed(avk)
                if pa and not self._aim_vk_prev:
                    self.aim_enabled = not self.aim_enabled
                    self.log("[开关] 跟随辅助已{}（{}）".format(
                        "开启" if self.aim_enabled else "关闭",
                        self.cfg.get("aim_enable_name", "")))
                self._aim_vk_prev = pa
            else:
                self._aim_vk_prev = False
            # —— 预览窗口显隐切换键（边沿触发，固定Tab；仅预览开启时轮询）——
            if self.preview_on:
                hp = key_pressed(0x09)   # VK_TAB
                if hp and not self._hide_prev:
                    self._toggle_preview_visible()
                self._hide_prev = hp
            else:
                self._hide_prev = False
            time.sleep(0.004)
    def _window_ok(self):
        """生效窗口过滤：前台窗口标题需包含关键字（留空=全局生效）"""
        filt = (self.cfg.get("win_filter") or "").strip()
        if not filt:
            return True
        return filt.lower() in foreground_title().lower()
    def _force_stop(self):
        self.running = False
        self._cap_closed.wait(timeout=3.0)
        self.state = "已停止"
        self.lock_flag = False
        self.log("[热键] 紧急停止已触发")
    def _preview_loop(self):
        # 纯cv2绘制（OpenCV C调用释放GIL，240fps重绘也不挤占检测线程；
        # PIL路径持GIL 3-5ms/帧，高速推理下会吃满GIL把检测线程饿死）
        win = "实时检测预览"
        colors = [(0, 170, 255), (0, 230, 120), (255, 170, 0), (230, 0, 255), (0, 255, 255),
                  (255, 80, 80), (160, 100, 255), (255, 255, 0)]
        state_en = {"运行中": "RUN", "搜索目标": "SEARCH", "已跟随": "LOCKED",
                    "跟随中": "TRACK", "跟随已关闭": "AUX OFF", "已停止": "STOP"}
        try:
            cv2.namedWindow(win, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(win, 720, 720)
            # 任务栏/Alt-Tab隐藏：窗口此时已创建但未显示（首次imshow才显示），
            # 先找到句柄改好样式再进入绘制循环——任务栏按钮/Alt-Tab项从头不出现
            hwnd = 0
            for _ in range(30):          # cv2窗口创建异步可见，重试至多300ms
                hwnd = capture_guard.find_window(win)
                if hwnd:
                    break
                time.sleep(0.01)
            if hwnd:
                self._pv_hwnd = hwnd
                capture_guard.hide_from_taskbar(hwnd, True)   # 默认行为：不进任务栏/Alt-Tab
            if self.cfg.get("anti_capture"):
                for _ in range(20):
                    hwnd = capture_guard.find_window(win)
                    if hwnd and capture_guard.apply(hwnd, True):
                        break
                    time.sleep(0.05)
            view = self.cfg["capture_size"]
            drawn_seq = -1
            last_draw = 0.0
            while self.preview_on and self.running:
                # 隐藏状态（Tab切换）下跳过imshow——HighGUI的imshow内部会
                # ShowWindow(SW_SHOW)把隐藏的窗口立刻顶回显示
                if getattr(self, "_preview_hidden", False):
                    if cv2.waitKey(15) == 27:
                        break
                    continue
                # 锁内先查帧版本号+绘制节流(120fps)：只有"真正新帧且距上次绘制
                # ≥8ms"才拷贝绘制。无新帧零拷贝不重绘（防WDA_EXCLUDE的DWM排除
                # 路径闪烁）。纯cv2路径下120fps重绘GIL占用约0，检测线程不受影响。
                now_t = time.perf_counter()
                with self.lock:
                    if (self.frame is None or self.frame_seq == drawn_seq
                            or now_t - last_draw < 0.008):
                        frame = None
                    else:
                        # 引用即可：发布数组每帧全新分配，绘制改动不会被复用
                        frame = self.frame
                        boxes = list(self.boxes)
                        fps, ms, n = self.fps, self.infer_ms, self.num_dets
                        state, mode = self.state, self.capture_mode
                        seq = self.frame_seq
                if frame is None:
                    if cv2.waitKey(5) == 27:
                        break
                    continue
                drawn_seq = seq
                last_draw = now_t
                h, w = frame.shape[:2]
                scale = max(1, round(view / 160))
                thin = max(1, scale)
                for b in boxes:
                    x1, y1, x2, y2 = int(b[0]), int(b[1]), int(b[2]), int(b[3])
                    color = colors[int(b[5]) % len(colors)]
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thin)
                    cv2.putText(frame, f"cls{int(b[5])} {b[4]:.2f}",
                                (x1, max(14, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX,
                                0.45, color, 1, cv2.LINE_AA)
                cx, cy = w // 2, h // 2
                # FOV跟随范围圈：圆心=区域中心(=准星)，半径=跟随范围，0=不限不画
                fov_v = float(self.cfg.get("fov_radius", 0.0) or 0.0)
                if fov_v > 0:
                    cv2.circle(frame, (cx, cy), int(fov_v), (60, 255, 60), max(1, thin))
                cv2.line(frame, (cx - 7 * scale, cy), (cx + 7 * scale, cy), (60, 255, 60), thin)
                cv2.line(frame, (cx, cy - 7 * scale), (cx, cy + 7 * scale), (60, 255, 60), thin)
                cv2.putText(frame, f"FPS {fps:.0f} | {ms:.1f}ms | tgt {n} | {mode} | "
                            f"{state_en.get(state, state)}",
                            (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                            (240, 240, 240), 1, cv2.LINE_AA)
                seg = getattr(self.detector, "seg", None)
                if seg:
                    cv2.putText(frame, f"stage {seg['stage']:.2f}  run {seg['run']:.2f}  "
                                f"copy {seg['copy']:.3f}  dec {seg['dec']:.2f} ms",
                                (8, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                                (150, 220, 150), 1, cv2.LINE_AA)
                shown = cv2.resize(frame, (720, 720), interpolation=cv2.INTER_LINEAR)
                cv2.imshow(win, shown)
                if cv2.waitKey(1) == 27:
                    break
        except Exception as e:
            self.log(f"[预览] 预览线程异常：{e}")
        finally:
            try:
                cv2.destroyWindow(win)
            except cv2.error:
                pass
            self._pv_hwnd = 0
            self.preview_on = False
            self.log("[预览] 预览窗口已关闭")
    def get_status(self):
        with self.lock:
            return {
                "state": self.state, "fps": self.fps, "infer_ms": self.infer_ms,
                "num_dets": self.num_dets, "err": self.target_err,
                "locked": self.lock_flag, "provider": self.provider,
                "backend": self.backend.name if self.backend else "-",
                "capture_mode": self.capture_mode,
                "refresh_hz": self._refresh_hz,
                "lock_time": self.lock_time_total,
                "aim_on": self.aim_enabled,
                "uptime": (time.perf_counter() - self.start_ts)
                          if self.running and self.start_ts else 0.0,
            }
    def update_cfg(self, key, value):
        if key == "capture_size" and self.detector is not None:
            # 强制锁定模型输入尺寸（模型按该尺度训练，其他尺寸会检测错位）
            size = int(getattr(self.detector, "input_size", 0) or 0)
            if size:
                value = size
        with self.lock:
            self.cfg[key] = value
            det = self.detector
        if key == "conf" and det is not None:
            det.conf = float(value)
        elif key == "nms_iou" and det is not None:
            det.nms_iou = float(value)
        elif key == "monitor":
            self._refresh_hz = _query_refresh_hz(value)
            if self.pipeline is not None and value and \
                    value not in self.pipeline.adapter_outputs:
                self.log("[GPU] 警告：新显示器由其他适配器输出——GPU直通绑定当前适配器，"
                         "需重启程序才能在该显示器上生效")
        elif key in ("crosshair_dx", "crosshair_dy") and self._cap is not None:
            try:
                self._cap.set_offset(int(self.cfg.get("crosshair_dx", 0)),
                                     int(self.cfg.get("crosshair_dy", 0)))
            except Exception:
                pass
        self._schedule_save()
