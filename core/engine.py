# -*- coding: utf-8 -*-
"""主控引擎：GPU捕获-推理线程 + 500Hz控制线程 + win32预览线程 + 状态机(IDLE/TRACKING/LOCKED)"""
import os
import math
import time
import ctypes
import threading
import cv2
import numpy as np
from core.config import (MODEL_PATH, LOGITECH_DLL, CONTROL_HZ, DEFAULTS,
                         load_settings, save_settings)
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
        self._gain_est = 2.5     # 单帧增益：每count鼠标位移→屏幕px（在线自适应标定）
        self._g_err = None       # 上一检测帧误差向量（增益采样配对）
        self._g_t = 0.0          # 上一检测帧时刻
        self._aim = None         # 跟随状态快照：检测线程发布，500Hz控制线程消费
        self._acc = [0.0, 0.0]   # 亚像素累加器（避免末段被整数取整卡死）
        self._acc_ff = [0.0, 0.0]  # 前馈通道亚像素累加器（速度匹配，不记在途账）
        self._flying = [0.0, 0.0]  # 在途位移(px)：已发送但画面尚未反映的移动量
        self._tgt = None         # 上一帧选中目标 {box,c,t}（速度估计+选目标持久化锚点）
        self._v = [0.0, 0.0]     # 目标屏幕速度估计(px/s，已扣除相机位移)
        self._v_t = 0.0          # 速度估计最后更新时刻（前馈新鲜度门控）
        self._dp = [0.0, 0.0]    # 两次目标测量间累计发送的鼠标位移(count)
        self._dt_avg = 0.011     # 推理tick周期平滑值（仅显示/限流参考，含重复帧tick，不能当测量周期用）
        self._meas_dt = 0.012    # 新帧(测量)间隔平滑值：控制律折算基准——比例步进/在途衰减必须按
                                 # "画面真正变化的间隔"折算，按推理tick(2~4ms)折算会把每帧
                                 # 实际消除的误差比例放大2~4倍 → 等效死拍控制 → 结构性震荡
        self._meas_t = 0.0       # 上一新帧时刻
        self._g_hist = []        # 最近有效增益样本窗（中位数滤波：单帧毒样本无法鞭甩增益）
        self._g_valid_t = 0.0    # 最近一次有效增益样本时刻（过冲死锁逃逸判据）
        self._gain_dirty = False  # 增益标定有新收敛值待落盘（检测线程置位，控制线程落盘）
        self._gain_saved_t = 0.0  # 上次增益落盘时刻（2s节流）
        self._ff_on = [False, False]  # 前馈通道迟滞开关：防检测框噪声让速度在阈值附近反复开关
        self._save_timer = None  # 配置防抖落盘定时器
        self._move_fail_t = 0.0  # 鼠标发送失败日志节流
        self._diag_t = 0.0       # 诊断日志节流
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
    def _persist_gain(self, force=False):
        """在线标定的收敛增益节流落盘(2s)：下次启动热启动继承，跳过开场收敛期。
        force=True跳过节流（停止推理/改联动系数时立即写入）。
        _gain_dirty由检测线程置位、本方法清位——竞态最多多落盘一次，无害"""
        if not self._gain_dirty:
            return
        now = time.perf_counter()
        if not force and now - self._gain_saved_t < 2.0:
            return
        self._gain_saved_t = now
        self._gain_dirty = False
        with self.lock:
            self.cfg["gain_est_saved"] = round(float(self._gain_est), 4)
        self._save()
    def _reseed_gain(self):
        """增益种子选择：优先继承上次会话在线标定的收敛值（热启动跳过开场
        收敛期，开跑即贴合）；无有效历史（首次运行/值越界）时退回"联动系数"
        (cam_g)作为起点。样本窗清空，在线自适应从种子继续微调"""
        saved = 0.0
        try:
            saved = float(self.cfg.get("gain_est_saved", 0) or 0)
        except Exception:
            saved = 0.0
        if 0.5 <= saved <= 8.0:
            self._gain_est = saved
            self.log(f"[增益] 热启动：继承上次标定值 {saved:.2f}（跳过收敛期）")
        else:
            self._gain_est = min(8.0, max(0.5, float(self.cfg.get("cam_g", 2.5) or 2.5)))
        self._g_hist = []
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
        # 增益热启动：继承上次会话在线标定的收敛值，跳过开场收敛期
        self._reseed_gain()
        self._g_err = None
        self._g_t = 0.0
        self._aim = None
        self._acc = [0.0, 0.0]
        self._acc_ff = [0.0, 0.0]
        self._flying = [0.0, 0.0]
        self._tgt = None
        self._v = [0.0, 0.0]
        self._v_t = 0.0
        self._dp = [0.0, 0.0]
        self._dt_avg = 0.011
        self._meas_dt = 0.012
        self._meas_t = 0.0
        self._g_valid_t = 0.0
        self._ff_on = [False, False]
        self.start_ts = time.perf_counter()
        self.lock_time_total = 0.0
        self.log("[引擎] 推理已启动，长按热键时启用跟随辅助")
    def stop(self):
        if not self.running:
            return
        self.running = False
        self._persist_gain(force=True)   # 会话内收敛的增益立即落盘，下次热启动
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
    def _update_target_velocity(self, sx, sy, box, now, err):
        """帧间匹配同一目标，估计其真实屏幕速度(px/s)。
        相机转动会把画面里所有物体整体平移：帧间原始位移 = 目标真实运动 - g·已发送位移。
        仅在低位移帧(|dp|<=4count，准星已基本贴合目标)更新——大位移收敛帧的补偿偏差
        (g_est-g_true)·dp/dt 会经前馈形成正反馈（过冲→弹回→循环摇晃的根源）；
        大位移帧保持旧估计并轻微衰减。"""
        t_prev = self._tgt
        self._tgt = {"box": box, "c": (sx, sy), "t": now}
        dp = self._dp
        self._dp = [0.0, 0.0]
        if t_prev is None:
            self._v = [0.0, 0.0]
            self._v_t = now
            return
        dt = now - t_prev["t"]
        if dt < 0.002 or dt > 0.12:
            self._v = [0.0, 0.0]      # 首帧/闪断超宽限：速度不可信，清零
            self._v_t = now
            return
        dxr = sx - t_prev["c"][0]
        dyr = sy - t_prev["c"][1]
        exp_dx = self._v[0] * dt - self._gain_est * dp[0]
        exp_dy = self._v[1] * dt - self._gain_est * dp[1]
        if (dxr - exp_dx) ** 2 + (dyr - exp_dy) ** 2 > 60.0 ** 2:
            self._v = [0.0, 0.0]      # 与预测差>60px：目标已切换，速度清零
            self._v_t = now
            return
        # —— 过冲泄惯性：误差真的穿零（准星越过目标，两侧幅度>8px才判真）
        #    时泄一半速度估计。旧条件 dxr·prev_err<0 用的是目标位移噪声
        #    (±3~4px天然跨零)——横移目标稳态跟踪时每窗误触发，速度估计被
        #    反复腰斩→前馈失效→纯比例滞后追踪 ——
        prev = self._g_err
        if prev is not None and err is not None:
            if prev[0] * err[0] < 0 and abs(prev[0]) > 8.0 and abs(err[0]) > 8.0:
                self._v[0] *= 0.5
            if prev[1] * err[1] < 0 and abs(prev[1]) > 8.0 and abs(err[1]) > 8.0:
                self._v[1] *= 0.5
        if max(abs(dp[0]), abs(dp[1])) <= 6.0:
            raw_vx = (dxr + self._gain_est * dp[0]) / dt
            raw_vy = (dyr + self._gain_est * dp[1]) / dt
            # 离群拒绝：与当前估计差>300px/s的样本是检测框跳变/框尺寸抖动的伪影
            # （日志实测：目标近静止时速度估计仍反复跳±170~440px/s——这些噪声
            #   经前馈直接注入鼠标位移=上下摇晃的根源），直接拒绝不混合。
            # 真实横向拉枪的速度变化是连续的（每窗Δv≈几十px/s），不会被误拒。
            if (raw_vx - self._v[0]) ** 2 + (raw_vy - self._v[1]) ** 2 > 300.0 ** 2:
                self._v[0] *= 0.9
                self._v[1] *= 0.9
            else:
                self._v[0] = 0.7 * self._v[0] + 0.3 * raw_vx
                self._v[1] = 0.7 * self._v[1] + 0.3 * raw_vy
                self._v_t = now
        else:
            self._v[0] *= 0.99        # 大位移收敛帧：不更新，旧估计缓衰减
            self._v[1] *= 0.99        # （0.99：快速追踪期前馈不失效，掉速减小）
        lim = 1600.0
        self._v[0] = max(-lim, min(lim, self._v[0]))
        self._v[1] = max(-lim, min(lim, self._v[1]))
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
                # 重复帧纹理未变，blob内容不变——跳过映射/预处理直接推理
                dets, ms = self.detector.infer_gpu(self.pipeline, do_stage=new_frame)
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
        # —— 直移跟随模式 + 500Hz插值跟随：
        #    检测线程只负责测量与估计（选目标/速度估计/增益自适应/发布状态快照），
        #    鼠标发射在500Hz控制线程按外推误差连续进行——响应提前最多一个检测周期，
        #    运动轨迹从阶梯变连续，检测链路延迟被实时补偿。 ——
        now = time.perf_counter()
        # —— 新帧(测量)间隔：控制律的唯一时间基准。WGC只在画面变化时交付新帧，
        #    重复帧tick(推理全速空转)不计入——按推理tick折算控制律会放大等效增益
        if new_frame:
            md = now - self._meas_t
            if 0.002 <= md <= 0.06:   # 排除首帧与闪断(>60ms)超宽限
                self._meas_dt = 0.7 * self._meas_dt + 0.3 * md
            self._meas_t = now
        best, best_d, best_box = None, None, None
        prev = self._tgt
        # 选目标/速度估计/增益自适应仅新帧参与：重复帧画面未变，结果相同，
        # 参与会把速度估计稀释向0（dxr=0）并污染增益样本
        if dets and new_frame:
            rx, ry = self.region["left"], self.region["top"]
            # 准星参考 = 捕获区域中心（FPS准星恒在屏幕中心，区域即以准星为中心开设）。
            # 不用 get_cursor_pos()：游戏raw input下OS光标会被推到屏幕边缘卡死，
            # 误差参考点失真（实测误差虚增到1300px）→ 控制完全失效
            cx = rx + self.region["width"] / 2.0
            cy = ry + self.region["height"] / 2.0
            off_y = float(self.cfg.get("aim_offset_y", 0.0))
            fov = float(self.cfg.get("fov_radius", 0) or 0)
            cands = []
            for d in dets:
                bx, by = (d[0] + d[2]) / 2, (d[1] + d[3]) / 2
                if fov > 0 and (bx - (cx - rx)) ** 2 + (by - (cy - ry)) ** 2 > fov * fov:
                    continue
                sx = rx + bx
                sy = ry + by + off_y * (d[3] - d[1]) / 2
                cands.append((d, sx, sy, (sx - cx) ** 2 + (sy - cy) ** 2))
            # —— 目标策略：最近中心 / 最高置信度（basic候选），再叠加持久化 ——
            # cands 可为空：fov>0 时所有检测目标都在范围圈外是合法状态——
            # 旧代码直接 min/max(空列表) 抛 ValueError，捕获线程每帧异常、
            # _aim 不再发布且不清除 → 控制线程永远拿不到新鲜目标 → 吸附失效
            near = None
            if cands:
                if self.cfg.get("strategy") == "最高置信度":
                    near = max(cands, key=lambda c: c[0][4])
                else:
                    near = min(cands, key=lambda c: c[3])
            # —— 选目标持久化：优先延续上一帧目标（60px预测门控内），
            #    除非另有目标近80px以上才切换——防止两个相近候选间逐帧翻转
            #    （检测噪声让"最近中心"每帧换人 → 准星左右甩）——
            pick = None
            if prev is not None and cands:
                dt_p = min(max(now - prev["t"], 0.0), 0.12)
                px_ = prev["c"][0] + self._v[0] * dt_p
                py_ = prev["c"][1] + self._v[1] * dt_p
                pm, pd2 = None, 1e18
                for c in cands:
                    dd2 = (c[1] - px_) ** 2 + (c[2] - py_) ** 2
                    if dd2 < pd2:
                        pm, pd2 = c, dd2
                if pd2 <= 60.0 ** 2 and math.sqrt(pm[3]) <= math.sqrt(near[3]) + 80.0:
                    pick = pm
            if pick is None:
                pick = near
            if pick is not None:
                best_box, best = pick[0], (pick[1], pick[2])
                best_d = math.sqrt(pick[3])
        # —— 检测线程只测量，不发射：选目标/速度估计/增益采样/发布状态快照 ——
        if best is not None:
            ex, ey = best[0] - cx, best[1] - cy
            dp_win = self._dp   # 上个测量窗内控制线程累计的屏幕位移(count)
            # —— 目标速度估计（外推/前馈与增益去污染共用）——
            self._update_target_velocity(best[0], best[1], best_box, now, (ex, ey))
            # —— 增益自适应（带符号投影+目标运动补偿）：以整个测量窗累计位移为样本，
            #    比旧版单帧位移信噪比更高；误差变化扣除目标自身位移(v·dt)后，
            #    才纯粹反映鼠标移动效果 ——
            if self._g_err is not None:
                dt_g = now - self._g_t
                if dt_g <= 0.06:
                    # 仅配对间隔≤60ms的测量窗——跨闪断间隔目标运动量未知，样本必然失真
                    ldx, ldy = dp_win
                    ln2 = ldx * ldx + ldy * ldy
                    if ln2 >= 36.0:   # 仅采信大位移窗（|dp|>=6count）
                        # 误差在窗内穿零=上一窗过冲：此刻的"误差变化量"主要由过冲
                        # 回拉贡献，投影像径全失真。旧版把这种毒样本照单全收(±40%
                        # 大步长更新)→增益随机游走0.8~4.0(日志实测)→左右摇晃的
                        # 根源之一。过冲窗一律拒绝。
                        crossed = (self._g_err[0] * ex < 0) or (self._g_err[1] * ey < 0)
                        if crossed:
                            # 过冲=有效步进比例>1（g_est偏低 或 aim_ratio过高）。
                            # 若>0.5s持续只有过冲窗、无任何有效样本（增益卡死在
                            # 过低值、步步过冲的死锁特征），缓升2%逃逸；正常情况
                            # 下有效样本持续到达，不会触发。
                            if now - self._g_valid_t > 0.5:
                                self._gain_est = min(8.0, self._gain_est * 1.02)
                                self._gain_dirty = True
                        else:
                            g_raw = (((self._g_err[0] + self._v[0] * dt_g) - ex) * ldx +
                                     ((self._g_err[1] + self._v[1] * dt_g) - ey) * ldy) / ln2
                            if 0.4 <= g_raw <= 8.0:
                                self._g_valid_t = now
                                # 中位数滤波：6样本窗内单帧野值无法移动中位数，
                                # 杜绝旧版EMA被单个毒样本鞭甩
                                hist = self._g_hist
                                hist.append(g_raw)
                                if len(hist) > 6:
                                    del hist[0]
                                if len(hist) >= 3:
                                    hs = sorted(hist)
                                    m = len(hs) // 2
                                    g_med = hs[m] if len(hs) % 2 else 0.5 * (hs[m - 1] + hs[m])
                                    # 慢步长：每次测量最多移5%，约0.5~1秒收敛到真实增益
                                    step = max(-0.05 * self._gain_est,
                                               min(0.05 * self._gain_est,
                                                   0.12 * (g_med - self._gain_est)))
                                    self._gain_est = min(8.0, max(0.5, self._gain_est + step))
                                    self._gain_dirty = True
            self._g_err, self._g_t = (ex, ey), now
            # —— 捕获链路延迟：WGC画面滞后真实世界约1~1.5帧。
            #    自动补偿=0.75帧/刷新率：补偿量大于滞后会超前冲过目标
            #    （1.5帧实测过冲——比例反馈已隐式吸收部分滞后，只补差额）；
            #    GUI可手动覆盖（capture_delay_ms≥0生效）——
            cd = float(self.cfg.get("capture_delay_ms", -1))
            cap_delay = cd / 1000.0 if cd >= 0 else 0.75 / max(self._refresh_hz, 30)
            # —— 发布跟随状态（500Hz控制线程读取并外推发射）——
            with self.lock:
                self._aim = {"err": (ex, ey), "v": (self._v[0], self._v[1]),
                             "v_age": now - self._v_t, "t": now, "dist": best_d,
                             "delay": cap_delay}
        else:
            if new_frame:
                # 检测闪断宽限：0.12s内保留目标记忆与速度估计不清零，
                # 防止单帧漏检让外推/前馈反复开关抽搐（摇晃来源之一）
                # （重复帧不清零：画面未变，目标必然还在）
                if self._tgt is None or now - self._tgt["t"] > 0.12:
                    self._v = [0.0, 0.0]
                    self._tgt = None
                with self.lock:
                    self._aim = None
            # 重复帧：保持当前aim与速度估计不变
        dt = now - self._last_tick
        # fps显示平滑(0.9)：新帧(6-8ms)/重复帧(3-4ms)交替会让0.3系数的数字
        # 在39~217间跳动，看起来"帧率不稳"；0.9只显示真实均值
        self.fps = 0.9 * self.fps + 0.1 * (1.0 / max(min(dt, 0.5), 0.001))
        self._dt_avg = 0.7 * self._dt_avg + 0.3 * dt   # 推理tick周期（仅记录，控制律用 _meas_dt）
        self._last_tick = now
        self.infer_ms = ms
        with self.lock:
            if new_frame:
                self.num_dets = len(dets)
                self.boxes = dets
            # 回读失败（Map偶发错）时保留上一帧：预览不出现黑屏跳变。
            # readback每次返回全新分配的数组，直接发布引用即可——预览线程在
            # 旧引用上绘制不影响后续发布（发布只替换引用，不改旧数组内容）
            if self.preview_on and frame is not None:
                self.frame = frame
                self.frame_seq += 1
        # 检测帧率上限（默认240Hz）：无新帧也持续推理当前纹理（帧率=推理速率），
        # 多应用共享GPU时可在GUI调低省占用
        wait = 1.0 / max(30.0, float(self.cfg.get("detect_fps", 240))) - (time.perf_counter() - t0)
        if wait > 0:
            time.sleep(wait)
    def _control_worker(self):
        """500Hz跟随控制线程：读取检测线程发布的跟随状态，外推到"此刻"误差后
        连续发射鼠标位移——响应比"检测一帧动一次"提前最多一个检测周期，
        运动轨迹从阶梯变连续，检测链路延迟被实时补偿。"""
        self._boost_current_thread(pin=1)
        interval = 1.0 / CONTROL_HZ
        last = time.perf_counter()
        while not self._exit.is_set():
            now = time.perf_counter()
            dt = min(max(now - last, 0.0005), 0.02)   # tick时长（异常大步长钳制）
            last = now
            self._persist_gain()   # 增益收敛值节流落盘(2s)，供下次启动热启动
            # 紧急停止热键：推理中按下立即终止所有控制
            if self.running and key_pressed(int(self.cfg.get("panic_vk", 0x23))):
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
            pressed = key_pressed(int(self.cfg.get("vk", 0x06))) and self._window_ok()
            st = self._aim
            # 总开关是硬闸门：关闭时无论触发热键/检测结果如何都不发鼠标位移
            aiming = self.aim_enabled and pressed and st is not None and (now - st["t"]) <= 0.2
            if not aiming:
                if not self.aim_enabled:
                    self.state = "跟随已关闭"
                elif pressed:
                    self.state = "搜索目标"
                else:
                    self.state = "运行中"
                self.lock_flag = False
                self._acc = [0.0, 0.0]
                self._acc_ff = [0.0, 0.0]
                self._ff_on = [False, False]   # 前馈迟滞状态复位（重新按压从关起步）
                self._flying = [0.0, 0.0]   # 停止发射时在途欠账作废（画面会自行反映）
            else:
                # —— 外推：误差 + 目标速度×(检测龄期+捕获链路延迟) =
                #    目标"此刻"的位置。捕获延迟是画面本身的龄期（WGC交付约
                #    2帧前的合成画面），不补则横移目标永远落后半步 ——
                age = min(max(now - st["t"], 0.0), 0.1)
                dly = min(max(float(st.get("delay", 0.0)), 0.0), 0.12)
                # 速度估计新鲜度：断流2帧(≈30ms)内全权重，之后才线性衰减。
                # 正常运行 age+v_age≈2帧，旧公式 1-(age+v_age)/0.15 会把前馈
                # 常驻削掉~20% → 稳态落后 v×20%×反馈延迟 → 吸不进框
                fs = max(0.0, min(1.0, 1.0 - max(0.0, (age + st["v_age"]) - 0.03) / 0.12))
                # 前馈迟滞开关：>150px/s开启、<40px/s关闭。旧版固定40px/s单阈值
                # 无迟滞——检测框噪声让|v|在阈值附近反复穿越，前馈高频开关抽搐。
                # 开启阈值取150：框噪声经离群拒绝+EMA后的残余速度σ≈80px/s，
                # 150可确保噪声不触发；真实横移(>150px/s)100ms内即开启 ——
                ax_ = abs(st["v"][0]) * fs
                ay_ = abs(st["v"][1]) * fs
                self._ff_on[0] = ax_ > 40.0 if self._ff_on[0] else ax_ > 150.0
                self._ff_on[1] = ay_ > 40.0 if self._ff_on[1] else ay_ > 150.0
                vx = st["v"][0] * fs if self._ff_on[0] else 0.0
                vy = st["v"][1] * fs if self._ff_on[1] else 0.0
                ex = st["err"][0] + vx * (age + dly)
                ey = st["err"][1] + vy * (age + dly)
                dist = (ex * ex + ey * ey) ** 0.5
                self.target_err = dist
                self.lock_flag = dist < float(self.cfg.get("lock_radius", 15.0))
                self.state = "已跟随" if self.lock_flag else "跟随中"
                dead = max(1.0, float(self.cfg.get("deadzone", 1.5)))
                # —— 在途位移抵扣（Smith预估器思想，治震荡的根本）：
                #    发送的位移要 ~3帧+测量周期 后才反映进画面，期间误差观测
                #    不变。若每tick都按同一误差发比例步进，等效增益被放大
                #    (反馈延迟/测量周期)倍 → 一步消除~97%误差 → 剧烈往返震荡。
                #    有效误差 = 外推误差 - 已发送未反映的位移。
                #    衰减基准用测量周期(_meas_dt)而非推理tick：真实反馈回路
                #    = 游戏渲染+合成器+WGC交付+推理，比旧公式(dt_avg≈2~4ms
                #    +2.5帧)更接近实测25~40ms端到端延迟，且方向偏保守（欠发
                #    送=迟缓但稳定，过发送=震荡） ——
                D_fb = max(self._meas_dt + 3.0 / max(self._refresh_hz, 30), 0.030)
                decay = min(1.0, dt / D_fb)
                self._flying[0] *= 1.0 - decay
                self._flying[1] *= 1.0 - decay
                eff_ex = ex - self._flying[0]
                eff_ey = ey - self._flying[1]
                dist_eff = (eff_ex * eff_ex + eff_ey * eff_ey) ** 0.5
                if dist_eff < dead:
                    self._acc = [0.0, 0.0]
                    self._acc_ff = [0.0, 0.0]
                else:
                    # —— 比例步进按tick折算：每测量周期(新帧间隔)消除 aim_ratio
                    #    比例误差，500Hz下按 (dt/T) 指数折算到每tick，收敛速率与
                    #    帧率无关。T必须用测量周期——旧版用推理tick(含重复帧，
                    #    2~4ms)折算，实际每帧消除量被放大2~4倍=等效死拍 → 震荡 ——
                    ratio = max(0.05, min(1.0, float(self.cfg.get("aim_ratio", 0.45))))
                    # 近距压制放宽（旧 0.6/40px 导致末段30px内收敛过慢=贴不住）
                    ratio *= max(0.7, min(1.0, dist_eff / 30.0))
                    T = max(self._meas_dt, 0.004)
                    r_tick = 1.0 - (1.0 - min(ratio, 0.9)) ** (dt / T)
                    # —— 双通道累计：比例通道(消残差)+前馈通道(速度匹配)分开
                    #    取整发送。在途欠账只记比例通道：前馈移动与目标移动对
                    #    误差的净影响≈0（测量自动体现，不欠账）；混记会把稳态
                    #    误差压低 v×反馈延迟 → 准星永远差半步吸不进框 ——
                    self._acc[0] += eff_ex * r_tick / self._gain_est
                    self._acc[1] += eff_ey * r_tick / self._gain_est
                    self._acc_ff[0] += vx * dt / self._gain_est
                    self._acc_ff[1] += vy * dt / self._gain_est
                    ax, ay = self._acc
                    n = (ax * ax + ay * ay) ** 0.5
                    if n > 90.0:    # 单tick比例位移上限：防异常一帧飞过
                        self._acc[0] = ax = ax * 90.0 / n
                        self._acc[1] = ay = ay * 90.0 / n
                    dpx, dpy = math.floor(ax), math.floor(ay)
                    if dpx or dpy:
                        self._acc[0] -= dpx
                        self._acc[1] -= dpy
                        # 比例步进记入在途欠账（发送后到画面反映前）
                        self._flying[0] += dpx * self._gain_est
                        self._flying[1] += dpy * self._gain_est
                    dfx = math.floor(self._acc_ff[0])
                    dfy = math.floor(self._acc_ff[1])
                    if dfx or dfy:
                        self._acc_ff[0] -= dfx
                        self._acc_ff[1] -= dfy
                    dx, dy = dpx + dfx, dpy + dfy
                    if dx or dy:
                        self._dp[0] += dx
                        self._dp[1] += dy
                        try:
                            # Y轴反向：引擎内部全程屏幕坐标约定
                            # （dp/增益/速度估计均在此空间），仅发送瞬间按配置翻转
                            self.backend.move(dx, -dy if self.cfg.get("invert_y") else dy)
                        except Exception as e:
                            if now - self._move_fail_t > 2.0:
                                self._move_fail_t = now
                                self.log(f"[鼠标] 位移发送失败：{e}")
                    if now - self._diag_t > 0.25:
                        self._diag_t = now
                        self.log(f"[跟随] 距离={dist:.0f}px "
                                 f"速度=({st['v'][0]:+.0f},{st['v'][1]:+.0f}) "
                                 f"增益={self._gain_est:.2f} 移动=({dx},{dy}) "
                                 f"帧距={self._meas_dt * 1000:.0f}ms "
                                 f"tick={dt * 1000:.1f}ms")
            # 跟随持续计时
            if self.lock_flag:
                self.lock_time_total += dt
            time.sleep(interval)
    def _window_ok(self):
        """生效窗口过滤：前台窗口标题需包含关键字（留空=全局生效）"""
        filt = (self.cfg.get("win_filter") or "").strip()
        if not filt:
            return True
        return filt.lower() in foreground_title().lower()
    def _force_stop(self):
        self.running = False
        self._cap_closed.wait(timeout=3.0)
        self._persist_gain(force=True)   # 紧急停止也保留已收敛的标定值
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
                cv2.line(frame, (cx - 7 * scale, cy), (cx + 7 * scale, cy), (60, 255, 60), thin)
                cv2.line(frame, (cx, cy - 7 * scale), (cx, cy + 7 * scale), (60, 255, 60), thin)
                # 辅助参数可视化（与跟随逻辑同一参数，均以准星/画面中心为圆心）：
                # 跟随范围 fov_radius（黄）、跟随判定 lock_radius（青）、收尾死区 deadzone（红）
                fov = float(self.cfg.get("fov_radius", 0) or 0)
                lr = float(self.cfg.get("lock_radius", 15) or 0)
                dz = max(1.0, float(self.cfg.get("deadzone", 2.0) or 0))
                if fov > 0:
                    cv2.circle(frame, (cx, cy), int(fov), (0, 200, 255), thin)
                    cv2.putText(frame, f"FOV {fov:.0f}", (cx + int(fov) - 24 * scale,
                                max(14, cy - int(fov) - 4)), cv2.FONT_HERSHEY_SIMPLEX,
                                0.4, (0, 200, 255), 1, cv2.LINE_AA)
                if lr > 0:
                    cv2.circle(frame, (cx, cy), int(lr), (255, 160, 0), max(1, thin - 1))
                cv2.circle(frame, (cx, cy), int(dz), (60, 60, 255), max(1, thin - 1))
                cv2.putText(frame, f"FOV {'OFF' if fov <= 0 else f'{fov:.0f}px'}  "
                            f"LOCK {lr:.0f}px  DEAD {dz:.1f}px",
                            (8, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                            (180, 180, 180), 1, cv2.LINE_AA)
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
        elif key == "cam_g":
            # 联动系数调整=重播种自适应增益（在线校准以此为起点继续微调）
            self._gain_est = min(8.0, max(0.5, float(value)))
            self._g_hist = []   # 旧增益样本属于旧灵敏度 regime，全部作废
            with self.lock:
                self.cfg["gain_est_saved"] = round(float(self._gain_est), 4)
            self._save()   # 显式重校准意图立即落盘，防止旧灵敏度下的旧收敛值被热启动继承
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
