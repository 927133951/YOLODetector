# -*- coding: utf-8 -*-
"""CCTools - 主程序入口"""
import os
import sys
import ctypes
import traceback
import threading
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
def _setup_frozen_logging():
    """打包版无控制台：stdout/stderr 同步写入 exe旁logs/startup.log，
    便于在其他设备上排查启动/运行问题（源码模式不启用）"""
    if not getattr(sys, "frozen", False):
        return
    try:
        app_dir = os.path.dirname(os.path.abspath(sys.executable))
        log_dir = os.path.join(app_dir, "logs")
        os.makedirs(log_dir, exist_ok=True)
        log_path = os.path.join(log_dir, "startup.log")
        class _Tee:
            def __init__(self, *streams):
                self.streams = streams
            def write(self, s):
                for st in self.streams:
                    try:
                        st.write(s)
                    except Exception:
                        pass
                try:
                    with open(log_path, "a", encoding="utf-8", errors="replace") as f:
                        f.write(s)
                except Exception:
                    pass
            def flush(self):
                for st in self.streams:
                    try:
                        st.flush()
                    except Exception:
                        pass
        sys.stdout = _Tee(sys.stdout)
        sys.stderr = _Tee(sys.stderr)
        try:
            import faulthandler
            # 必须用真实文件句柄（fd）：faulthandler在原生崩溃时通过fd直写，
            # Tee无fileno会让enable静默失效——崩溃时拿不到Python调用栈
            global _crash_log_f
            _crash_log_f = open(log_path, "a", encoding="utf-8", errors="replace")
            faulthandler.enable(file=_crash_log_f)
        except Exception:
            pass
        def _hook(t, v, tb):
            print("[未处理异常]", t.__name__, v, "\n", "".join(traceback.format_tb(tb)))
        sys.excepthook = _hook
        def _thread_hook(args):
            print("[线程异常]", args.thread.name if args.thread else "?", args.exc_type.__name__, args.exc_value, "\n", "".join(traceback.format_tb(args.exc_traceback)))
        threading.excepthook = _thread_hook
    except Exception:
        pass

def _ensure_admin():
    """未以管理员运行时自动弹UAC重启自身（游戏内注入/驱动鼠标需要管理员权限）"""
    if os.environ.get("CCTOOLS_NO_ELEVATE") == "1":
        print("[启动] 调试模式：跳过UAC提权")
        return
    try:
        if ctypes.windll.shell32.IsUserAnAdmin():
            return
        params = " ".join(f'"{a}"' for a in sys.argv[1:]) or f'"{os.path.abspath(sys.argv[0])}"'
        if os.path.splitext(sys.executable)[1].lower() == ".exe" and "python" not in os.path.basename(sys.executable).lower():
            target, args = sys.executable, params
        else:
            target, args = sys.executable, f'"{os.path.abspath(sys.argv[0])}"'
        ret = ctypes.windll.shell32.ShellExecuteW(None, "runas", target, args, None, 1)
        if ret > 32:
            sys.exit(0)
        print("[启动] 未获得管理员权限，游戏内鼠标控制/热键可能失效")
    except Exception:
        pass
def _acquire_single_instance():
    """单实例守卫：进程存活期间持有命名互斥体（Windows在进程退出时自动释放，
    无需手动清理——含任务管理器杀进程/崩溃等异常退出）。返回True=成功持有可启动；
    False=已有实例在运行（典型场景：窗口被隐藏/最小化以为没开又双击了一次）"""
    try:
        import ctypes.wintypes as wt
        kernel32 = ctypes.windll.kernel32
        kernel32.CreateMutexW.restype = wt.HANDLE
        kernel32.CreateMutexW.argtypes = [wt.LPCVOID, wt.BOOL, wt.LPCWSTR]
        kernel32.CloseHandle.argtypes = [wt.HANDLE]
        h = kernel32.CreateMutexW(None, False, "Local\\CCTools_YOLODetector_SingleInstance")
        err = kernel32.GetLastError()
        if h:
            if err == 183:          # ERROR_ALREADY_EXISTS：已有实例持有同名互斥体
                kernel32.CloseHandle(h)
                return False
            return True             # 新建成功：句柄不放，进程退出系统自动回收
        # 句柄拿不到且访问被拒 = 高完整性（管理员）实例已持有 → 视为已在运行
        return err != 5             # ERROR_ACCESS_DENIED=5
    except Exception:
        return True                 # 守卫机制自身故障不拦截正常启动
def main():
    _setup_frozen_logging()
    # 单实例检查必须先于UAC提权：否则重复启动会先弹UAC再提示，多一道无意义授权
    if not _acquire_single_instance():
        ctypes.windll.user32.MessageBoxW(
            None,
            "程序已在运行中（窗口可能被隐藏或最小化到任务栏）。\n请勿重复启动，请到任务栏找到已运行的窗口。",
            "CCTools", 0x40040)     # MB_ICONWARNING | MB_TOPMOST：确保能被注意到
        return
    _ensure_admin()
    # GIL切换间隔5ms→1ms：检测线程是CPU密集Python（预处理调度/解码/状态机），
    # 被500Hz控制线程、预览线程、GUI轮询抢走GIL后默认最长5ms才归还——
    # 这是推理延迟比C++版多出数毫秒的主因之一
    sys.setswitchinterval(0.001)
    # GC遍历本应用的原生互操作对象（ORT/TRT会话、CUDA管线、WinRT/D3D捕获
    # 包装）= 原生崩溃：faulthandler实测10+次 access violation 均发生在
    # GC遍历中（python313.dll+0x15ce2）。曾经"加载完freeze再enable"的方案
    # 仍崩溃（freeze只保护存量对象，推理中重建的捕获/会话对象会被遍历）——
    # 故GC在进程生命周期内永久禁用（engine侧不再enable）。引用计数覆盖
    # 绝大部分回收，成环垃圾极少；GC停顿对500Hz控制/240FPS推理也是延迟毛刺
    import gc
    gc.disable()
    ctypes.windll.shcore.SetProcessDpiAwareness(2)
    try:
        # 高优先级：降低推理/控制线程被系统调度抢占造成的延迟抖动
        ctypes.windll.kernel32.SetPriorityClass(
            ctypes.windll.kernel32.CurrentProcess(), 0x00000080)  # HIGH_PRIORITY_CLASS
    except Exception:
        pass
    # 注：不设置 AppUserModelID——设置了AUMID却没有配套开始菜单快捷方式时，
    # 任务栏会回退显示 python.exe 图标；不设置则按窗口iconbitmap显示软件图标
    import tkinter as tk
    from core.engine import Engine
    from core.config import APP_NAME, ICON_PATH
    from ui.gui import App
    root = tk.Tk()
    root.title(APP_NAME)
    # 窗口尺寸自适应屏幕工作区：矮屏/系统DPI缩放下窗口完整显示在屏幕内，
    # 内容超出部分由 UI 内滚动条承接（ui/gui.py _setup_scroll）
    _w, _h = 680, 1010
    try:
        import ctypes.wintypes as _wt
        _wa = _wt.RECT()
        if ctypes.windll.user32.SystemParametersInfoW(0x0030, 0, ctypes.byref(_wa), 0):
            _avail_w, _avail_h = _wa.right - _wa.left, _wa.bottom - _wa.top
            _x0, _y0 = _wa.left, _wa.top
        else:
            _avail_w, _avail_h = root.winfo_screenwidth(), root.winfo_screenheight()
            _x0, _y0 = 0, 0
    except Exception:
        _avail_w, _avail_h = root.winfo_screenwidth(), root.winfo_screenheight()
        _x0, _y0 = 0, 0
    _w = min(_w, _avail_w - 16)
    _h = min(_h, _avail_h - 16)
    root.geometry(f"{_w}x{_h}+{_x0 + max(0, (_avail_w - _w) // 2)}+{_y0 + max(0, (_avail_h - _h) // 2)}")
    root.minsize(min(680, _w), min(900, _h))
    root.configure(bg="#14161a")
    if os.path.isfile(ICON_PATH):
        try:
            root.iconbitmap(ICON_PATH)
        except Exception:
            pass
    engine = Engine(log_fn=print)
    app = App(root, engine)
    root.protocol("WM_DELETE_WINDOW", app.on_close)
    print("[启动] 系统就绪，点击「开始推理」运行实时检测")
    root.mainloop()
if __name__ == "__main__":
    main()
