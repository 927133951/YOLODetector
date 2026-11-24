# -*- coding: utf-8 -*-
"""窗口防截屏防护：调用 SetWindowDisplayAffinity 让窗口对截图/录屏/远程共享软件不可见
适用于微信/QQ 截图、OBS 等录屏、远程会议共享、远程软件画面捕获等。
Win10 2004+ 支持完全排除捕获，旧系统自动降级为“捕获时黑屏”。"""
import ctypes

user32 = ctypes.windll.user32
WDA_NONE = 0x0
WDA_MONITOR = 0x1          # 旧系统降级：捕获画面中显示黑屏
WDA_EXCLUDE = 0x11         # Win10 2004+：捕获画面中完全不可见

GWL_EXSTYLE = -20
WS_EX_TOOLWINDOW = 0x00000080   # 工具窗口样式：不进任务栏与Alt-Tab
WS_EX_APPWINDOW = 0x00040000    # 强制进任务栏样式


def find_window(title):
    """按标题查找顶层窗口句柄，找不到返回 0。
    cv2 HighGUI会把UTF-8标题字节经ANSI API传给Win32——窗口真实标题是GBK乱码
    （如"瀹炴椂妫€..."），FindWindowW(宽字符)永远匹配不上，必须用
    FindWindowA+UTF-8原始字节兜底"""
    try:
        hwnd = user32.FindWindowW(None, title)
        if hwnd:
            return hwnd
        return user32.FindWindowA(None, title.encode("utf-8")) or 0
    except Exception:
        return 0


def apply(hwnd, on):
    """开启/关闭指定窗口的防截屏，返回实际生效模式描述，失败返回 None"""
    if not hwnd:
        return None
    try:
        if on:
            if user32.SetWindowDisplayAffinity(hwnd, WDA_EXCLUDE):
                return "排除捕获"
            if user32.SetWindowDisplayAffinity(hwnd, WDA_MONITOR):
                return "捕获时黑屏"
            return None
        user32.SetWindowDisplayAffinity(hwnd, WDA_NONE)
        return "已关闭"
    except Exception:
        return None


def hide_from_taskbar(hwnd, on):
    """切换窗口在任务栏/Alt-Tab中的可见性（WS_EX_TOOLWINDOW样式）。
    需在窗口首次显示前调用，任务栏按钮/Alt-Tab项才完全不会出现"""
    if not hwnd:
        return False
    try:
        glp = getattr(user32, "GetWindowLongPtrW", None) or user32.GetWindowLongW
        slp = getattr(user32, "SetWindowLongPtrW", None) or user32.SetWindowLongW
        style = glp(hwnd, GWL_EXSTYLE)
        if on:
            style = (style | WS_EX_TOOLWINDOW) & ~WS_EX_APPWINDOW
        else:
            style = (style & ~WS_EX_TOOLWINDOW) | WS_EX_APPWINDOW
        slp(hwnd, GWL_EXSTYLE, style)
        return True
    except Exception:
        return False
