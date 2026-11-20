# -*- coding: utf-8 -*-
"""窗口防截屏防护：调用 SetWindowDisplayAffinity 让窗口对截图/录屏/远程共享软件不可见
适用于微信/QQ 截图、OBS 等录屏、远程会议共享、远程软件画面捕获等。
Win10 2004+ 支持完全排除捕获，旧系统自动降级为“捕获时黑屏”。"""
import ctypes

user32 = ctypes.windll.user32
WDA_NONE = 0x0
WDA_MONITOR = 0x1          # 旧系统降级：捕获画面中显示黑屏
WDA_EXCLUDE = 0x11         # Win10 2004+：捕获画面中完全不可见


def find_window(title):
    """按标题查找顶层窗口句柄，找不到返回 0"""
    try:
        return user32.FindWindowW(None, title) or 0
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
