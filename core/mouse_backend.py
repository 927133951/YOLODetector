# -*- coding: utf-8 -*-
"""鼠标控制双后端：罗技驱动DLL（MouseControl.dll）与 Windows SendInput"""
import os
import json
import ctypes
import atexit
from ctypes import wintypes
user32 = ctypes.windll.user32
SPI_GETMOUSE, SPI_SETMOUSE = 0x0003, 0x0004
SPI_GETMOUSESPEED, SPI_SETMOUSESPEED = 0x0070, 0x0071
class SendInputBackend:
    """Windows SendInput 相对移动后端
    运行期间临时关闭"提高指针精确度"与指针加速（设为标准10档=1:1像素移动），
    保证控制输出与光标位移严格1:1，进程退出时恢复用户原始设置
    （罗技DLL后端为驱动级移动，天然1:1不受系统缩放影响）"""
    name = "SendInput"
    def __init__(self):
        class MOUSEINPUT(ctypes.Structure):
            _fields_ = [("dx", wintypes.LONG), ("dy", wintypes.LONG),
                        ("mouseData", wintypes.DWORD), ("dwFlags", wintypes.DWORD),
                        ("time", wintypes.DWORD), ("dwExtraInfo", ctypes.c_size_t)]
        class _INPUTUNION(ctypes.Union):
            _fields_ = [("mi", MOUSEINPUT)]
        class INPUT(ctypes.Structure):
            _fields_ = [("type", wintypes.DWORD), ("union", _INPUTUNION)]
        self.MOUSEEVENTF_MOVE = 0x0001
        self.MOUSEEVENTF_ABSOLUTE = 0x8000
        self.MOUSEEVENTF_VIRTUALDESK = 0x4000
        self.INPUT_MOUSE = 0
        self.INPUT = INPUT
        self.size = ctypes.sizeof(INPUT)
        self._bak_file = os.path.join(os.environ.get("LOCALAPPDATA", os.path.expanduser("~")),
                                      "CCTools", "pointer_backup.json")
        self._recover_crashed()
        self._old_mouse = (wintypes.INT * 3)()
        self._old_speed = wintypes.UINT()
        user32.SystemParametersInfoW(SPI_GETMOUSE, 0, self._old_mouse, 0)
        user32.SystemParametersInfoW(SPI_GETMOUSESPEED, 0, ctypes.byref(self._old_speed), 0)
        self._save_backup()
        std = (wintypes.INT * 3)(6, 10, 0)
        user32.SystemParametersInfoW(SPI_SETMOUSE, 0, std, 0)
        user32.SystemParametersInfoW(SPI_SETMOUSESPEED, 0, wintypes.UINT(10), 1)
        print("[鼠标] 已临时切换标准1:1指针模式（关闭指针加速），退出时自动恢复原设置")
        atexit.register(self._restore_system)
    def _recover_crashed(self):
        """上次进程异常退出未恢复指针设置时，先还原用户原始设置"""
        try:
            if not os.path.isfile(self._bak_file):
                return
            with open(self._bak_file, "r", encoding="utf-8") as f:
                bak = json.load(f)
            user32.SystemParametersInfoW(SPI_SETMOUSE, 0, (wintypes.INT * 3)(*bak["mouse"]), 0)
            user32.SystemParametersInfoW(SPI_SETMOUSESPEED, 0, wintypes.UINT(int(bak["speed"])), 1)
            os.remove(self._bak_file)
            print("[鼠标] 检测到上次异常退出，已恢复指针加速原设置")
        except Exception:
            pass
    def _save_backup(self):
        """备份当前指针设置到磁盘，进程崩溃后下次启动可自恢复"""
        try:
            os.makedirs(os.path.dirname(self._bak_file), exist_ok=True)
            with open(self._bak_file, "w", encoding="utf-8") as f:
                json.dump({"mouse": list(self._old_mouse),
                           "speed": int(self._old_speed.value)}, f)
        except Exception:
            pass
    def _raw_send(self, dx, dy):
        inp = self.INPUT()
        inp.type = self.INPUT_MOUSE
        inp.union.mi.dx = int(dx)
        inp.union.mi.dy = int(dy)
        inp.union.mi.dwFlags = self.MOUSEEVENTF_MOVE
        user32.SendInput(1, ctypes.byref(inp), self.size)
    def _restore_system(self):
        try:
            user32.SystemParametersInfoW(SPI_SETMOUSE, 0, self._old_mouse, 0)
            user32.SystemParametersInfoW(SPI_SETMOUSESPEED, 0, ctypes.byref(self._old_speed), 1)
        except Exception:
            pass
        try:
            if os.path.isfile(self._bak_file):
                os.remove(self._bak_file)
        except Exception:
            pass
    def move(self, dx, dy):
        self._raw_send(int(dx), int(dy))
    def move_abs(self, x, y):
        """AimYolo原生绝对移动（z_ctypes.py同款）：光标直接定位到虚拟屏幕坐标(x,y)。
        MOUSEEVENTF_ABSOLUTE按65535归一化；加VIRTUALDESK以整个虚拟桌面为归一化基准
        （捕获region的left/top是虚拟屏幕坐标，不加此标志只对主屏正确）"""
        vx = int(user32.GetSystemMetrics(76))    # SM_XVIRTUALSCREEN
        vy = int(user32.GetSystemMetrics(77))    # SM_YVIRTUALSCREEN
        vw = max(1, int(user32.GetSystemMetrics(78)))   # SM_CXVIRTUALSCREEN
        vh = max(1, int(user32.GetSystemMetrics(79)))   # SM_CYVIRTUALSCREEN
        inp = self.INPUT()
        inp.type = self.INPUT_MOUSE
        inp.union.mi.dx = int((x - vx) * 65535 / vw)
        inp.union.mi.dy = int((y - vy) * 65535 / vh)
        inp.union.mi.dwFlags = (self.MOUSEEVENTF_MOVE | self.MOUSEEVENTF_ABSOLUTE
                                | self.MOUSEEVENTF_VIRTUALDESK)
        user32.SendInput(1, ctypes.byref(inp), self.size)
class LogitechBackend:
    """罗技驱动DLL后端（MouseControl.dll 驱动级相对移动）"""
    name = "罗技驱动"
    def __init__(self, dll_path):
        self.dll = ctypes.CDLL(dll_path)
        self.dll.move_R.argtypes = [ctypes.c_int, ctypes.c_int]
        self.dll.move_R.restype = None
        print("[鼠标] 罗技驱动DLL加载成功：MouseControl.dll")
    def move(self, dx, dy):
        self.dll.move_R(int(dx), int(dy))
def get_cursor_pos():
    pt = wintypes.POINT()
    user32.GetCursorPos(ctypes.byref(pt))
    return pt.x, pt.y
def key_pressed(vk):
    return bool(user32.GetAsyncKeyState(int(vk)) & 0x8000)
def foreground_title():
    """当前前台窗口标题（用于生效窗口过滤）"""
    try:
        hwnd = user32.GetForegroundWindow()
        if not hwnd:
            return ""
        n = user32.GetWindowTextLengthW(hwnd)
        buf = ctypes.create_unicode_buffer(n + 1)
        user32.GetWindowTextW(hwnd, buf, n + 1)
        return buf.value
    except Exception:
        return ""
