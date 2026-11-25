# -*- coding: utf-8 -*-
"""全局配置"""
import os
import sys
import json
APP_NAME = "CCTools"
def _resolve_dirs():
    """源码运行：项目根目录；PyInstaller打包：APP_DIR=exe所在目录(可写,存配置/日志)，
    BUNDLE_DIR=资源目录(sys._MEIPASS，onedir模式下指向_internal)"""
    if getattr(sys, "frozen", False):
        app_dir = os.path.dirname(os.path.abspath(sys.executable))
        bundle = getattr(sys, "_MEIPASS", app_dir)
        return app_dir, bundle
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return root, root
APP_DIR, BUNDLE_DIR = _resolve_dirs()
APP_ROOT = os.path.dirname(APP_DIR)
PROJECT_ROOT = APP_DIR
def _first_file(*candidates):
    for c in candidates:
        if c and os.path.isfile(c):
            return c
    return candidates[-1] if candidates else ""
ICON_PATH = _first_file(
    os.path.join(BUNDLE_DIR, "assets", "icon.ico"),
    os.path.join(APP_DIR, "assets", "icon.ico"))
SETTINGS_PATH = os.path.join(APP_DIR, "cctools_settings.json")
LOG_DIR = os.path.join(APP_DIR, "logs")
def _model_path():
    """优先用exe旁models目录的模型（可替换/更新），其次打包内置模型，最后回退Downloads原始位置"""
    return _first_file(
        os.path.join(APP_DIR, "models", "v26m_0626_640.onnx"),
        os.path.join(BUNDLE_DIR, "models", "v26m_0626_640.onnx"),
        r"C:\Users\92713\Downloads\model\v26m_0626_640.onnx")
MODEL_PATH = _model_path()
def _dll_path():
    """优先exe旁MouseControl目录（可替换），其次打包内置DLL"""
    return _first_file(
        os.path.join(APP_DIR, "MouseControl", "MouseControl.dll"),
        os.path.join(BUNDLE_DIR, "MouseControl", "MouseControl.dll"),
        os.path.join(APP_ROOT, "MouseControl", "MouseControl.dll"))
LOGITECH_DLL = _dll_path()
INPUT_SIZE = 640
MAX_FPS = 240
CONTROL_HZ = 500
HOTKEY_VK = {
    "未设置": 0,
    "鼠标左键": 0x01, "鼠标右键": 0x02, "鼠标中键": 0x04,
    "鼠标侧键X1": 0x05, "鼠标侧键X2": 0x06,
    "右Ctrl": 0xA3, "左Ctrl": 0xA2, "左Alt": 0xA4, "左Shift": 0xA1,
    "CapsLock": 0x14, "Tab": 0x09, "End": 0x23,
}
HOTKEY_VK.update({f"F{i}": 0x6F + i for i in range(1, 13)})
HOTKEY_VK.update({f"数字{i}": 0x30 + i for i in range(10)})
HOTKEY_VK.update({chr(0x41 + i): 0x41 + i for i in range(26)})
DEFAULTS = {
    "conf": 0.45, "capture_size": 640, "detect_fps": 240,
    "selected_classes": None,
    "hotkey_name": "鼠标侧键X2",
    "vk": 0x06, "panic_name": "End", "panic_vk": 0x23,
    "aim_enable_name": "F8", "aim_enable_vk": 0x77,
    "nms_iou": 0.45, "crosshair_dx": 0, "crosshair_dy": 0,
    "monitor": None, "win_filter": "", "anti_capture": True,
    # —— AimYolo原生算法参数（含鼠标/目标策略）——
    "strategy": "最近中心",       # 目标选择：最近中心 / 最高置信度
    "backend": "logitech",       # 鼠标后端：logitech / sendinput
    "invert_y": False,           # Y轴反向（相对移动路径有效）
    "fov_radius": 0.0,           # FOV范围(px)：仅准星该半径内目标可选，0=不限
    "lock_radius": 15.0,         # 跟随判定(px)：状态显示LOCKED阈值
    "aim_offset_y": 0.0,         # 跟随点Y偏移：-0.5框顶 0中心 +0.5框底
    "aim_speed": 0.3,            # 鼠标速度：<1.0=每帧走误差百分比(平滑收敛)；1.0=直满(SendInput原生)
    "sens_k": 1.0,               # 视角增益K：鼠标动1px时目标在屏幕反向移动的px数；步长=误差/K
    "deadzone": 2.0,             # 收尾死区(px)：距离小于该值不动
    "aim_lead": 0.0,             # 移动目标预测提前(ms)：按目标速度外推瞄准点，0=关闭
}
def load_settings():
    """启动时加载持久化配置（仅接受已知键，避免脏数据）"""
    try:
        with open(SETTINGS_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            return {k: v for k, v in data.items() if k in DEFAULTS or k == "model_path"}
    except Exception:
        pass
    return {}
# 一键重置用的通用稳定参数（不含 capture_size——与模型训练尺度绑定，不参与重置）
RESET_PARAMS = {
    "conf": 0.5, "nms_iou": 0.45, "detect_fps": 240,
    "strategy": "最近中心", "fov_radius": 0.0, "lock_radius": 15.0,
    "aim_offset_y": 0.0, "aim_speed": 0.3, "deadzone": 2.0, "sens_k": 1.0,
    "aim_lead": 0.0,
}
def save_settings(cfg):
    """参数变更后写入本地配置，重启自动恢复"""
    try:
        data = {k: (sorted(v) if isinstance(v, set) else v)
                for k, v in cfg.items() if k in DEFAULTS or k == "model_path"}
        with open(SETTINGS_PATH, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=1)
    except Exception:
        pass
