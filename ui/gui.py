# -*- coding: utf-8 -*-
"""主界面：深色主题 tkinter 控制面板（CCTools），含防截屏开关、准星偏移、持久化参数"""
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import os
import time
import queue
import ctypes
import core.config as cfg
from core import capture_guard
from core.mouse_backend import key_pressed
from core.capture import list_monitors
HOTKEY_NAMES = list(cfg.HOTKEY_VK.keys()) + ["自定义"]


class ToolTip:
    """悬停说明气泡：鼠标悬停参数标签/控件后显示调参说明"""
    def __init__(self, widget, text, delay=350):
        self.widget, self.text, self.delay = widget, text, delay
        self._after = None
        self._tw = None
        widget.bind("<Enter>", self._schedule, add="+")
        widget.bind("<Leave>", self._hide, add="+")
        widget.bind("<ButtonPress>", self._hide, add="+")

    def _schedule(self, _e=None):
        self._cancel()
        self._after = self.widget.after(self.delay, self._show)

    def _cancel(self):
        if self._after:
            try:
                self.widget.after_cancel(self._after)
            except Exception:
                pass
            self._after = None

    def _show(self):
        if self._tw is not None:
            return
        x = self.widget.winfo_rootx() + 8
        y = self.widget.winfo_rooty() + self.widget.winfo_height() + 4
        self._tw = tk.Toplevel(self.widget)
        self._tw.wm_overrideredirect(True)
        self._tw.attributes("-topmost", True)
        self._tw.wm_geometry(f"+{x}+{y}")
        frame = tk.Frame(self._tw, bg="#0c0e11", highlightthickness=1,
                         highlightbackground="#00b4ff")
        frame.pack(fill="both", expand=True)
        tk.Label(frame, text=self.text, bg="#0c0e11", fg="#dfe3ea",
                 justify="left", wraplength=440, padx=9, pady=6,
                 font=("Microsoft YaHei UI", 9)).pack()

    def _hide(self, _e=None):
        self._cancel()
        if self._tw is not None:
            self._tw.destroy()
            self._tw = None


# —— 滑条参数说明（key → 说明文案）——
PARAM_TIPS = {
    "aim_ratio": "移动比例：每帧消除剩余误差的比例(0.1~1.0)。\n"
                 "越大→跟得越快、越贴框；过大→易在目标框两侧来回摆(过冲)。\n"
                 "建议0.3~0.45起步；出现过冲/画圈就调小。",
    "aim_offset_y": "跟随点Y偏移：跟随点在目标框内的垂直位置。\n"
                    "-0.5=框顶(上沿)  0=正中心  +0.5=框底。\n"
                    "建议0(最稳)或-0.1~-0.2(略偏上沿)。",
    "fov_radius": "跟随范围：只跟随准星周围该半径(px)内的目标，0=不限制。\n"
                  "防止远处目标被选为跟随对象。建议30~80；越大搜索范围越大。",
    "lock_radius": "跟随判定：误差小于该值(px)时状态栏显示'已跟随'。\n"
                   "仅状态显示用，不影响跟随力度，保持默认即可。",
    "nms_iou": "NMS IoU：重叠检测框的去重阈值。\n"
               "同一目标出现双重框→调低(0.4)；目标密集互相遮挡漏检→调高(0.5)。",
    "deadzone": "收尾死区：误差小于该值(px)时停止移动，\n"
                "防止检测框自身抖动让准星在框内微颤。\n"
                "建议2~3；过大→末段提前停住。",
    "cam_g": "联动系数：1个鼠标count对应的屏幕像素数(与游戏灵敏度相关)。\n"
             "软件会自动校准，此值仅作起点，调整后立即重新校准。\n"
             "仅在自动校准明显不准时微调：鼠标灵敏度高→值调大。",
    "detect_fps": "检测帧率上限：检测线程每秒推理次数上限(60~240)，默认240。\n"
                  "调低→GPU占用大幅下降，其他共用GPU的软件更流畅；\n"
                  "调高→检测更新鲜但GPU占用线性上涨。\n"
                  "多应用共享GPU时建议≤110，纯游戏场景保持240。",
    "capture_delay_ms": "延迟补偿(ms)：目标横移时按画面滞后时长外推超前补偿。\n"
                        "未拖动滑条前=自动(0.75帧/刷新率，60Hz≈12ms)；拖动后为手动值。\n"
                        "横移目标追不上→调大；出现超前冲过目标→调小。",
}
TIP_CONF = ("置信度阈值：检测结果的把握程度门槛(0.1~0.9)。\n"
            "越高→只选有把握的目标(误检少、可能漏检)；越低→检出更多(易误检、易跳目标)。\n"
            "误选背景/道具→调高；远处目标选不到→调低。建议0.45~0.55。")
TIP_CAP = ("检测区域：自动锁定为模型输入尺寸（模型按该尺度训练），不可手动修改。\n"
           "检测范围=游戏屏幕中心该尺寸的正方形区域。")
TIP_MON = "选择游戏所在显示器。多显示器必须选对，否则检测到的是另一块屏幕的画面。"
TIP_OFFSET = ("准星偏移：游戏准星不在屏幕正中心时的手动补偿。X右正左负、Y下正上负(像素)。\n"
              "判断：开启预览后，绿色十字与游戏准星不重合就调这两个值。")
TIP_STRATEGY = ("目标策略：\n最近中心=优先选距离准星最近的目标(推荐，配合自动选点防翻转)；\n"
                "最高置信度=优先选画面里最有把握的目标。")
TIP_ANTI = "防截屏：让本窗口对微信/QQ/OBS等截图、录屏软件不可见，不影响游戏。"
TIP_BACKEND = ("控制后端：\n罗技驱动DLL=驱动级鼠标移动(推荐，1:1精确，不受系统指针加速影响)；\n"
               "SendInput=系统级模拟(需系统关闭'提高指针精确度')。")
TIP_INVERT = "Y轴反向：如果准星上下方向反了(目标在上却往下移)就勾选。"
TIP_HOTKEY = ("触发热键：按住该键时启用跟随辅助。\n"
              "可选预设键，或选'自定义'输入VK码(十进制，可查虚拟键码表)。")
TIP_PANIC = "紧急停止：推理中按下立即停止全部鼠标控制。\n失控时的保底手段，建议保持End。"
TIP_AEN = "跟随辅助开关：按一下在'开启/关闭跟随辅助'之间切换。\n临时想手动操作时用，无需停止推理。"
TIP_WIN = ("生效窗口：只在标题包含该关键字的窗口生效，留空=全局生效。\n"
           "建议填游戏窗口标题的一部分(如'Counter')，防止切到桌面/聊天窗口误触发。")
TIP_RESET = ("一键重置：将检测/跟随调参恢复为通用稳定值\n"
             "(任何设备、任何刷新率下都不会过冲的保守组合)。\n"
             "不会改动：热键、显示器、后端、防截屏、准星偏移、捕获区域、已选类别。")
class App:
    def __init__(self, root, engine):
        self.root = root
        self.engine = engine
        self.previewing = False
        self.class_widgets = {}
        self.class_built = False
        self._logq = queue.Queue()
        engine.log = self._log
        self._build_style()
        self._setup_scroll()
        self._build_ui()
        self._apply_root_guard()
        self._apply_root_taskbar_hide()   # 主窗口默认不进任务栏/Alt-Tab
        self._home_prev = False
        self.root.after(50, self._poll_home_key)   # Home键显隐主界面
        self.root.after(200, self._poll_status)
    def _apply_root_taskbar_hide(self):
        """主窗口默认不进任务栏/Alt-Tab（WS_EX_TOOLWINDOW，无勾选项）"""
        try:
            hwnd = ctypes.windll.user32.GetParent(self.root.winfo_id())
            if hwnd:
                capture_guard.hide_from_taskbar(hwnd, True)
        except Exception:
            pass
    def _poll_home_key(self):
        """Home键显示/隐藏主界面（边沿触发，50ms轮询）。
        焦点在输入框/下拉框内时不触发——Home也是文本编辑键，避免打字误触"""
        try:
            hp = key_pressed(0x24)   # VK_HOME
            if hp and not self._home_prev:
                w = self.root.focus_get()
                if not isinstance(w, (tk.Entry, ttk.Entry, ttk.Combobox, tk.Spinbox)):
                    self._toggle_main_visible()
            self._home_prev = hp
        except Exception:
            pass
        finally:
            self.root.after(50, self._poll_home_key)
    def _toggle_main_visible(self):
        if self.root.state() == "withdrawn":
            self.root.deiconify()
            self.root.lift()
            self.root.focus_force()
            self._log("[窗口] 主界面已显示")
        else:
            self.root.withdraw()
            self._log("[窗口] 主界面已隐藏（按Home恢复显示）")
    def _build_style(self):
        self.bg, self.panel, self.fg = "#14161a", "#1d2026", "#e8eaed"
        self.accent, self.dim, self.green = "#00b4ff", "#8b93a1", "#2ecc71"
        self.root.configure(bg=self.bg)
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("TCombobox", fieldbackground="#22262e", background="#2a2f3a",
                        foreground=self.fg, arrowcolor=self.accent, bordercolor="#333944")
        style.map("TCombobox", fieldbackground=[("readonly", "#22262e")],
                  foreground=[("readonly", self.fg)])
    def _setup_scroll(self):
        """整窗滚动容器：矮屏/系统DPI缩放下窗口高度小于内容时，
        滚轮+右侧滚动条可查看全部内容（开始推理按钮不再被截断）"""
        self.canvas = tk.Canvas(self.root, bg=self.bg, highlightthickness=0, bd=0)
        self.scroll = ttk.Scrollbar(self.root, orient="vertical", command=self.canvas.yview)
        self.canvas.configure(yscrollcommand=self.scroll.set)
        self.canvas.pack(side="left", fill="both", expand=True)
        self.body = tk.Frame(self.canvas, bg=self.bg)
        self._body_win = self.canvas.create_window((0, 0), window=self.body, anchor="nw")
        self.body.bind("<Configure>", self._sync_scroll)
        self.canvas.bind("<Configure>", self._on_canvas_resize)
        self.canvas.bind_all("<MouseWheel>", self._on_mousewheel)
    def _sync_scroll(self, event=None):
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        need = self.body.winfo_height() > self.canvas.winfo_height()
        if need != getattr(self, "_scroll_shown", None):
            self._scroll_shown = need
            if need:
                self.scroll.pack(side="right", fill="y")
            else:
                self.scroll.pack_forget()
    def _on_canvas_resize(self, event):
        self.canvas.itemconfigure(self._body_win, width=event.width)
        self._sync_scroll()
    def _on_mousewheel(self, event):
        # Text/Listbox/Scale/Combobox 自带滚轮行为，不抢占
        try:
            cls = event.widget.winfo_class()
        except Exception:
            cls = ""
        if cls in ("Text", "Listbox", "Scale", "Combobox", "TCombobox", "Spinbox"):
            return
        if self.canvas.winfo_height() >= self.body.winfo_height():
            return  # 内容未超出视口
        self.canvas.yview_scroll(-1 * (event.delta // 120), "units")
    def _title_label(self, parent, text):
        return tk.Label(parent, text=text, bg=parent["bg"], fg=self.accent,
                        font=("Microsoft YaHei UI", 10, "bold"), anchor="w")
    def _tip(self, widget, text):
        """给控件挂悬停说明"""
        ToolTip(widget, text)
    def _panel(self):
        return tk.Frame(self.body, bg=self.panel, highlightthickness=1,
                        highlightbackground="#2a2f3a")
    def _build_ui(self):
        pad = {"padx": 14, "pady": 3}
        head = tk.Frame(self.body, bg=self.bg)
        head.pack(fill="x", padx=14, pady=(10, 4))
        tk.Label(head, text="⊙ CCTools", bg=self.bg, fg=self.accent,
                 font=("Microsoft YaHei UI", 15, "bold")).pack(side="left")
        # —— 模型与检测参数 ——
        p1 = self._panel()
        p1.pack(fill="x", padx=14, pady=4)
        self.lbl_model = tk.Label(p1, text="模型：加载中...", bg=self.panel, fg=self.dim,
                                  font=("Microsoft YaHei UI", 9), anchor="w")
        self.lbl_model.grid(row=0, column=0, columnspan=2, sticky="we", **pad)
        self.btn_model = tk.Button(p1, text="⟳ 选择模型", bg="#2a2f3a", fg=self.fg,
                                   activebackground="#3a4150",
                  activeforeground=self.fg, relief="flat", bd=0, cursor="hand2", width=10,
                  font=("Microsoft YaHei UI", 9), command=self._choose_model)
        self.btn_model.grid(
            row=0, column=2, sticky="e", **pad)
        conf_lbl = tk.Label(p1, text="置信度阈值", bg=self.panel, fg=self.fg,
                            font=("Microsoft YaHei UI", 10))
        conf_lbl.grid(row=1, column=0, sticky="w", **pad)
        self._tip(conf_lbl, TIP_CONF)
        self.conf_var = tk.DoubleVar(value=self.engine.cfg["conf"])
        sc = tk.Scale(p1, variable=self.conf_var, from_=0.1, to=0.9, resolution=0.05,
                      orient="horizontal", length=300, bg=self.panel, fg=self.fg,
                      highlightthickness=0, troughcolor="#2a2f3a", bd=0, showvalue=True)
        sc.grid(row=1, column=1, columnspan=2, sticky="w", **pad)
        self._tip(sc, TIP_CONF)
        sc.config(command=lambda v: self.engine.update_cfg("conf", float(v)))
        cap_lbl = tk.Label(p1, text="捕获区域", bg=self.panel, fg=self.fg,
                           font=("Microsoft YaHei UI", 10))
        cap_lbl.grid(row=2, column=0, sticky="w", **pad)
        self._tip(cap_lbl, TIP_CAP)
        size0 = int(self.engine.cfg.get("capture_size", 448))
        self.cap_text = tk.StringVar(value=f"屏幕中心 {size0}×{size0}")
        self.lbl_cap = tk.Label(p1, textvariable=self.cap_text,
                                bg=self.panel, fg=self.green,
                                font=("Microsoft YaHei UI", 10, "bold"))
        self.lbl_cap.grid(row=2, column=1, columnspan=2, sticky="w", **pad)
        self._tip(self.lbl_cap, TIP_CAP)
        strat_lbl = tk.Label(p1, text="目标策略", bg=self.panel, fg=self.fg,
                             font=("Microsoft YaHei UI", 10))
        strat_lbl.grid(row=3, column=0, sticky="w", **pad)
        self._tip(strat_lbl, TIP_STRATEGY)
        sf = tk.Frame(p1, bg=self.panel)
        sf.grid(row=3, column=1, columnspan=2, sticky="w", **pad)
        self.strategy_var = tk.StringVar(value=self.engine.cfg["strategy"])
        cb = ttk.Combobox(sf, textvariable=self.strategy_var, values=["最近中心", "最高置信度"],
                          state="readonly", width=12, font=("Microsoft YaHei UI", 10))
        cb.pack(side="left")
        cb.bind("<<ComboboxSelected>>", lambda e: self.engine.update_cfg("strategy", self.strategy_var.get()))
        anti_lbl = tk.Label(p1, text="防截屏捕获", bg=self.panel, fg=self.fg,
                            font=("Microsoft YaHei UI", 10))
        anti_lbl.grid(row=4, column=0, sticky="w", **pad)
        self._tip(anti_lbl, TIP_ANTI)
        self.anti_var = tk.BooleanVar(value=bool(self.engine.cfg.get("anti_capture", True)))
        tk.Checkbutton(p1, text="防止微信/QQ/OBS等截图录屏（本窗口对捕获软件不可见）",
                       variable=self.anti_var, bg=self.panel, fg=self.dim,
                       selectcolor="#22262e", activebackground=self.panel,
                       activeforeground=self.fg, highlightthickness=0, bd=0,
                       font=("Microsoft YaHei UI", 9),
                       command=self._anti_capture_changed).grid(row=4, column=1, columnspan=2,
                                                                sticky="w", **pad)
        # —— 捕获方式（独立一行，避免右侧被裁切）——
        capm_lbl = tk.Label(p1, text="捕获方式", bg=self.panel, fg=self.fg,
                            font=("Microsoft YaHei UI", 10))
        capm_lbl.grid(row=5, column=0, sticky="w", **pad)
        self._tip(capm_lbl, "捕获方式：WGC=Windows Graphics Capture（推荐）；\n"
                     "DXGI=桌面复制引擎。帧纹理均GPU直通进管线，无CPU拷贝。")
        mf = tk.Frame(p1, bg=self.panel)
        mf.grid(row=5, column=1, columnspan=2, sticky="w", **pad)
        self.capm_names = ["WGC (Windows Graphics Capture)", "DXGI桌面复制"]
        self.capm_keys = ["wgc", "dxgi"]
        cur_m = str(self.engine.cfg.get("capture_method", "wgc"))
        cur_m = cur_m if cur_m in self.capm_keys else "wgc"
        self.capm_var = tk.StringVar(value=self.capm_names[self.capm_keys.index(cur_m)])
        self.capm_cb = ttk.Combobox(mf, textvariable=self.capm_var, values=self.capm_names,
                               state="readonly", width=30, font=("Microsoft YaHei UI", 10))
        self.capm_cb.pack(side="left")
        self.capm_cb.bind("<<ComboboxSelected>>", self._capture_method_changed)
        # —— 捕获微调 ——
        fine_lbl = tk.Label(p1, text="捕获微调", bg=self.panel, fg=self.fg,
                            font=("Microsoft YaHei UI", 10))
        fine_lbl.grid(row=6, column=0, sticky="w", **pad)
        self._tip(fine_lbl, TIP_CAP + "\n" + TIP_MON)
        cf = tk.Frame(p1, bg=self.panel)
        cf.grid(row=6, column=1, columnspan=2, sticky="w", **pad)
        size_lbl = tk.Label(cf, text="区域大小", bg=self.panel, fg=self.dim,
                            font=("Microsoft YaHei UI", 9))
        size_lbl.pack(side="left", padx=(0, 4))
        self._tip(size_lbl, TIP_CAP)
        self.capsize_var = tk.StringVar(value=f"{int(self.engine.cfg.get('capture_size', 640))}")
        self.capsize_lbl = tk.Label(cf, textvariable=self.capsize_var, bg=self.panel,
                                    fg=self.green, font=("Microsoft YaHei UI", 9))
        self.capsize_lbl.pack(side="left")
        self._tip(self.capsize_lbl, TIP_CAP)
        # 显示器列表用Engine在TRT加载前缓存的快照——模型加载期间并发调用
        # EnumDisplayMonitors会触发0xC0000005原生崩溃，禁止在此枚举
        mons = getattr(self.engine, "monitors", None)
        if not mons:
            mons = list_monitors()
        self.mon_devs = [None] + [m[0] for m in mons]
        self.mon_names = ["主显示器"] + [f"显示器{i+1}" for i in range(len(self.mon_devs) - 1)]
        cur = self.engine.cfg.get("monitor")
        cur_idx = self.mon_devs.index(cur) if cur in self.mon_devs else 0
        self.mon_var = tk.StringVar(value=self.mon_names[cur_idx])
        self.mon_cb = ttk.Combobox(cf, textvariable=self.mon_var, values=self.mon_names,
                              state="readonly", width=10, font=("Microsoft YaHei UI", 9))
        self.mon_cb.pack(side="left")
        self._tip(self.mon_cb, TIP_MON)
        self.mon_cb.bind("<<ComboboxSelected>>", self._monitor_changed)
        off_lbl = tk.Label(cf, text="准星偏移", bg=self.panel, fg=self.dim,
                           font=("Microsoft YaHei UI", 9))
        off_lbl.pack(side="left", padx=(14, 0))
        self._tip(off_lbl, TIP_OFFSET)
        self.dx_var = tk.IntVar(value=int(self.engine.cfg.get("crosshair_dx", 0)))
        self.dy_var = tk.IntVar(value=int(self.engine.cfg.get("crosshair_dy", 0)))
        for lab, var, key in (("X", self.dx_var, "crosshair_dx"), ("Y", self.dy_var, "crosshair_dy")):
            tk.Label(cf, text=lab, bg=self.panel, fg=self.dim,
                     font=("Microsoft YaHei UI", 9)).pack(side="left", padx=(10, 2))
            sp = tk.Spinbox(cf, from_=-300, to=300, width=5, textvariable=var,
                            bg="#22262e", fg=self.fg, buttonbackground="#2a2f3a",
                            relief="flat", insertbackground=self.fg,
                            font=("Microsoft YaHei UI", 9))
            sp.pack(side="left")
            var.trace_add("write", lambda *a, k=key, v=var: self._spin_changed(k, v))
        # —— 识别类别 ——
        p2 = self._panel()
        p2.pack(fill="x", padx=14, pady=4)
        self._title_label(p2, "── 识别类别（勾选后只检测所选类别）").grid(
            row=0, column=0, columnspan=2, sticky="w", **pad)
        btns = tk.Frame(p2, bg=self.panel)
        btns.grid(row=0, column=2, sticky="e", **pad)
        tk.Button(btns, text="全选", bg="#2a2f3a", fg=self.fg, activebackground="#3a4150",
                  activeforeground=self.fg, relief="flat", bd=0, width=6, cursor="hand2",
                  font=("Microsoft YaHei UI", 9),
                  command=lambda: self._set_all_classes(True)).pack(side="left", padx=2)
        tk.Button(btns, text="全不选", bg="#2a2f3a", fg=self.fg, activebackground="#3a4150",
                  activeforeground=self.fg, relief="flat", bd=0, width=6, cursor="hand2",
                  font=("Microsoft YaHei UI", 9),
                  command=lambda: self._set_all_classes(False)).pack(side="left", padx=2)
        self.class_canvas = tk.Canvas(p2, bg=self.panel, highlightthickness=0, height=66)
        self.class_canvas.grid(row=1, column=0, columnspan=3, sticky="we", padx=10, pady=(0, 8))
        # —— 鼠标设置 ——
        p3 = self._panel()
        p3.pack(fill="x", padx=14, pady=4)
        self._title_label(p3, "── 鼠标设置").grid(row=0, column=0, columnspan=2, sticky="w", **pad)
        be_lbl = tk.Label(p3, text="控制后端", bg=self.panel, fg=self.fg,
                          font=("Microsoft YaHei UI", 10))
        be_lbl.grid(row=1, column=0, sticky="w", **pad)
        self._tip(be_lbl, TIP_BACKEND)
        bf = tk.Frame(p3, bg=self.panel)
        bf.grid(row=1, column=1, sticky="w", **pad)
        self.backend_var = tk.StringVar(value="罗技驱动DLL" if self.engine.cfg["backend"] == "logitech" else "SendInput")
        for text, mode in (("罗技驱动DLL", "logitech"), ("SendInput", "sendinput")):
            tk.Radiobutton(bf, text=text, variable=self.backend_var, value=text, bg=self.panel,
                           fg=self.fg, selectcolor="#22262e", activebackground=self.panel,
                           activeforeground=self.fg, highlightthickness=0, bd=0,
                           font=("Microsoft YaHei UI", 10),
                           command=self._backend_changed).pack(side="left", padx=(0, 12))
        self.invert_var = tk.BooleanVar(value=self.engine.cfg["invert_y"])
        inv_cb = tk.Checkbutton(bf, text="Y轴反向(方向相反时勾选)", variable=self.invert_var,
                                bg=self.panel, fg=self.dim, selectcolor="#22262e",
                                activebackground=self.panel, activeforeground=self.fg,
                                highlightthickness=0, bd=0, font=("Microsoft YaHei UI", 9),
                                command=lambda: self.engine.update_cfg("invert_y", self.invert_var.get()))
        inv_cb.pack(side="left", padx=(8, 0))
        self._tip(inv_cb, TIP_INVERT)
        # —— 热键与控制参数 ——
        p4 = self._panel()
        p4.pack(fill="x", padx=14, pady=4)
        self._title_label(p4, "── 热键与控制参数").grid(row=0, column=0, columnspan=4, sticky="w", **pad)
        hk_lbl = tk.Label(p4, text="触发热键", bg=self.panel, fg=self.fg,
                          font=("Microsoft YaHei UI", 10))
        hk_lbl.grid(row=1, column=0, sticky="w", **pad)
        self._tip(hk_lbl, TIP_HOTKEY)
        hf = tk.Frame(p4, bg=self.panel)
        hf.grid(row=1, column=1, columnspan=3, sticky="w", **pad)
        self.hotkey_var = tk.StringVar(value=self.engine.cfg.get("hotkey_name", "鼠标侧键X2"))
        hk_cb = ttk.Combobox(hf, textvariable=self.hotkey_var, values=HOTKEY_NAMES,
                             state="readonly", width=14, font=("Microsoft YaHei UI", 10))
        hk_cb.pack(side="left")
        hk_cb.bind("<<ComboboxSelected>>", self._hotkey_changed)
        self.vk_entry = tk.Entry(hf, width=6, bg="#22262e", fg=self.fg,
                                 insertbackground=self.fg, relief="flat",
                                 font=("Microsoft YaHei UI", 9))
        self.vk_entry.insert(0, str(int(self.engine.cfg.get("vk", 0x06))))
        self.vk_entry.pack(side="left", padx=(8, 0))
        self.vk_entry.bind("<KeyRelease>", self._vk_typed)
        tk.Label(hf, text="长按触发 · 选'自定义'后输入VK码(十进制)", bg=self.panel, fg=self.dim,
                 font=("Microsoft YaHei UI", 9)).pack(side="left", padx=(12, 0))
        pk_lbl = tk.Label(p4, text="紧急停止", bg=self.panel, fg=self.fg,
                          font=("Microsoft YaHei UI", 10))
        pk_lbl.grid(row=2, column=0, sticky="w", **pad)
        self._tip(pk_lbl, TIP_PANIC)
        pf = tk.Frame(p4, bg=self.panel)
        pf.grid(row=2, column=1, columnspan=3, sticky="w", **pad)
        self.panic_var = tk.StringVar(value=self.engine.cfg.get("panic_name", "End"))
        pk_cb = ttk.Combobox(pf, textvariable=self.panic_var, values=list(cfg.HOTKEY_VK.keys()),
                             state="readonly", width=14, font=("Microsoft YaHei UI", 10))
        pk_cb.pack(side="left")
        pk_cb.bind("<<ComboboxSelected>>", self._panic_changed)
        tk.Label(pf, text="推理中按下立即停止全部控制", bg=self.panel, fg=self.dim,
                 font=("Microsoft YaHei UI", 9)).pack(side="left", padx=(12, 0))
        ae_lbl = tk.Label(p4, text="跟随辅助", bg=self.panel, fg=self.fg,
                          font=("Microsoft YaHei UI", 10))
        ae_lbl.grid(row=3, column=0, sticky="w", **pad)
        self._tip(ae_lbl, TIP_AEN)
        af = tk.Frame(p4, bg=self.panel)
        af.grid(row=3, column=1, columnspan=3, sticky="w", **pad)
        self.aen_var = tk.StringVar(value=self.engine.cfg.get("aim_enable_name", "F8"))
        ae_cb = ttk.Combobox(af, textvariable=self.aen_var,
                             values=list(cfg.HOTKEY_VK.keys()) + ["自定义"],
                             state="readonly", width=14, font=("Microsoft YaHei UI", 10))
        ae_cb.pack(side="left")
        ae_cb.bind("<<ComboboxSelected>>", self._aim_enable_changed)
        self.aevk_entry = tk.Entry(af, width=6, bg="#22262e", fg=self.fg,
                                   insertbackground=self.fg, relief="flat",
                                   font=("Microsoft YaHei UI", 9))
        self.aevk_entry.insert(0, str(int(self.engine.cfg.get("aim_enable_vk", 0x77))))
        self.aevk_entry.pack(side="left", padx=(8, 0))
        self.aevk_entry.bind("<KeyRelease>", self._aevk_typed)
        tk.Label(af, text="按下切换跟随辅助开/关（信息显示在日志与状态栏）", bg=self.panel, fg=self.dim,
                 font=("Microsoft YaHei UI", 9)).pack(side="left", padx=(12, 0))
        win_lbl = tk.Label(p4, text="生效窗口", bg=self.panel, fg=self.fg,
                           font=("Microsoft YaHei UI", 10))
        win_lbl.grid(row=4, column=0, sticky="w", **pad)
        self._tip(win_lbl, TIP_WIN)
        wf = tk.Frame(p4, bg=self.panel)
        wf.grid(row=4, column=1, columnspan=3, sticky="w", **pad)
        self.win_var = tk.StringVar(value=self.engine.cfg.get("win_filter", "") or "")
        we = tk.Entry(wf, textvariable=self.win_var, width=22, bg="#22262e", fg=self.fg,
                      insertbackground=self.fg, relief="flat", font=("Microsoft YaHei UI", 9))
        we.pack(side="left")
        self.win_var.trace_add("write", self._win_changed)
        tk.Label(wf, text="仅游戏窗口生效（标题包含关键字），留空=全局", bg=self.panel, fg=self.dim,
                 font=("Microsoft YaHei UI", 9)).pack(side="left", padx=(10, 0))
        params = (
            ("移动比例", "aim_ratio", 0.1, 1.0, 0.05, 0), ("跟随点Y偏移", "aim_offset_y", -0.5, 0.5, 0.05, 1),
            ("跟随范围(px)", "fov_radius", 0.0, 160.0, 5.0, 0), ("跟随判定(px)", "lock_radius", 5.0, 40.0, 1.0, 1),
            ("NMS IoU", "nms_iou", 0.3, 0.7, 0.05, 0), ("收尾死区(px)", "deadzone", 1.0, 6.0, 0.5, 1),
            ("联动系数", "cam_g", 1.0, 6.0, 0.1, 0), ("检测帧率上限", "detect_fps", 60.0, 240.0, 10.0, 1),
            ("延迟补偿(ms)", "capture_delay_ms", 0.0, 80.0, 5.0, 0),
        )
        self._param_vars = {}
        for i, (text, key, frm, to, res, col) in enumerate(params):
            self._param_scale(p4, 5 + i // 2, text, key, frm, to, res, col,
                              PARAM_TIPS.get(key, ""))
        # —— 启动与状态 ——
        p5 = self._panel()
        p5.pack(fill="x", padx=14, pady=4)
        btnf = tk.Frame(p5, bg=self.panel)
        btnf.pack(fill="x", padx=14, pady=10)
        self.btn_start = tk.Button(btnf, text="▶ 开始推理", bg="#0078d4", fg="white",
                                   activebackground="#1084e8", activeforeground="white",
                                   relief="flat", width=14, bd=0, cursor="hand2",
                                   font=("Microsoft YaHei UI", 11, "bold"), command=self._toggle_run)
        self.btn_start.pack(side="left", padx=(0, 10))
        self.btn_preview = tk.Button(btnf, text="⛶ 开启预览", bg="#2a2f3a", fg=self.fg,
                                     activebackground="#3a4150", activeforeground=self.fg,
                                     relief="flat", width=14, bd=0, cursor="hand2",
                                     font=("Microsoft YaHei UI", 11, "bold"), command=self._toggle_preview)
        self.btn_preview.pack(side="left")
        self.btn_reset = tk.Button(btnf, text="⟲ 重置参数", bg="#2a2f3a", fg=self.fg,
                                   activebackground="#3a4150", activeforeground=self.fg,
                                   relief="flat", width=14, bd=0, cursor="hand2",
                                   font=("Microsoft YaHei UI", 11, "bold"),
                                   command=self._reset_params)
        self.btn_reset.pack(side="left", padx=(10, 0))
        self._tip(self.btn_reset, TIP_RESET)
        stat = tk.Frame(p5, bg=self.panel)
        stat.pack(fill="x", padx=14, pady=(0, 10))
        self.status_vars = {}
        for i, name in enumerate(("状态", "设备", "帧率", "耗时", "检测", "捕获", "跟随", "持续", "运行")):
            box = tk.Frame(stat, bg=self.panel)
            box.grid(row=0, column=i, padx=(0, 10), sticky="w")
            tk.Label(box, text=name, bg=self.panel, fg=self.dim,
                     font=("Microsoft YaHei UI", 9)).pack(anchor="w")
            var = tk.StringVar(value="-")
            tk.Label(box, textvariable=var, bg=self.panel, fg=self.green,
                     font=("Microsoft YaHei UI", 10, "bold")).pack(anchor="w")
            self.status_vars[name] = var
        self.log_text = tk.Text(self.body, height=4, bg="#101216", fg="#9adfbf",
                                insertbackground=self.fg, relief="flat", bd=0,
                                font=("Microsoft YaHei UI", 9), state="disabled")
        self.log_text.pack(fill="both", expand=True, padx=14, pady=(4, 10))
        self.log_text.tag_configure("dim", foreground="#5c6470")
        # 日志可复制：Ctrl+C 复制（无选中时复制全部），右键菜单含复制/清空
        self.log_menu = tk.Menu(self.log_text, tearoff=0)
        self.log_menu.add_command(label="复制选中", command=self._copy_log_sel)
        self.log_menu.add_command(label="复制全部", command=self._copy_log_all)
        self.log_menu.add_separator()
        self.log_menu.add_command(label="清空日志", command=self._clear_log)
        self.log_text.bind("<Button-1>", lambda e: self.log_text.focus_set())
        self.log_text.bind("<Control-c>", self._copy_log_sel)
        self.log_text.bind("<Control-C>", self._copy_log_sel)
        self.log_text.bind("<Button-3>", lambda e: self.log_menu.tk_popup(e.x_root, e.y_root))
        self._log("[提示] 日志可鼠标选中后 Ctrl+C 复制；右键菜单支持复制全部/清空")
        self._log("[启动] 系统就绪，模型加载中...")
    def _param_scale(self, parent, row, text, key, frm, to, res, col, tip=""):
        pad = {"padx": 14, "pady": 2}
        lbl = tk.Label(parent, text=text, bg=parent["bg"], fg=self.fg,
                       font=("Microsoft YaHei UI", 9))
        lbl.grid(row=row, column=col * 2, sticky="w", **pad)
        var = tk.DoubleVar(value=float(self.engine.cfg.get(key, frm)))
        s = tk.Scale(parent, variable=var, from_=frm, to=to, resolution=res,
                     orient="horizontal", length=150, bg=parent["bg"], fg=self.fg,
                     highlightthickness=0, troughcolor="#2a2f3a", bd=0,
                     showvalue=True, font=("Microsoft YaHei UI", 8))
        s.grid(row=row, column=col * 2 + 1, sticky="w", padx=(0, 6))
        s.config(command=lambda v, k=key: self.engine.update_cfg(k, float(v)))
        if tip:
            self._tip(lbl, tip)
            self._tip(s, tip)
        self._param_vars[key] = var
    def _copy_log_sel(self, event=None):
        """Ctrl+C：复制选中文本，未选中则复制全部日志"""
        try:
            text = self.log_text.get("sel.first", "sel.last")
        except tk.TclError:
            text = self.log_text.get("1.0", "end-1c")
        if text:
            self.root.clipboard_clear()
            self.root.clipboard_append(text)
        return "break"
    def _copy_log_all(self):
        text = self.log_text.get("1.0", "end-1c")
        self.root.clipboard_clear()
        self.root.clipboard_append(text)
        self._log("[日志] 全部日志已复制到剪贴板")
    def _clear_log(self):
        self.log_text.configure(state="normal")
        self.log_text.delete("1.0", "end")
        self.log_text.configure(state="disabled")
    def _log(self, msg):
        self._logq.put(str(msg))
        try:
            os.makedirs(cfg.LOG_DIR, exist_ok=True)
            path = os.path.join(cfg.LOG_DIR, time.strftime("cctools_%Y%m%d.log"))
            try:
                # 简单轮转：单日日志超5MB改名留档，防止无限增长
                if os.path.isfile(path) and os.path.getsize(path) > 5 * 1024 * 1024:
                    os.replace(path, path + ".old")
            except Exception:
                pass
            with open(path, "a", encoding="utf-8") as f:
                f.write(time.strftime("[%H:%M:%S] ") + str(msg) + "\n")
        except Exception:
            pass
    def _append_log(self, msg):
        self.log_text.configure(state="normal")
        self.log_text.insert("end", msg + "\n")
        self.log_text.see("end")
        self.log_text.configure(state="disabled")
        print(msg, flush=True)
    @staticmethod
    def _fmt_dur(t):
        t = int(t or 0)
        return f"{t // 60}分{t % 60}秒" if t >= 60 else f"{t}秒"
    def _int(self, var, default):
        try:
            return int(var.get())
        except Exception:
            return default
    def _build_class_panel(self):
        names = self.engine.model_info.get("names", {})
        if not names:
            return
        self.class_built = True
        family = self.engine.model_info.get("family", "未知")
        self.lbl_model.configure(text=f"模型：{family}，共{len(names)}类", fg=self.green)
        self._log(f"[类别] 模型版本：{family}")
        # 检测区域自动匹配模型输入尺寸（模型按该尺度训练）
        size = int(getattr(self.engine.detector, "input_size", 0) or 640)
        self.capsize_var.set(str(size))
        self.cap_text.set(f"屏幕中心 {size}×{size}")
        inner = tk.Frame(self.class_canvas, bg=self.panel)
        self.class_canvas.create_window((0, 0), window=inner, anchor="nw")
        ids = sorted(names.keys())
        saved = self.engine.get_selected_classes()   # 恢复上次保存的类别筛选（None=全选）
        for i, cid in enumerate(ids):
            var = tk.BooleanVar(value=(True if saved is None else cid in saved))
            var.trace_add("write", lambda *a: self._sync_classes())
            cbx = tk.Checkbutton(inner, text=f"{cid}. {names[cid]}", variable=var,
                                 bg=self.panel, fg=self.fg, selectcolor="#22262e",
                                 activebackground=self.panel, activeforeground=self.fg,
                                 highlightthickness=0, bd=0, font=("Microsoft YaHei UI", 10))
            cbx.grid(row=i // 4, column=i % 4, padx=10, pady=2, sticky="w")
            self.class_widgets[cid] = var
        inner.update_idletasks()
        self.class_canvas.configure(scrollregion=self.class_canvas.bbox("all"))
        self.class_canvas.bind("<MouseWheel>", lambda e: self.class_canvas.yview_scroll(-1 * (e.delta // 120), "units"))
        self._sync_classes()
    def _sync_classes(self):
        sel = [cid for cid, var in self.class_widgets.items() if var.get()]
        self.engine.set_classes(sel)
    def _choose_model(self):
        if self.engine.running:
            self._log("[模型] 请先停止推理再切换模型")
            return
        init_dir = os.path.dirname(self.engine.cfg.get("model_path") or "") or None
        path = filedialog.askopenfilename(title="选择YOLO ONNX模型", initialdir=init_dir,
                                          filetypes=[("ONNX模型", "*.onnx"), ("全部文件", "*.*")])
        if not path:
            return
        self.class_built = False
        self.class_widgets = {}
        self.class_canvas.delete("all")
        self.lbl_model.configure(text=f"模型：加载中 {os.path.basename(path)}...", fg=self.dim)
        self._log(f"[模型] 正在加载：{path}")
        self.engine.load_model(path)
    def _set_all_classes(self, on):
        for var in self.class_widgets.values():
            var.set(on)
    def _apply_root_guard(self):
        """启动时按配置应用主窗口防截屏"""
        try:
            hwnd = ctypes.windll.user32.GetParent(self.root.winfo_id())
            mode = capture_guard.apply(hwnd, self.anti_var.get())
            if mode == "排除捕获":
                self._log("[防护] 防截屏已开启：主窗口对微信/QQ/OBS等截图录屏不可见")
            elif mode == "捕获时黑屏":
                self._log("[防护] 系统版本较旧，已降级为“捕获时黑屏”防护模式")
        except Exception:
            pass
    def _anti_capture_changed(self):
        on = self.anti_var.get()
        self.engine.update_cfg("anti_capture", on)
        try:
            hwnd = ctypes.windll.user32.GetParent(self.root.winfo_id())
            mode = capture_guard.apply(hwnd, on)
            ph = capture_guard.find_window("实时检测预览")
            if ph:
                capture_guard.apply(ph, on)
            if on:
                self._log(f"[防护] 防截屏已开启：{mode}" if mode
                          else "[防护] 防截屏开启失败（系统可能不支持）")
            else:
                self._log("[防护] 防截屏已关闭，截图/录屏可正常捕获本窗口")
        except Exception as e:
            self._log(f"[防护] 防截屏设置异常：{e}")
    def _monitor_changed(self, event=None):
        idx = self.mon_names.index(self.mon_var.get())
        self.engine.update_cfg("monitor", self.mon_devs[idx])
        self._log(f"[捕获] 显示器已切换：{self.mon_var.get()}（下次推理生效）")

    def _capture_method_changed(self, event=None):
        idx = self.capm_names.index(self.capm_var.get())
        self.engine.update_cfg("capture_method", self.capm_keys[idx])
        self._log(f"[捕获] 捕获方式已切换：{self.capm_var.get()}（下次推理生效）")
    def _spin_changed(self, key, var):
        try:
            val = int(var.get())
        except Exception:
            return
        self.engine.update_cfg(key, val)
    def _hotkey_changed(self, event=None):
        name = self.hotkey_var.get()
        self.engine.update_cfg("hotkey_name", name)
        if name == "自定义":
            vk = self._int(self.vk_entry, 0x06) & 0xFF
        else:
            vk = cfg.HOTKEY_VK.get(name, 0x06)
            self.vk_entry.delete(0, "end")
            self.vk_entry.insert(0, str(vk))
        self.engine.update_cfg("vk", vk)
        self._log(f"[热键] 触发键已设置为：{name} (VK {vk})")
    def _vk_typed(self, event=None):
        if self.hotkey_var.get() != "自定义":
            return
        vk = self._int(self.vk_entry, 0x06) & 0xFF
        self.engine.update_cfg("vk", vk)
    def _panic_changed(self, event=None):
        name = self.panic_var.get()
        vk = cfg.HOTKEY_VK.get(name, 0x23)
        self.engine.update_cfg("panic_name", name)
        self.engine.update_cfg("panic_vk", vk)
        self._log(f"[热键] 紧急停止键已设置为：{name}")
    def _aim_enable_changed(self, event=None):
        name = self.aen_var.get()
        self.engine.update_cfg("aim_enable_name", name)
        if name == "自定义":
            vk = self._int(self.aevk_entry, 0x77) & 0xFF
        else:
            vk = cfg.HOTKEY_VK.get(name, 0x77)
            self.aevk_entry.delete(0, "end")
            self.aevk_entry.insert(0, str(vk))
        self.engine.update_cfg("aim_enable_vk", vk)
        self._log(f"[开关] 跟随辅助键已设置为：{name}" + ("" if vk else "（未启用）"))
    def _aevk_typed(self, event=None):
        if self.aen_var.get() != "自定义":
            return
        vk = self._int(self.aevk_entry, 0x77) & 0xFF
        self.engine.update_cfg("aim_enable_vk", vk)
    def _win_changed(self, *a):
        self.engine.update_cfg("win_filter", self.win_var.get())
    def _backend_changed(self):
        mode = "logitech" if self.backend_var.get() == "罗技驱动DLL" else "sendinput"
        self.engine.update_cfg("backend", mode)
        self.engine.backend = None
        if self.engine.running:
            try:
                self.engine._init_backend()
            except Exception as e:
                self._log(f"[鼠标] 后端切换失败：{e}")
    def _reset_params(self):
        """一键重置：恢复通用无过冲稳定参数（不动热键/显示器/后端/类别等个人设置）"""
        if not messagebox.askyesno(
                "重置参数",
                "将检测/跟随调参重置为通用稳定值：\n"
                "任何设备、任何刷新率下都不会出现过冲与摇晃。\n\n"
                "不会改动：热键、显示器、后端、防截屏、准星偏移、\n"
                "捕获区域、已选类别。\n\n继续？"):
            return
        for key, val in cfg.RESET_PARAMS.items():
            self.engine.update_cfg(key, val)
        # 同步UI控件显示（捕获区域不属于重置范围——它与模型训练尺度绑定）
        self.conf_var.set(float(cfg.RESET_PARAMS["conf"]))
        self.strategy_var.set(cfg.RESET_PARAMS["strategy"])
        for key, val in cfg.RESET_PARAMS.items():
            if key in self._param_vars and isinstance(val, (int, float)) \
                    and not isinstance(val, bool):
                self._param_vars[key].set(float(val))
        self._log("[参数] 已重置为通用稳定参数（无过冲）：conf=0.5 ratio=0.45 "
                  "deadzone=2.0 cam_g=2.5 FOV=0 延迟补偿=自动（捕获区域保持不变）")
    def _toggle_run(self):
        if not self.engine.running:
            self.btn_start.configure(text="■ 停止推理", bg="#c0392b", activebackground="#d64535")
            self.engine.start()
        else:
            self.btn_start.configure(text="▶ 开始推理", bg="#0078d4", activebackground="#1084e8")
            self.engine.stop()
    def _toggle_preview(self):
        self.previewing = not self.previewing
        if self.previewing:
            self.btn_preview.configure(text="⛶ 关闭预览", bg="#b9770e", activebackground="#cb8a1a")
            if not self.engine.running:
                self._log("[预览] 提示：预览需在开始推理后才有画面")
            self.engine.toggle_preview(True)
        else:
            self.btn_preview.configure(text="⛶ 开启预览", bg="#2a2f3a", activebackground="#3a4150")
            self.engine.toggle_preview(False)
    def _sync_locks(self):
        """推理运行中锁定：选择模型/显示器/捕获方式——这三项涉及捕获与
        TRT会话生命周期，运行中改动有竞态风险，须停止推理后才能操作"""
        run = bool(getattr(self.engine, "running", False))
        for w in (self.btn_model, self.capm_cb, self.mon_cb):
            want = "disabled" if run else ("normal" if w is self.btn_model else "readonly")
            if str(w.cget("state")) != want:
                w.configure(state=want)
    def _poll_status(self):
        try:
            # 后台线程日志入队显示
            while True:
                try:
                    msg = self._logq.get_nowait()
                except queue.Empty:
                    break
                self._append_log(msg)
            # 紧急停止/引擎异常退出时同步按钮状态
            if not self.engine.running and self.btn_start.cget("text") == "■ 停止推理":
                self.btn_start.configure(text="▶ 开始推理", bg="#0078d4", activebackground="#1084e8")
                self._log("[引擎] 推理已停止")
            self._sync_locks()
            # 模型未就绪时置灰开始按钮（start改为立即返回，不再阻塞界面等待）
            if not self.engine.running:
                if not self.engine.detector_ready:
                    if str(self.btn_start.cget("state")) != "disabled":
                        self.btn_start.configure(state="disabled", text="模型加载中...")
                elif str(self.btn_start.cget("state")) == "disabled":
                    self.btn_start.configure(state="normal", text="▶ 开始推理",
                                             bg="#0078d4", activebackground="#1084e8")
            if not self.class_built and self.engine.detector_ready:
                if self.engine.model_error:
                    self.lbl_model.configure(text=f"模型加载失败：{self.engine.model_error}", fg="#e74c3c")
                else:
                    self._build_class_panel()
            s = self.engine.get_status()
            p = s["provider"]
            dev = ("GPU (TensorRT FP16)" if "Tensorrt" in p else
                   "GPU (CUDA)" if "CUDA" in p else "-" if p == "-" else p)
            self.status_vars["状态"].set(s["state"] + (" ✓已跟随" if s["locked"] else ""))
            self.status_vars["设备"].set(dev)
            self.status_vars["帧率"].set(
                f"{s['fps']:.0f} FPS (上限{s['refresh_hz']}Hz)" if s["fps"] > 0 else "-")
            self.status_vars["耗时"].set(f"{s['infer_ms']:.1f} ms" if s["infer_ms"] > 0 else "-")
            self.status_vars["检测"].set(str(s["num_dets"]) if self.engine.running else "-")
            self.status_vars["捕获"].set(s["capture_mode"])
            if self.engine.running:
                self.status_vars["跟随"].set("开启" if s.get("aim_on") else "已关闭")
            else:
                self.status_vars["跟随"].set("-")
            self.status_vars["持续"].set(self._fmt_dur(s.get("lock_time", 0)) if self.engine.running else "-")
            self.status_vars["运行"].set(self._fmt_dur(s.get("uptime", 0)) if self.engine.running else "-")
        finally:
            self.root.after(200, self._poll_status)
    def on_close(self):
        self.engine.toggle_preview(False)
        self.engine.stop()
        self.engine.flush_settings()
        if self.engine.pipeline:
            self.engine.pipeline.close()
        self.engine._exit.set()
        self.root.destroy()
