# -*- coding: utf-8 -*-
"""YOLO ONNX 推理模块：自动识别模型版本(v5/v8/v11/v12/v10/v26)，读取训练类别，GPU加速"""
import os
import sys
import glob
import ast
import ctypes
import time
import numpy as np
import cv2
import onnxruntime as ort
from core.config import INPUT_SIZE
def preload_cuda_dlls():
    """显式预加载NVIDIA运行库DLL，确保CUDA/TensorRT Provider可用"""
    bases = [os.path.join(os.path.dirname(sys.executable), "Lib", "site-packages")]
    if getattr(sys, "frozen", False) and getattr(sys, "_MEIPASS", None):
        bases.insert(0, sys._MEIPASS)  # PyInstaller打包：DLL解压在_internal下
    loaded = 0
    dirs = []
    for base in bases:
        dirs += glob.glob(os.path.join(base, "nvidia", "*", "bin"))
        dirs += [os.path.join(base, "tensorrt_libs"), os.path.join(base, "tensorrt")]
    for d in dirs:
        if not os.path.isdir(d):
            continue
        try:
            os.add_dll_directory(d)
            os.environ["PATH"] = d + os.pathsep + os.environ.get("PATH", "")
        except OSError:
            pass
        for dll in glob.glob(os.path.join(d, "*.dll")):
            try:
                ctypes.CDLL(dll)
                loaded += 1
            except OSError:
                pass
    return loaded
def letterbox(bgr, size=INPUT_SIZE):
    h, w = bgr.shape[:2]
    r = min(size / w, size / h)
    nw, nh = int(round(w * r)), int(round(h * r))
    resized = cv2.resize(bgr, (nw, nh), interpolation=cv2.INTER_LINEAR)
    px, py = (size - nw) // 2, (size - nh) // 2
    padded = np.full((size, size, 3), 114, np.uint8)
    padded[py:py + nh, px:px + nw] = resized
    return padded, px, py, r
def parse_names(text):
    """解析Ultralytics元数据中的类别字典"""
    if not text:
        return None
    try:
        d = ast.literal_eval(text)
        return {int(k): str(v) for k, v in d.items()}
    except Exception:
        return None
def _ascii_safe_path(p):
    """路径含中文时TensorRT引擎编译会崩溃(UTF-8解码失败)，复制到纯英文路径使用"""
    try:
        p.encode("ascii")
        return p
    except UnicodeEncodeError:
        pass
    import shutil
    base = os.environ.get("PROGRAMDATA") or os.environ.get("TEMP") or "C:\\Temp"
    safe_dir = os.path.join(base, "YOLO_TRT")
    os.makedirs(safe_dir, exist_ok=True)
    # 文件名带"大小+修改时间"指纹，避免同名/同大小不同内容的模型误用旧副本
    stem, ext = os.path.splitext(os.path.basename(p))
    key = f"{os.path.getsize(p)}_{int(os.path.getmtime(p))}"
    dst = os.path.join(safe_dir, f"{stem}_{key}{ext}")
    if not os.path.isfile(dst):
        shutil.copyfile(p, dst)
    return dst
SAFE_CACHE_DIR = None
def _safe_cache_dir():
    global SAFE_CACHE_DIR
    if SAFE_CACHE_DIR is None:
        base = os.environ.get("PROGRAMDATA") or os.environ.get("TEMP") or "C:\\Temp"
        SAFE_CACHE_DIR = os.path.join(base, "YOLO_TRT", "trt_cache")
        os.makedirs(SAFE_CACHE_DIR, exist_ok=True)
    return SAFE_CACHE_DIR
class Detector:
    """推理器：自动适配各版本输出结构，返回 [x1,y1,x2,y2,置信度,类别id]（区域坐标）
    pipeline传入时：GPU直通零拷贝推理（NVRTC预处理+TRT FP16外部指针IOBinding）"""
    def __init__(self, model_path, conf=0.45, nms_iou=0.45, pipeline=None):
        self.conf = conf
        self.nms_iou = nms_iou
        self.pipeline = pipeline
        self.seg = {"stage": 0.0, "run": 0.0, "copy": 0.0, "dec": 0.0}  # 分段耗时EMA(ms)
        self.names = {}
        self.family = "未知"
        preload_cuda_dlls()
        self.so = ort.SessionOptions()
        self.so.log_severity_level = 3
        cache_dir = _safe_cache_dir()
        model_path = _ascii_safe_path(model_path)
        os.makedirs(cache_dir, exist_ok=True)
        avail = ort.get_available_providers()
        self.sess = None
        did = pipeline.cuda_ordinal if pipeline is not None else 0
        if "TensorrtExecutionProvider" in avail:
            print("[推理] 尝试TensorRT FP16，首次编译引擎可能需要1-3分钟", flush=True)
            trt_opts = {"trt_fp16_enable": "True",
                        "trt_engine_cache_enable": "True",
                        "trt_engine_cache_path": cache_dir,
                        "trt_timing_cache_enable": "True",
                        "trt_timing_cache_path": cache_dir,
                        "device_id": str(did)}
            try:
                self.sess = ort.InferenceSession(model_path, self.so,
                                                 providers=[("TensorrtExecutionProvider", trt_opts),
                                                            ("CUDAExecutionProvider",
                                                             {"do_copy_in_default_stream": "1",
                                                              "device_id": str(did)}),
                                                            "CPUExecutionProvider"])
                if self.sess.get_providers()[0] != "TensorrtExecutionProvider":
                    raise RuntimeError(f"TRT未生效，实际: {self.sess.get_providers()[0]}")
            except Exception as e:
                print(f"[推理] TensorRT初始化失败({e})，回退CUDA", flush=True)
                self.sess = None
        if self.sess is None:
            cuda_opts = {"cudnn_conv_algo_search": "EXHAUSTIVE", "do_copy_in_default_stream": "1",
                         "device_id": str(did)}
            self.sess = ort.InferenceSession(model_path, self.so,
                                             providers=[("CUDAExecutionProvider", cuda_opts)])
        if self.sess.get_providers()[0] == "CPUExecutionProvider":
            raise RuntimeError("GPU初始化失败：本软件强制使用GPU推理，禁止CPU回退。"
                               "请检查NVIDIA显卡驱动与CUDA环境后重启")
        self.provider = self.sess.get_providers()[0]
        self.inp_name = self.sess.get_inputs()[0].name
        # 模型输入尺寸（捕获区域必须与之一致——模型按该尺度训练）
        try:
            shp = self.sess.get_inputs()[0].shape
            h, w = int(shp[-2]), int(shp[-1])
            self.input_size = w if (h == w and w > 0) else INPUT_SIZE
        except Exception:
            self.input_size = INPUT_SIZE
        # 选择首个3维输出作为检测输出，避免多输出模型计算全部输出头
        outs = self.sess.get_outputs()
        self._out_name = outs[0].name
        for o in outs:
            if len(o.shape) == 3:
                self._out_name = o.name
                break
        meta = self.sess.get_modelmeta().custom_metadata_map
        self.names = parse_names(meta.get("names")) or {}
        e2e_meta = str(meta.get("end2end", "")).lower() == "true"
        dummy = np.zeros((1, 3, INPUT_SIZE, INPUT_SIZE), np.float32)
        out = self.sess.run(None, {self.inp_name: dummy})[0]
        self.layout = self._detect_layout(out.shape, e2e_meta)
        if self.layout == "e2e" and not self.names:
            self.names = {0: "类别0"}
        elif self.layout == "v8" and not self.names:
            self.names = {i: f"类别{i}" for i in range(out.shape[1] - 4)}
        elif self.layout == "v5" and not self.names:
            self.names = {i: f"类别{i}" for i in range(out.shape[2] - 5)}
        dev = {"TensorrtExecutionProvider": "GPU (TensorRT FP16)",
               "CUDAExecutionProvider": "GPU (CUDA)", "CPUExecutionProvider": "CPU"}.get(self.provider, self.provider)
        print(f"[推理] 模型加载完成：{self.family}，类别数 {len(self.names)}，设备 {dev}", flush=True)
        # 复用预处理缓冲：避免每帧分配/拷贝（三通道strided直填一次完成BGR→RGB+转置+float32）
        self._blob = np.empty((1, 3, INPUT_SIZE, INPUT_SIZE), np.float32)
    def _detect_layout(self, shape, e2e_meta):
        if len(shape) == 3:
            _, a, b = shape
            if (e2e_meta or (b in (5, 6, 7) and a < 4000)):
                self.family = "YOLOv10/v26 端到端" if e2e_meta else "端到端"
                return "e2e"
            if a < b and a <= 128:
                self.family = "YOLOv8/v11/v12"
                return "v8"
            if a > b and b <= 128:
                self.family = "YOLOv5"
                return "v5"
        self.family = "未知结构"
        return "e2e"
    def infer(self, bgr):
        t0 = time.perf_counter()
        ih, iw = bgr.shape[:2]
        if ih == iw and iw <= INPUT_SIZE:
            # 方形且不大于模型输入：原尺度居中贴装+灰边补齐，禁止拉伸放大——
            # 放大(如448→640放大1.43倍)会让目标尺寸偏离训练分布，
            # 导致漏检真目标、枪械viewmodel被误检成敌人（框"偏移"到枪上）
            if iw < INPUT_SIZE:
                px = py = (INPUT_SIZE - iw) // 2
                canvas = np.full((INPUT_SIZE, INPUT_SIZE, 3), 114, np.uint8)
                canvas[py:py + ih, px:px + iw] = bgr
                bgr = canvas
            else:
                px = py = 0
            r = 1.0
        else:
            padded, px, py, r = letterbox(bgr)
            bgr = padded
        if bgr.shape[0] == bgr.shape[1] == INPUT_SIZE:
            # 复用缓冲直填：三通道strided拷贝一步完成BGR→RGB+HWC→CHW+float32缩放，
            # 无cvtColor、无中间分配——比blobFromImage/视图链省约1-2ms/帧
            b = self._blob[0]
            np.copyto(b[0], bgr[:, :, 2].T)
            np.copyto(b[1], bgr[:, :, 1].T)
            np.copyto(b[2], bgr[:, :, 0].T)
            b *= np.float32(1.0 / 255.0)
            blob = self._blob
        else:
            blob = cv2.dnn.blobFromImage(bgr, 1.0 / 255.0, (INPUT_SIZE, INPUT_SIZE), swapRB=True, crop=False)
        out = self.sess.run([self._out_name], {self.inp_name: blob})[0]
        dets = []
        if out.ndim == 3:
            rows = out[0]
            if self.layout == "e2e":
                dets = self._decode_e2e(rows, iw, ih, px, py, r)
            elif self.layout == "v8":
                dets = self._decode_v8(rows, iw, ih, px, py, r)
            else:
                dets = self._decode_v5(rows, iw, ih, px, py, r)
        return dets, (time.perf_counter() - t0) * 1000
    def _setup_gpu_io(self, pipeline):
        """IOBinding绑定零拷贝blob显存指针（指针持久，只需绑定一次）。
        P1a/P1b已验证：TRT FP16外部指针输入与常规路径输出逐位一致"""
        self._io = self.sess.io_binding()
        self._io.bind_input(self.inp_name, "cuda", pipeline.cuda_ordinal,
                            np.float32, [1, 3, INPUT_SIZE, INPUT_SIZE],
                            pipeline.blob_ptr)
        self._io.bind_output(self._out_name)   # CPU输出：推理引擎内部完成D2H
        self._gpu_io_ready = True
    def infer_gpu(self, pipeline):
        """零拷贝推理：管线纹理 -> NVRTC预处理(blob显存) -> TRT IOBinding -> 解码
        仅在画面更新(新帧)时调用——重复帧由引擎直接复用上次结果，不再进本函数"""
        t0 = time.perf_counter()
        if not getattr(self, "_gpu_io_ready", False):
            self._setup_gpu_io(pipeline)
        seg = self.seg
        ts = time.perf_counter()
        pipeline.stage()             # Map/线性化/预处理内核/同步 -> blob就绪
        seg["stage"] = 0.9 * seg["stage"] + 0.1 * (time.perf_counter() - ts) * 1000
        tr = time.perf_counter()
        self.sess.run_with_iobinding(self._io)
        seg["run"] = 0.9 * seg["run"] + 0.1 * (time.perf_counter() - tr) * 1000
        tc = time.perf_counter()
        out = self._io.copy_outputs_to_cpu()[0]
        seg["copy"] = 0.9 * seg["copy"] + 0.1 * (time.perf_counter() - tc) * 1000
        td = time.perf_counter()
        dets = self._decode_out(out, pipeline._w, pipeline._h)
        seg["dec"] = 0.9 * seg["dec"] + 0.1 * (time.perf_counter() - td) * 1000
        return dets, (time.perf_counter() - t0) * 1000
    def _decode_out(self, out, iw, ih):
        dets = []
        if out.ndim == 3:
            rows = out[0]
            px, py, r = 0, 0, iw / INPUT_SIZE
            if self.layout == "e2e":
                dets = self._decode_e2e(rows, iw, ih, px, py, r)
            elif self.layout == "v8":
                dets = self._decode_v8(rows, iw, ih, px, py, r)
            else:
                dets = self._decode_v5(rows, iw, ih, px, py, r)
        return dets
    def _decode_e2e(self, rows, iw, ih, px, py, r):
        result = []
        if rows.size == 0:
            return result
        if np.median(rows[:, 2] - rows[:, 0]) <= 0:
            cx, cy, bw, bh = rows[:, 0], rows[:, 1], rows[:, 2], rows[:, 3]
            x1, y1, x2, y2 = cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2
        else:
            x1, y1, x2, y2 = rows[:, 0], rows[:, 1], rows[:, 2], rows[:, 3]
        conf = rows[:, 4]
        cls = rows[:, 5] if rows.shape[1] >= 6 else np.zeros(len(rows))
        for i in np.where(conf >= self.conf)[0]:
            self._push(result, x1[i], y1[i], x2[i], y2[i], conf[i], cls[i], iw, ih, px, py, r)
        return result
    def _decode_v8(self, rows, iw, ih, px, py, r):
        """v8/v11/v12: [4+nc, anchors]，cx,cy,w,h + 每类分数"""
        nc = rows.shape[0] - 4
        if nc <= 0:
            return []
        scores = rows[4:].max(axis=0)
        cls_ids = rows[4:].argmax(axis=0)
        boxes = []
        confs = []
        clss = []
        for i in np.where(scores >= self.conf)[0]:
            cx, cy, bw, bh = rows[0, i], rows[1, i], rows[2, i], rows[3, i]
            boxes.append([cx - bw / 2, cy - bh / 2, bw, bh])
            confs.append(float(scores[i]))
            clss.append(int(cls_ids[i]))
        return self._nms(boxes, confs, clss, iw, ih, px, py, r)
    def _decode_v5(self, rows, iw, ih, px, py, r):
        """v5: [anchors, 5+nc]，cx,cy,w,h,obj + 每类分数(需乘obj)"""
        nc = rows.shape[1] - 5
        if nc <= 0:
            return []
        obj = rows[:, 4]
        scores_all = rows[:, 5:] * obj[:, None]
        scores = scores_all.max(axis=1)
        cls_ids = scores_all.argmax(axis=1)
        boxes = []
        confs = []
        clss = []
        for i in np.where(scores >= self.conf)[0]:
            cx, cy, bw, bh = rows[i, 0], rows[i, 1], rows[i, 2], rows[i, 3]
            boxes.append([cx - bw / 2, cy - bh / 2, bw, bh])
            confs.append(float(scores[i]))
            clss.append(int(cls_ids[i]))
        return self._nms(boxes, confs, clss, iw, ih, px, py, r)
    def _nms(self, boxes, confs, clss, iw, ih, px, py, r):
        keep = cv2.dnn.NMSBoxes(boxes, confs, self.conf, self.nms_iou)
        result = []
        if len(keep) == 0:
            return result
        for idx in np.array(keep).flatten():
            bx, by, bw, bh = boxes[idx]
            self._push(result, bx, by, bx + bw, by + bh, confs[idx], clss[idx], iw, ih, px, py, r)
        return result
    def _push(self, result, x1, y1, x2, y2, conf, cls, iw, ih, px, py, r):
        gx1 = max(0, min(iw, (x1 - px) / r))
        gy1 = max(0, min(ih, (y1 - py) / r))
        gx2 = max(0, min(iw, (x2 - px) / r))
        gy2 = max(0, min(ih, (y2 - py) / r))
        if gx2 - gx1 < 2 or gy2 - gy1 < 2:
            return
        result.append([gx1, gy1, gx2, gy2, float(conf), int(cls)])
