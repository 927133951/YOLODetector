# -*- mode: python ; coding: utf-8 -*-
"""CCTools 打包配置（单文件夹绿色版）
用法: pyinstaller --clean --noconfirm CCTools.spec
输出: dist/CCTools/CCTools.exe + _internal/
"""
import os
import sys
import glob
from PyInstaller.utils.hooks import copy_metadata

PROJECT = os.path.abspath(os.path.dirname(SPEC))
SP = os.path.join(sys.prefix, "Lib", "site-packages")


def collect_gpu_binaries():
    """收集 nvidia wheels 与 tensorrt_libs 的全部DLL，保留相对目录结构
    （打包后位于 _internal/nvidia/<pkg>/bin 与 _internal/tensorrt_libs，
    与源码运行时 preload_cuda_dlls/_prep_dll_dirs 的 glob 规则一致）
    注意：onnxruntime_providers_cuda.dll 的导入表依赖 cufft64_11.dll，
    不可排除 cufft，否则 CUDA Provider 加载失败回退 CPU"""
    out = {}
    patterns = [
        (os.path.join(SP, "nvidia", "**", "*.dll"), None),
        (os.path.join(SP, "tensorrt_libs", "*.dll"), "tensorrt_libs"),
        (os.path.join(SP, "tensorrt_libs", "**", "*.dll"), None),
    ]
    for pattern, forced_dest in patterns:
        for dll in glob.glob(pattern, recursive=True):
            if forced_dest:
                dest = forced_dest
            else:
                dest = os.path.relpath(os.path.dirname(dll), SP).replace("\\", "/")
            key = (dest + "/" + os.path.basename(dll)).lower()
            out[key] = (dll, dest)
    return list(out.values())


def collect_cuda_core():
    """收集 cuda-core 的 cu13 版本化子包真实文件，摊平放入 _internal/cuda/core/。
    NVIDIA merged-wheel 的 cuda.core 在运行时才动态 import cu13 并把
    cuda.core.__path__ 劫持指向 cu13 目录（机制藏在未校验 pyc 中），PyInstaller
    静态分析不可见：PYZ 只含 dispatcher 壳，运行时 import cuda.core 必然报
    "cannot import name 'checkpoint' from partially initialized module 'cuda.core'"。
    摊平后 dispatcher 的 `from cuda.core import checkpoint/system/utils` 与
    `from cuda.core._context import *` 等直接经 __path__（_MEIPASS/cuda/core）
    命中磁盘文件，模块名与源码环境最终状态一致（均为 cuda.core._X）。
    同时携带 delvewheel 依赖目录（pyd 导入表按哈希名引用 msvcp140 等 DLL，
    官方 delvewheel patch 在 __init__ 里按 __file__ 相对路径自动 add_dll_directory）"""
    out = []
    cu13 = os.path.join(SP, "cuda", "core", "cu13")
    for root, dirs, files in os.walk(cu13):
        dirs[:] = [d for d in dirs if d not in ("__pycache__", "_cpp", "_include")]
        for f in files:
            if os.path.splitext(f)[1].lower() not in (".py", ".pyd"):
                continue
            src = os.path.join(root, f)
            rel = os.path.relpath(root, cu13)
            dest = os.path.join("cuda", "core") if rel == "." else os.path.join("cuda", "core", rel)
            out.append((src, dest))
    # cuda/bindings 的 Cython pyd（cydriver/cyruntime/_internal/_v2 等）由
    # driver/runtime 大包装 pyd 在 C 层动态导入，PyInstaller 静态分析只收集到
    # driver/runtime/nvvm 三个，其余全部丢失 → "No module named cuda.bindings.cydriver"。
    # 摊平整个 bindings 目录（pxd/pxi 为 Cython 源码声明，运行时不需要）
    for root, dirs, files in os.walk(os.path.join(SP, "cuda", "bindings")):
        dirs[:] = [d for d in dirs if d != "__pycache__"]
        for f in files:
            if os.path.splitext(f)[1].lower() in (".pxd", ".pxi", ".pyi"):
                continue
            src = os.path.join(root, f)
            rel = os.path.relpath(root, os.path.join(SP, "cuda", "bindings"))
            dest = os.path.join("cuda", "bindings") if rel == "." else os.path.join("cuda", "bindings", rel)
            out.append((src, dest))
    for libs in ("cuda_core.libs", "cuda_bindings.libs"):
        ldir = os.path.join(SP, libs)
        if os.path.isdir(ldir):
            for f in os.listdir(ldir):
                out.append((os.path.join(ldir, f), libs))
    return out


a = Analysis(
    ['main.py'],
    pathex=[PROJECT],
    binaries=collect_gpu_binaries(),
    datas=[
        ('models/v26m_0626_640.onnx', 'models'),
        ('MouseControl/MouseControl.dll', 'MouseControl'),
        ('assets/icon.ico', 'assets'),
        *collect_cuda_core(),
        # cuda 包运行时经 importlib.metadata 查版本，frozen 下必须带 dist-info
        *copy_metadata("cuda-python"),
        *copy_metadata("cuda-bindings"),
        *copy_metadata("cuda-core"),
        *copy_metadata("cuda-pathfinder"),
    ],
    hiddenimports=['cuda.pathfinder', 'cuda.pathfinder._optional_cuda_import'],
    hookspath=[],
    runtime_hooks=[],
    excludes=[
        'onnx', 'tensorrt', 'PyQt5', 'torch', 'ultralytics', 'matplotlib',
        'scipy', 'sklearn', 'librosa', 'numba', 'llvmlite', 'winrt',
        'PySide2', 'PySide6', 'pandas',
    ],
    noarchive=False,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='CCTools',
    debug=False,
    strip=False,
    upx=False,
    console=False,
    icon='assets/icon.ico',
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    name='CCTools',
)
