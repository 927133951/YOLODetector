@echo off
title CCTools Launcher
net session >nul 2>&1
if %errorlevel% neq 0 (
    echo Requesting administrator privileges, please click Yes on UAC prompt...
    powershell -NoProfile -Command "Start-Process -FilePath '%~f0' -Verb RunAs"
    exit /b
)
cd /d "%~dp0"
echo ============================================
echo   CCTools - environment check and launch
echo   Running as Administrator
echo ============================================
where python >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python not found. Install Python 3.10/3.13 and enable "Add to PATH".
    pause
    exit /b
)
echo [1/3] Checking dependencies...
python -c "import numpy,cv2,onnxruntime,comtypes,mss" >nul 2>&1
if %errorlevel% neq 0 (
    echo Missing packages, installing via Tsinghua mirror, please wait...
    pip install -i https://pypi.tuna.tsinghua.edu.cn/simple numpy opencv-python onnxruntime-gpu comtypes mss tensorrt
    if %errorlevel% neq 0 (
        echo [ERROR] Dependency install failed. Check your network and rerun.
        pause
        exit /b
    )
) else (
    echo Dependencies OK
)
echo [2/3] Checking NVIDIA GPU...
nvidia-smi >nul 2>&1
if %errorlevel% neq 0 (
    echo [WARNING] NVIDIA GPU or driver not found. GPU inference is required.
    echo Please install the NVIDIA driver and try again.
    pause
    exit /b
) else (
    echo NVIDIA GPU detected
)
echo [3/3] Launching CCTools...
rem Launch via CCTools.exe (copy of pythonw) so the process shows as CCTools.exe
set "PYW=pythonw.exe"
set "PYDIR="
for /f "delims=" %%i in ('where python') do if not defined PYDIR set "PYDIR=%%~dpi"
if defined PYDIR if exist "%PYDIR%pythonw.exe" (
    if not exist "%PYDIR%CCTools.exe" copy /y "%PYDIR%pythonw.exe" "%PYDIR%CCTools.exe" >nul 2>&1
    if exist "%PYDIR%CCTools.exe" set "PYW=%PYDIR%CCTools.exe"
)
start "" "%PYW%" main.py
timeout /t 3 /nobreak >nul
exit
