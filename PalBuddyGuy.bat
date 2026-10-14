@echo off
rem ==========================================================================
rem  Pal Buddy Guy launcher - just double-click this file.
rem
rem  First run: finds (or offers to install) Python, creates a private
rem  environment in .venv, installs PyTorch (CUDA build if an NVIDIA GPU is
rem  found) and the other packages, and offers a desktop shortcut.
rem  Every later run: starts the GUI immediately, without a console window.
rem
rem  To reinstall from scratch, delete the .venv folder and run this again.
rem  (This file is plain ASCII on purpose: UTF-8 batch files can misbehave.)
rem ==========================================================================
setlocal EnableExtensions
cd /d "%~dp0"
title Pal Buddy Guy

set "APPDIR=%~dp0"
set "VENV=%APPDIR%.venv"
set "PY=%VENV%\Scripts\python.exe"
set "PYW=%VENV%\Scripts\pythonw.exe"

if exist "%VENV%\.installed" if exist "%PYW%" goto run

echo.
echo  ==============================================
echo    Pal Buddy Guy - first-time setup
echo  ==============================================
echo.

rem ---------------------------------------------------------------- Python
rem Prefer versions PyTorch surely has wheels for; "py -3" alone picks the newest.
set "CHECK=import sys, tkinter; sys.exit(0 if (3, 9) <= sys.version_info[:2] <= (3, 13) else 1)"
set "SYS_PY="
call :probe py -3.12
call :probe py -3.11
call :probe py -3.13
call :probe py -3.10
call :probe python
if defined SYS_PY goto have_python

echo  A suitable Python (3.10 - 3.13, with tkinter) was not found.
where winget >nul 2>&1
if errorlevel 1 goto manual_python
choice /C YN /M " Install Python 3.12 with winget"
if errorlevel 2 goto manual_python
winget install -e --id Python.Python.3.12 --scope user --accept-package-agreements --accept-source-agreements
set "SYS_PY=%LOCALAPPDATA%\Programs\Python\Python312\python.exe"
if exist "%SYS_PY%" goto have_python
echo  Please run this file again after the Python install finishes.
pause
exit /b 1

:manual_python
echo  Install Python 3.12 from https://www.python.org/downloads/ and run this file again.
echo  (Tick "Add python.exe to PATH" in the installer.)
start "" "https://www.python.org/downloads/windows/"
pause
exit /b 1

:have_python
echo  Python: %SYS_PY%

rem ---------------------------------------------------------------- venv
if exist "%PY%" goto venv_ready
echo  Creating the environment...
"%SYS_PY%" -m venv "%VENV%"
if errorlevel 1 goto fail
:venv_ready
"%PY%" -m pip install --upgrade pip >nul

rem ---------------------------------------------------------------- PyTorch
set "TORCH_INDEX=https://download.pytorch.org/whl/cpu"
set "GPU=none"
where nvidia-smi >nul 2>&1
if errorlevel 1 goto install_torch
for /f "delims=" %%g in ('nvidia-smi --query-gpu^=name --format^=csv^,noheader 2^>nul') do set "GPU=%%g"
set "TORCH_INDEX=https://download.pytorch.org/whl/cu126"
rem RTX 50xx (Blackwell) needs the CUDA 12.8 build
echo %GPU% | findstr /R /C:"RTX 50[0-9][0-9]" >nul
if not errorlevel 1 set "TORCH_INDEX=https://download.pytorch.org/whl/cu128"

:install_torch
echo  GPU: %GPU%
echo  Installing PyTorch (2-3 GB download, this takes a few minutes)...
"%PY%" -m pip install torch --index-url %TORCH_INDEX%
if errorlevel 1 goto fail
echo  Installing the other packages...
"%PY%" -m pip install -r "%APPDIR%requirements.txt"
if errorlevel 1 goto fail

rem ONNX Runtime: faster / lighter tracking. The DirectML build also runs on AMD and Intel GPUs.
rem Optional - if it fails, tracking just uses PyTorch.
"%PY%" -m pip uninstall -y onnxruntime >nul 2>&1
"%PY%" -m pip install onnxruntime-directml
if errorlevel 1 "%PY%" -m pip install onnxruntime

"%PY%" -c "import torch; print(' PyTorch', torch.__version__, '/ CUDA:', torch.cuda.is_available())"
if errorlevel 1 goto fail
echo ok> "%VENV%\.installed"

rem ---------------------------------------------------------------- shortcut
echo.
choice /C YN /M " Create a desktop shortcut"
if errorlevel 2 goto run
powershell -NoProfile -ExecutionPolicy Bypass -Command ^
  "$s = (New-Object -ComObject WScript.Shell).CreateShortcut([Environment]::GetFolderPath('Desktop') + '\Pal Buddy Guy.lnk');" ^
  "$s.TargetPath = $env:PYW; $s.Arguments = '-m palbuddy'; $s.WorkingDirectory = $env:APPDIR;" ^
  "$s.Description = 'Pal Buddy Guy'; $s.Save()"

:run
rem pythonw = no console window. Errors are shown in a dialog and written to palbuddy.log.
start "" "%PYW%" -m palbuddy %*
exit /b 0

:fail
echo.
echo  Setup failed - see the messages above. Delete the .venv folder to start over.
pause
exit /b 1

rem ---------------------------------------------------------------- helpers
rem call :probe <python launcher and args> - sets SYS_PY if it is a suitable Python
:probe
if defined SYS_PY exit /b 0
set "CAND="
for /f "delims=" %%i in ('%* -c "import sys; print(sys.executable)" 2^>nul') do set "CAND=%%i"
if not defined CAND exit /b 0
"%CAND%" -c "%CHECK%" >nul 2>&1
if not errorlevel 1 set "SYS_PY=%CAND%"
exit /b 0
