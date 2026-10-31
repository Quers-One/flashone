@echo off
chcp 65001 >nul
setlocal enabledelayedexpansion
set "ROOT=%~dp0"

REM ---- 0) 临时目录（cl 在含空格的 TEMP 下会报 D8050，指到仓库内）----
set "TEMP=%ROOT%_tmp"
set "TMP=%ROOT%_tmp"
if not exist "%ROOT%_tmp" mkdir "%ROOT%_tmp"

REM ---- 1) MSVC ----
REM 优先用环境里已就绪的 VS，其次用 vswhere 找最新的 BuildTools/Community。
if not defined VSCMD_ARG_TGT_ARCH (
  set "VSDIR="
  if exist "%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe" (
    for /f "usebackq delims=" %%i in (`"%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe" -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath`) do set "VSDIR=%%i"
  )
  if not defined VSDIR if exist "%ProgramFiles(x86)%\Microsoft Visual Studio\2022\BuildTools" set "VSDIR=%ProgramFiles(x86)%\Microsoft Visual Studio\2022\BuildTools"
  if not defined VSDIR if exist "%ProgramFiles%\Microsoft Visual Studio\2022\Community" set "VSDIR=%ProgramFiles%\Microsoft Visual Studio\2022\Community"
  if not defined VSDIR (
    echo [build] 找不到 MSVC。请安装 "Visual Studio Build Tools 2022" 并勾选 "C++ 生成工具"。
    exit /b 1
  )
  call "!VSDIR!\VC\Auxiliary\Build\vcvars64.bat" >nul
  if errorlevel 1 ( echo [build] vcvars64 失败 & exit /b 1 )
)

REM ---- 2) libtorch：直接复用 pip 装的 torch，不额外下载 ----
REM 可用 TORCHROOT 覆盖；默认问 python 要 torch.__file__ 的目录。
if not defined TORCHROOT (
  for /f "usebackq delims=" %%i in (`python -c "import torch,os;print(os.path.dirname(torch.__file__))" 2^>nul`) do set "TORCHROOT=%%i"
)
if not defined TORCHROOT (
  echo [build] 找不到 torch。先执行: pip install torch
  exit /b 1
)
if not exist "%TORCHROOT%\share\cmake\Torch\TorchConfig.cmake" (
  echo [build] "%TORCHROOT%" 里没有 TorchConfig.cmake，TORCHROOT 可能不对。
  exit /b 1
)

REM ---- 3) CMake / Ninja：仓库内自带优先，否则用 PATH ----
if not defined CMAKE (
  if exist "%ROOT%tools\cmake-4.1.2-windows-x86_64\bin\cmake.exe" (
    set "CMAKE=%ROOT%tools\cmake-4.1.2-windows-x86_64\bin\cmake.exe"
  ) else (
    set "CMAKE=cmake"
  )
)
if not defined NINJA (
  if exist "%ROOT%tools\ninja\ninja.exe" (
    set "NINJA=%ROOT%tools\ninja\ninja.exe"
  ) else (
    set "NINJA=ninja"
  )
)

if /I "%~1"=="clean" (
  echo [build] 清理 build\
  rmdir /s /q "%ROOT%build" 2>nul
)

if not exist "%ROOT%build\CMakeCache.txt" (
  echo [build] 配置 CMake ...
  "%CMAKE%" -S "%ROOT%." -B "%ROOT%build" -G Ninja ^
    -DCMAKE_MAKE_PROGRAM="%NINJA%" ^
    -DCMAKE_BUILD_TYPE=Release ^
    -DCMAKE_PREFIX_PATH="%TORCHROOT%"
  if errorlevel 1 ( echo [build] 配置失败 & exit /b 1 )
)

echo [build] 编译 ...
"%CMAKE%" --build "%ROOT%build" --parallel
if errorlevel 1 ( echo [build] 编译失败 & exit /b 1 )
echo [build] 完成
