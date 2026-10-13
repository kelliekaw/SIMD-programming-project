@echo off
rem ===================================================================
rem  build.cmd - builds Debug and Release x64 executables from cmd.exe
rem  without opening Visual Studio.
rem
rem  Usage:   build.cmd            (builds Debug and Release)
rem           build.cmd Debug      (Debug only)
rem           build.cmd Release    (Release only)
rem
rem  Output:  build\Debug\matvec.exe and build\Release\matvec.exe
rem
rem  NASM:    set NASM=C:\path\to\nasm.exe to override the search.
rem ===================================================================
setlocal EnableDelayedExpansion
cd /d "%~dp0"

set "CONFIGS=%~1"
if "%CONFIGS%"=="" set "CONFIGS=Debug Release"

rem ---- Locate NASM -------------------------------------------------
if defined NASM if exist "%NASM%" goto :have_nasm
set "NASM="
for %%P in ("C:\mingw64\bin\nasm.exe" "C:\Program Files\NASM\nasm.exe" "C:\Program Files (x86)\SASM\NASM\nasm.exe" "%LOCALAPPDATA%\bin\NASM\nasm.exe") do (
    if not defined NASM if exist %%P set "NASM=%%~P"
)
if not defined NASM for /f "delims=" %%P in ('where nasm 2^>nul') do if not defined NASM set "NASM=%%P"
if not defined NASM (
    echo ERROR: nasm.exe not found. Install NASM or run: set NASM=C:\path\to\nasm.exe
    exit /b 1
)
:have_nasm
echo Using NASM: %NASM%

rem ---- Set up the MSVC x64 environment if cl.exe is not on PATH -----
where cl >nul 2>&1
if not errorlevel 1 goto :have_msvc
set "VSWHERE=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe"
if not exist "%VSWHERE%" (
    echo ERROR: vswhere.exe not found. Run this from an "x64 Native Tools Command Prompt".
    exit /b 1
)
set "VSDIR="
rem Run vswhere from its own folder: "(x86)" in a quoted path breaks for /f.
pushd "%ProgramFiles(x86)%\Microsoft Visual Studio\Installer"
for /f "delims=" %%I in ('.\vswhere.exe -all -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath') do set "VSDIR=%%I"
popd
if not defined VSDIR (
    echo ERROR: No Visual Studio installation with the C++ x64 tools was found.
    echo        Install the "Desktop development with C++" workload.
    exit /b 1
)
echo Using MSVC from: %VSDIR%
call "%VSDIR%\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1
if errorlevel 1 exit /b 1
:have_msvc

rem ---- Assemble the kernels (shared by both configurations) ---------
if not exist build mkdir build
for %%K in (scalar xmm ymm) do (
    "%NASM%" -f win64 kernel_%%K.asm -o build\kernel_%%K.obj
    if errorlevel 1 ( echo ERROR: NASM failed on kernel_%%K.asm & exit /b 1 )
)

rem ---- Compile and link each configuration --------------------------
rem  Flags mirror the Visual Studio project:
rem    Debug  : /Od /RTC1 /MDd /D_DEBUG          (no optimization)
rem    Release: /O2 /Oi /GL /Gy /MD /DNDEBUG + /LTCG
rem  Both use /fp:precise so the compiler never reorders float additions.
for %%C in (%CONFIGS%) do (
    echo.
    echo ===== Building %%C =====
    if not exist build\%%C mkdir build\%%C
    if /i "%%C"=="Debug" (
        set "CFLAGS=/Od /RTC1 /MDd /D_DEBUG /Zi"
        set "LFLAGS=/DEBUG"
    ) else (
        set "CFLAGS=/O2 /Oi /GL /Gy /MD /DNDEBUG /Zi"
        set "LFLAGS=/LTCG /DEBUG /OPT:REF /OPT:ICF"
    )
    cl /nologo /W3 /sdl /fp:precise /D_CONSOLE !CFLAGS! /Fo"build\%%C\\" /Fd"build\%%C\\" main.c build\kernel_scalar.obj build\kernel_xmm.obj build\kernel_ymm.obj /Fe"build\%%C\matvec.exe" /link /SUBSYSTEM:CONSOLE !LFLAGS!
    if errorlevel 1 ( echo ERROR: %%C build failed & exit /b 1 )
    echo Built build\%%C\matvec.exe
)

echo.
echo Build succeeded.
exit /b 0
