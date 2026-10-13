@echo off
rem ===================================================================
rem  run_tests.cmd - runs the full test matrix for one configuration
rem  and saves each run's output to results\<config>\*.txt
rem
rem  Usage:   run_tests.cmd Release
rem           run_tests.cmd Debug
rem
rem  Program usage (for running a single case by hand):
rem      build\Release\matvec.exe [n] [m] [reps]
rem      e.g. build\Release\matvec.exe 33554432 10 30
rem  n defaults to 2^25, m to 10, reps to 30.
rem  Exit code: 0 = all kernels match C, 3 = mismatch, 2 = size not run
rem  (not enough RAM), 1 = error.
rem ===================================================================
setlocal
cd /d "%~dp0"

set "CFG=%~1"
if "%CFG%"=="" set "CFG=Release"
set "EXE=build\%CFG%\matvec.exe"
if not exist "%EXE%" (
    echo %EXE% not found - run build.cmd first.
    exit /b 1
)
set "OUT=results\%CFG%"
if not exist "%OUT%" mkdir "%OUT%"

rem --- Exact-integer self-test: every element processed exactly once
echo.
echo ##### %CFG%: exact-integer self-test  (log: %OUT%\selftest.txt)
"%EXE%" --selftest > "%OUT%\selftest.txt"
echo exit code %ERRORLEVEL%
type "%OUT%\selftest.txt"

rem --- Boundary tests: n mod 8 = 0..7 (1003 is the README boundary case)
for %%N in (1003 1004 1005 1006 1007 1008 1009 1010 4097 4099 4103) do call :run %%N 10 boundary_n%%N

rem --- Required sizes with the original m = 10 configuration
call :run 1048576   10 case1_n2p20
call :run 33554432  10 case2_n2p25

rem --- n = 2^30 / 2^29 / 2^28 with m = 10: the program checks RAM first
rem     and prints NOT RUN instead of allocating if the size cannot fit.
call :run 1073741824 10 case3_n2p30_m10
call :run 536870912  10 fallback_n2p29_m10
call :run 268435456  10 fallback_n2p28_m10

rem --- Reduced-row fallback (m = 1): only X and one row of A are stored
call :run 268435456  1  fallback_n2p28_m1

rem --- Square-matrix configuration (m = n) at sizes that fit in memory
call :run 1003 1003 square_n1003
call :run 8192 8192 square_n8192

echo.
echo All runs finished. Logs are in %OUT%\
exit /b 0

:run
echo.
echo ##### %CFG%: n=%1 m=%2  (log: %OUT%\%3.txt)
"%EXE%" %1 %2 30 > "%OUT%\%3.txt"
echo exit code %ERRORLEVEL%
type "%OUT%\%3.txt"
exit /b 0
