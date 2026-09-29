@echo off
rem ============================================================
rem  One-shot launcher: Redis (WSL) + trading node + dashboard
rem
rem    launch.bat    keeps Redis up in WSL, starts the Nautilus trading
rem                  node in its own window, then serves the dashboard
rem                  at http://localhost:8000 in this window.
rem
rem  Strategies start IDLE: arm them from the dashboard (right rail >
rem  Active Strategies > click one > Start). Health tab = consistency report.
rem ============================================================
cd /d "%~dp0"

set PORT=8000
set WSL_DISTRO=Ubuntu
set PY=%~dp0.venv\Scripts\python.exe
set DEPS=import uvicorn, fastapi, redis, dotenv, requests, psycopg2, nautilus_trader, trading

rem --- interpreter -----------------------------------------------------
rem Always the project's own venv. "python" on PATH changes whenever another
rem Python gets installed (it became a bare 3.12 here), which broke everything.
if not exist "%PY%" goto novenv
"%PY%" -c "%DEPS%" >nul 2>&1
if not errorlevel 1 goto depsok
echo [launch] installing missing packages into .venv ...
"%PY%" -m pip install -q -e . "uvicorn[standard]" requests psycopg2-binary
"%PY%" -c "%DEPS%" >nul 2>&1
if errorlevel 1 goto nodeps
:depsok

rem --- Redis -----------------------------------------------------------
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\ensure_redis.ps1" -Distro %WSL_DISTRO%
if errorlevel 1 goto noredis

rem --- trading node ----------------------------------------------------
rem Two nodes on one Redis would fight over the same streams and strategy
rem state, so reuse a running one. (python processes only: this powershell
rem command line itself contains the search string.)
powershell -NoProfile -Command "if (Get-CimInstance Win32_Process | Where-Object { $_.Name -like 'python*' -and $_.CommandLine -like '*binance_data*' }) { exit 0 } exit 1" >nul 2>&1
if %errorlevel%==0 (
    echo [launch] trading node already running - reusing it
) else (
    echo [launch] starting trading node in a new window
    start "nautilus node" cmd /k ""%PY%" scripts\binance_data.py"
)

rem --- dashboard -------------------------------------------------------
netstat -ano | findstr /R /C:":%PORT% .*LISTENING" >nul
if %errorlevel%==0 (
    echo [launch] dashboard already running at http://localhost:%PORT% - reusing it
    start "" http://localhost:%PORT%
    goto :eof
)
echo [launch] starting dashboard at http://localhost:%PORT%
start "" /b powershell -NoProfile -WindowStyle Hidden -ExecutionPolicy Bypass -File "%~dp0scripts\open_when_ready.ps1" -Port %PORT%
"%PY%" -m uvicorn server.app:app --port %PORT%
goto :eof

:novenv
echo.
echo [launch] ERROR: .venv not found. Create it once with:
echo          py -3.14 -m venv .venv
echo          .venv\Scripts\python.exe -m pip install -e . "uvicorn[standard]" requests psycopg2-binary
echo.
pause
goto :eof

:nodeps
echo.
echo [launch] ERROR: packages still missing from .venv after install. Run this to see why:
echo          "%PY%" -c "%DEPS%"
echo.
pause
goto :eof

:noredis
echo.
echo [launch] ERROR: Redis did not answer on 127.0.0.1:6379.
echo          Check WSL:  wsl -d %WSL_DISTRO% -e sh -c "redis-cli ping"
echo          Redis log:  wsl -d %WSL_DISTRO% -u root -e tail -50 /var/log/redis/redis-server.log
echo.
pause
goto :eof
