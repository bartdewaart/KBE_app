@echo off
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
    echo ERROR: .venv not found. Please follow the setup instructions in README.md.
    pause
    exit /b 1
)
echo ========================================
echo   UAV Propulsion System Design Tool
echo ========================================
echo.
echo Opening mission.xlsx — update your parameters, save, and close it.
echo Then press any key here to start the optimisation...
echo.
start "" "data\input\mission.xlsx"
pause >nul
.venv\Scripts\python.exe main.py
echo.
echo Done. Press any key to close.
pause >nul
