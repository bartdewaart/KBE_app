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
echo Edit data\input\mission.xlsx to set your mission parameters, then close it.
echo Press any key when ready...
echo.
pause >nul
.venv\Scripts\python.exe main.py
echo.
echo Done. Press any key to close.
pause >nul
