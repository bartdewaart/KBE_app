@echo off
cd /d "%~dp0"
echo Building distribution ZIP (this may take a minute)...
powershell -Command "Compress-Archive -Path '.venv','src','data','polars','xfoil.exe','kbeutils-1.0-py3-none-any.whl','main.py','pyproject.toml','uv.lock','run.bat','README.md' -DestinationPath 'KBE_PropDesign_dist.zip' -Force"
echo.
echo Done: KBE_PropDesign_dist.zip
echo Upload this file to GitHub Releases for easy sharing.
pause
