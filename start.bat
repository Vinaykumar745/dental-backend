@echo off
echo ========================================
echo   DentalScan AI Backend Starting...
echo ========================================
echo.

cd /d "%~dp0"

if not exist "venv" (
    echo Creating virtual environment...
    python -m venv venv
    echo Done!
    echo.
)

echo Activating virtual environment...
call venv\Scripts\activate

echo Installing/checking packages...
pip install -r requirements.txt --quiet

echo.
echo ========================================
echo   Backend running at:
echo   http://10.20.166.127:8000
echo   API Docs: http://10.20.166.127:8000/docs
echo ========================================
echo.
echo Press CTRL+C to stop the server
echo.

python main.py

pause
