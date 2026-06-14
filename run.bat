@echo off
REM Activate virtual environment and start the web app
call venv\Scripts\activate.bat
set PORT=%PORT%
if "%PORT%"=="" set PORT=8002
echo Starting app on port %PORT%
python -m uvicorn app:app --host 127.0.0.1 --port %PORT%
