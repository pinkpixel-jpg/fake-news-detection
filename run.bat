@echo off
REM Activate virtual environment and start the web app
call venv\Scripts\activate.bat
python -m uvicorn app:app --host 127.0.0.1 --port 8002
