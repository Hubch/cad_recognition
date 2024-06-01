@echo off

REM Get the current script directory
set SCRIPT_DIR="%~dp0venv\Scripts\activate.bat"
echo Script directory: %SCRIPT_DIR%

REM Run the backend
cd backend
start /B cmd /c "call %SCRIPT_DIR%  && call poetry run uvicorn main:app --reload --port 8001"

REM Wait for the frontend command to finish
timeout /t 5 /nobreak

cd ..
REM Run the frontend
cd frontend
start /B cmd /c "call yarn dev &&  http://127.0.0.1:5137"

REM Wait for the frontend command to finish
timeout /t 5 /nobreak

start http://localhost:5173



REM Wait for the frontend command to finish
timeout /t 5 /nobreak

cd ..
REM Run model serve
cd modelserve
start /B cmd /c "call %SCRIPT_DIR%  && call  ray stop && call  ray start --head && call serve run config.yaml"



pause
