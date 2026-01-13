@echo off
echo Starting Sentinel AI System...

:: Start Backend
start "Sentinel Backend" cmd /k "cd backend && pip install -r requirements.txt && python -m uvicorn main:app --reload --port 8000"

:: Start Frontend
start "Sentinel Frontend" cmd /k "cd frontend && npm install && npm run dev"

echo.
echo ===================================================
echo   Servers are starting in separate windows.
echo   Once loaded, open: http://localhost:3000
echo ===================================================
echo.
pause
