# Sentinel System - Run Instructions

## 1. Start the Backend
Open a terminal in the `backend` folder:
```powershell
cd backend
pip install -r requirements.txt
python -m uvicorn main:app --reload --port 8000
```
*Wait for "Application startup complete".*

## 2. Start the Frontend
Open a NEW terminal in the `frontend` folder:
```powershell
cd frontend
npm install
npm run dev
```

## 3. Access
Open [http://localhost:3000](http://localhost:3000) in your browser.
Upload a video to test detection.
