import os
import cv2
import shutil
import uuid
import asyncio
import traceback
from typing import List
from fastapi import FastAPI, File, UploadFile, WebSocket, WebSocketDisconnect, BackgroundTasks, Form
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO

app = FastAPI(title="Sentinel AI Backend")

# Setup Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "FINAL_BEST_ACCIDENT_MODEL.pt")
STATIC_DIR = os.path.join(BASE_DIR, "static")

# Ensure static directory exists
os.makedirs(STATIC_DIR, exist_ok=True)

# CORS Setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_origin_regex='.*', # Explicitly allow regex for stricter envs
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# Connection Manager for WebSockets
class ConnectionManager:
    def __init__(self):
        self.active_connections: dict[str, WebSocket] = {}

    async def connect(self, websocket: WebSocket, client_id: str):
        await websocket.accept()
        self.active_connections[client_id] = websocket
        print(f"Client {client_id} connected")

    def disconnect(self, client_id: str):
        if client_id in self.active_connections:
            del self.active_connections[client_id]
            print(f"Client {client_id} disconnected")

    async def send_personal_message(self, message: dict, client_id: str):
        if client_id in self.active_connections:
            await self.active_connections[client_id].send_json(message)

manager = ConnectionManager()

# Load YOLO Model
try:
    print(f"Loading model from {MODEL_PATH}...")
    model = YOLO(MODEL_PATH)
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

@app.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str):
    await manager.connect(websocket, client_id)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(client_id)

async def process_video(input_path: str, output_path: str, output_filename: str, client_id: str):
    """
    Background task to process video and send real-time updates via WebSocket.
    """
    print(f"Starting processing for client {client_id}")
    await manager.send_personal_message(
        {"type": "status_update", "status": "Initializing Video Engine..."},
        client_id
    )
    
    # Open video in a thread to avoid blocking
    cap = await asyncio.to_thread(cv2.VideoCapture, input_path)
    
    if not await asyncio.to_thread(cap.isOpened):
        await manager.send_personal_message(
            {"type": "error", "message": "Could not open video file."}, 
            client_id
        )
        return

    # Video properties
    original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0: fps = 30

    # Resize logic
    target_width = original_width
    target_height = original_height
    if original_width > 640:
        scale_ratio = 640 / original_width
        target_width = 640
        target_height = int(original_height * scale_ratio)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
    
    # Create writer in thread
    out = await asyncio.to_thread(cv2.VideoWriter, output_path, fourcc, fps, (target_width, target_height))
    
    if not await asyncio.to_thread(out.isOpened):
        print("Error: Could not initialize video writer.")
        await manager.send_personal_message(
            {"type": "error", "message": "Server video codec error."}, 
            client_id
        )
        return

    frame_count = 0
    total_confidence = 0
    detections_count = 0
    accident_detected_frames = 0
    consecutive_accident_frames = 0
    accident_alert_sent = False
    
    total_frames_est = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Processing video: {output_filename} ({target_width}x{target_height}) - {total_frames_est} frames")

    SKIP_FRAMES = 5
    ALERT_THRESHOLD = 5 # Require 5 consecutive checks (approx 25 frames / ~1 sec) to trigger alert
    last_results = None

    try:
        while True:
            # Non-blocking read
            ret, frame = await asyncio.to_thread(cap.read)
            if not ret:
                break
            
            # Resize frame
            if target_width != original_width:
                frame = await asyncio.to_thread(cv2.resize, frame, (target_width, target_height))

            # Run YOLO inference
            if frame_count % SKIP_FRAMES == 0:
                # Run model prediction in thread pool
                results = await asyncio.to_thread(model.predict, frame, conf=0.6, verbose=False)
                last_results = results
            
            # Draw results & Check for accident
            if last_results:
                # Plotting can be slow, move to thread
                res_plotted = await asyncio.to_thread(last_results[0].plot, img=frame)
                
                if frame_count % SKIP_FRAMES == 0:
                    boxes = last_results[0].boxes
                    if len(boxes) > 0:
                        detections_count += len(boxes)
                        
                        # Calculate current frame confidence
                        current_conf_sum = sum(box.conf.item() for box in boxes)
                        current_avg_conf = current_conf_sum / len(boxes)
                        
                        total_confidence += current_conf_sum
                        accident_detected_frames += 1
                        consecutive_accident_frames += 1
                        
                        # IMMEDIATE ALERT NOTIFICATION
                        # We require noise filtering: must detect accident in 'ALERT_THRESHOLD' consecutive checks
                        if consecutive_accident_frames >= ALERT_THRESHOLD and not accident_alert_sent:
                            print(f"ACCIDENT CONFIRMED at frame {frame_count} (Consecutive: {consecutive_accident_frames})! Sending alert...")
                            await manager.send_personal_message(
                                {
                                    "type": "accident_alert",
                                    "status": "Accident Detected",
                                    "frame": frame_count,
                                    "confidence": float(f"{current_avg_conf:.2f}")
                                }, 
                                client_id
                            )
                            accident_alert_sent = True
                    else:
                        # Reset consecutive counter if no accident detected in this check
                        consecutive_accident_frames = 0
            else:
                res_plotted = frame

            # Write frame in thread
            await asyncio.to_thread(out.write, res_plotted)
            frame_count += 1
            
            # Progress updates (optional, every 10% or so)
            if total_frames_est > 0 and frame_count % 50 == 0:
                progress = int((frame_count / total_frames_est) * 100)
                await manager.send_personal_message(
                    {"type": "progress", "progress": progress}, 
                    client_id
                )

            # Safety break
            if frame_count > 1800:
                print("Video too long, truncating...")
                break
    except Exception as e:
        print(f"Error during processing loop: {e}")
        traceback.print_exc()
        await manager.send_personal_message(
            {"type": "error", "message": f"Processing Server Error: {str(e)}"},
            client_id
        )
        # Try to clean up
    finally:
        await asyncio.to_thread(cap.release)
        await asyncio.to_thread(out.release)
    
    # Calculate metrics
    avg_conf = 0.0
    if detections_count > 0:
        avg_conf = total_confidence / detections_count
    
    status = "Normal Traffic"
    if accident_detected_frames > 0:
        status = "Accident Detected"

    # Use Render URL if available, else localhost
    base_url = os.getenv("RENDER_EXTERNAL_URL", "http://127.0.0.1:8000")
    video_url = f"{base_url}/static/{output_filename}"

    print(f"Processing complete for {client_id}. Status: {status}")
    
    # Final Completion Message
    await manager.send_personal_message(
        {
            "type": "complete",
            "status": status,
            "video_url": video_url,
            "confidence": float(f"{avg_conf:.2f}")
        }, 
        client_id
    )

@app.post("/detect")
async def detect_accident(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    client_id: str = Form(...)
):
    if model is None:
        return {"status": "Error", "message": "Model not loaded."}

    # Generate unique filenames
    file_id = str(uuid.uuid4())
    filename = f"{file_id}_{file.filename}"
    input_path = os.path.join(STATIC_DIR, f"input_{filename}")
    output_filename = f"processed_{filename}"
    output_path = os.path.join(STATIC_DIR, output_filename)

    # Save uploaded file
    with open(input_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Start processing in background
    background_tasks.add_task(process_video, input_path, output_path, output_filename, client_id)

    return {"status": "Processing Started", "message": "Video analysis has begun in the background."}
