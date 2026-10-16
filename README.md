# Sentinel: Real-Time Road Accident Detection

Sentinel finds road accidents in traffic video. You upload a clip. A fine-tuned **YOLOv8m** model scans it. You get a live alert the moment a crash is confirmed, plus the video annotated with bounding boxes.

![Python](https://img.shields.io/badge/Python-3.10-3776AB?logo=python&logoColor=white)
![YOLOv8](https://img.shields.io/badge/YOLOv8m-Ultralytics-111F68)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?logo=opencv&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-009688?logo=fastapi&logoColor=white)
![Next.js](https://img.shields.io/badge/Next.js_14-000000?logo=next.js&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-2496ED?logo=docker&logoColor=white)

---

## Results

YOLOv8m, fine-tuned from the COCO-pretrained `yolov8m.pt` checkpoint on a custom accident dataset. The dataset includes negative images (normal traffic, empty roads) so the model learns that dense traffic is not a crash.

Validation metrics, read from the saved checkpoint (`backend/models/FINAL_BEST_ACCIDENT_MODEL.pt`):

| Metric | Score |
|---|---|
| Precision | **84.1%** |
| Recall | 69.1% |
| mAP@50 | **78.0%** |
| mAP@50-95 | 35.3% |

**How to read these numbers.** When the model flags an accident, it is right about 84% of the time. It misses about 31% of the accidents in the validation set. The model leans toward precision over recall, which suits alerting, because a false alarm sent to emergency services is costly. The lower mAP@50-95 means the boxes find the crash but are not tight around it. That is acceptable here, because the system's output is "accident or not", not a precise outline.

### Training setup

| Setting | Value |
|---|---|
| Base model | `yolov8m.pt` (transfer learning) |
| Image size | 640 |
| Batch size | 16 |
| Epochs | up to 100, early stopping (patience 15) |
| Learning rate | 0.01, optimizer `auto` |
| Framework | Ultralytics 8.3.252, trained on Google Colab GPU |

---

## How it works

```
Next.js frontend ──(POST /detect: video + client_id)──▶ FastAPI backend
       ▲                                                    │
       │                                    background task │
       │                                                    ▼
       └──(WebSocket /ws/{client_id})──── frame loop: resize ▶ YOLOv8m ▶ draw boxes ▶ write video
              status · progress · accident_alert · complete
```

1. **Upload.** The frontend sends the video and a client ID to `POST /detect`. The server saves the file, starts a background task, and returns right away.
2. **Frame loop** (`backend/main.py`):
   - Frames wider than 640 px are downscaled to 640 px wide, the size the model was trained on.
   - YOLOv8 runs on **every 5th frame** at confidence ≥ 0.6. The frames in between reuse the last detections, which cuts model inference by about 5×.
   - Blocking OpenCV and model calls run in a thread pool (`asyncio.to_thread`), so the server can keep sending WebSocket updates during processing.
3. **Noise filter.** One positive frame does not trigger an alert. An `accident_alert` is sent only after **5 consecutive positive checks** (about 25 frames, roughly one second of video). This stops one-frame false positives from becoming alarms.
4. **Live updates.** The client receives these WebSocket messages:

   | Message | Payload |
   |---|---|
   | `status_update` | engine status text |
   | `progress` | percent done, every 50 frames |
   | `accident_alert` | frame number and confidence, sent once when the crash is confirmed |
   | `complete` | final status (`Accident Detected` / `Normal Traffic`), annotated video URL, mean confidence |
   | `error` | error message |

5. **Result.** The annotated video is served from `/static/` and played back in the dashboard.

### Running on a free tier

The backend was deployed on a 512 MB free-tier container, so it is tuned for low memory:
- CPU-only PyTorch wheels, to keep the Docker image small.
- One thread each for Torch, OpenCV, OMP and MKL, plus `MALLOC_ARENA_MAX=2`, to stop memory fragmentation.
- `gc.collect()` every 30 frames.
- A 1,800-frame cap per video.

---

## Project structure

```
accident_detection_system/
├── backend/
│   ├── main.py            # FastAPI app: /detect, /ws/{client_id}, video pipeline
│   ├── models/            # fine-tuned YOLOv8m weights
│   ├── Dockerfile         # CPU-only, memory-tuned image
│   └── requirements.txt
├── frontend/              # Next.js 14 + TypeScript + Tailwind
│   ├── app/page.tsx       # upload flow, WebSocket client
│   └── components/        # Hero, UploadZone, Dashboard
└── start.bat              # starts backend and frontend together (Windows)
```

---

## Run it locally

**Requirements:** Python 3.10+ and Node.js 18+.

**Backend**
```bash
cd backend
pip install -r requirements.txt
python -m uvicorn main:app --reload --port 8000
```
Open `http://127.0.0.1:8000/` to confirm it reports `System Online`.

**Frontend** (in a second terminal)
```bash
cd frontend
cp .env.local.example .env.local     # sets NEXT_PUBLIC_API_URL=http://127.0.0.1:8000
npm install
npm run dev
```
Open `http://localhost:3000` and upload a traffic video.

On Windows, `start.bat` runs both steps in separate windows.

**Backend with Docker**
```bash
cd backend
docker build -t sentinel-backend .
docker run -p 8000:8000 sentinel-backend
```

---

## Limitations

- **Recall is 69%.** About a third of the accidents in the validation set are missed. This is a prototype, not a replacement for human monitoring.
- **Validation only.** The metrics come from the held-out validation split, not from a field trial on live CCTV.
- **Video uploads, not live streams.** The pipeline processes uploaded files. RTSP or webcam input would need a streaming front end.
- **Demo-grade server.** CORS allows all origins, uploads are not cleaned up, and there is no authentication. Fix these before any real deployment.

## Next steps

- Raise recall by adding more crash examples (night, rain, dashcam views) and tuning the confidence threshold against a precision-recall curve.
- Track vehicles across frames (ByteTrack) to detect collisions from motion, not single frames.
- Export to ONNX or TensorRT for faster CPU and edge inference.

---

## Author

**Sameer Choudhary**, B.Tech Data Science & AI, IIIT Naya Raipur
GitHub: [@Sameerkc3579](https://github.com/Sameerkc3579)
