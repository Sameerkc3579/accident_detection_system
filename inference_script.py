import cv2
from ultralytics import YOLO
import argparse
import os

def detect_accident(video_path, model_path, conf_threshold):
    # 1. Load Model
    if not os.path.exists(model_path):
        print(f"❌ Error: Model file not found at {model_path}")
        return

    print(f"🧠 Loading AI Model: {model_path}...")
    try:
        model = YOLO(model_path)
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return

    # 2. Open Video
    if not os.path.exists(video_path):
        print(f"❌ Error: Video file not found at {video_path}")
        return

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ Error: Could not open video source.")
        return

    # --- NEW: Setup Video Writer to Save Results ---
    # Get video properties (width, height, fps)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # Define the output file path
    output_path = "data/accident_detected_output.mp4"
    
    # Initialize the writer
    video_writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
    # -----------------------------------------------

    print(f"🎥 Processing Video: {video_path}")
    print(f"💾 Saving Output to: {output_path}")
    print("   (A window will open. Press 'q' to quit early)")
    print("-" * 40)

    # Variables for tracking
    frame_count = 0
    accident_frames = 0
    accident_detected = False

    while True:
        ret, frame = cap.read()
        if not ret:
            break # End of video

        frame_count += 1

        # 3. Run AI Inference
        results = model(frame, conf=conf_threshold, verbose=False)

        # 4. Check for Detections
        has_detection = False
        for r in results:
            if len(r.boxes) > 0:
                has_detection = True
                accident_frames += 1
                accident_detected = True
            
            # Draw boxes (This returns the BGR numpy array with boxes)
            # We overwrite 'frame' so we can save it with boxes
            frame = r.plot()

        # --- NEW: Write the frame with boxes to the file ---
        video_writer.write(frame)
        # ---------------------------------------------------

        # 5. Show Video Window
        # Resize large 4K videos to fit screen (Display only, does not affect saved file)
        display_frame = cv2.resize(frame, (1020, 600))
        cv2.imshow("Sentinel Accident Detector", display_frame)

        # Press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("🛑 User stopped the video.")
            break

    # 6. Cleanup & Final Report
    cap.release()
    video_writer.release() # <--- Release the writer!
    cv2.destroyAllWindows()

    print("\n" + "=" * 40)
    print("📊 FINAL DIAGNOSTIC REPORT")
    print("=" * 40)
    print(f"Total Frames Scanned: {frame_count}")
    
    if accident_detected:
        print(f"🚨 STATUS: ACCIDENT DETECTED")
        print(f"   The model flagged crashes in {accident_frames} frames.")
        print(f"   ✅ Output video saved: {output_path}")
    else:
        print(f"✅ STATUS: NO ACCIDENT DETECTED")
        print(f"   The video contains normal traffic.")
        print(f"   ✅ Output video saved: {output_path}")
    print("=" * 40 + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=str, required=True, help="Path to video")
    parser.add_argument("--model", type=str, default="model/FINAL_BEST_ACCIDENT_MODEL.pt", help="Path to model")
    parser.add_argument("--conf", type=float, default=0.5, help="Confidence (0.0 - 1.0)")
    
    args = parser.parse_args()
    detect_accident(args.video, args.model, args.conf)