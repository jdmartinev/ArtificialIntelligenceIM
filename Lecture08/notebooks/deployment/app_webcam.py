import os
import cv2
import numpy as np
import gradio as gr
from functools import lru_cache
from ultralytics import YOLO

# ---------- Model (cached) ----------
@lru_cache(maxsize=2)
def load_model(weights: str = "yolov8n.pt"):
    model = YOLO(weights)
    try:
        model.fuse()
    except Exception:
        pass
    return model

# ---------- Drawing ----------
def draw_label(img, text, x, y):
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale, thickness = 0.5, 1
    (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)
    cv2.rectangle(img, (x, y - th - 6), (x + tw + 6, y), (0, 0, 0), -1)
    cv2.putText(img, text, (x + 3, y - 3), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)

def draw_detections(frame: np.ndarray, result) -> np.ndarray:
    names = result.names
    boxes = getattr(result, "boxes", None)
    if boxes is None or len(boxes) == 0:
        return frame
    for b in boxes:
        x1, y1, x2, y2 = map(int, b.xyxy[0].tolist())
        cls = int(b.cls.item()) if b.cls is not None else -1
        conf = float(b.conf.item()) if b.conf is not None else 0.0
        label = f"{names.get(cls, str(cls))} {conf:.2f}"
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 210, 0), 2)
        draw_label(frame, label, x1, y1)
    return frame

# ---------- Per-frame inference ----------
def detect_frame(video_frame, weights, conf, iou, max_side):
    """
    Receives a single RGB frame from Gradio, returns an annotated RGB frame.
    """
    if video_frame is None:
        return None

    # Gradio provides RGB; YOLO expects BGR
    frame_bgr = cv2.cvtColor(video_frame, cv2.COLOR_RGB2BGR)

    h, w = frame_bgr.shape[:2]
    if max(h, w) > max_side > 0:
        scale = max_side / float(max(h, w))
        frame_bgr = cv2.resize(frame_bgr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

    model = load_model(weights)
    result = model.predict(frame_bgr, conf=conf, iou=iou, verbose=False)[0]
    annotated = draw_detections(frame_bgr, result)

    # Return RGB for display
    return cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)

# ---------- UI ----------
MODEL_CHOICES = ["best.py", "yolov8n.pt", "yolov8s.pt", "yolov8m.pt", "yolov8l.pt", "yolov8x.pt"]

def build_ui():
    with gr.Blocks() as demo:
        gr.Markdown("## YOLO Webcam Detector — Live annotations")

        with gr.Row():
            cam = gr.Image(label="Webcam", sources=["webcam"], streaming=True)
            out = gr.Image(label="Detections (live)")

        with gr.Row():
            weights = gr.Dropdown(MODEL_CHOICES, value="yolov8n.pt", label="Model weights")
            conf    = gr.Slider(0.05, 0.95, value=0.25, step=0.05, label="Confidence")
            iou     = gr.Slider(0.10, 0.90, value=0.45, step=0.05, label="IoU")
            max_side = gr.Slider(480, 1920, value=960, step=16, label="Max frame side (downscale)")

        # Stream: Gradio calls detect_frame for each incoming frame
        cam.stream(fn=detect_frame, inputs=[cam, weights, conf, iou, max_side], outputs=out)

    return demo

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=int(os.getenv("PORT", os.getenv("GRADIO_SERVER_PORT", 8081))))
    args = parser.parse_args()

    ui = build_ui()
    ui.launch(server_name="0.0.0.0", server_port=args.port, share=False)