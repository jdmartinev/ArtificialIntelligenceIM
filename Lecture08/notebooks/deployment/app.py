import os
import cv2
import tempfile
import numpy as np
import gradio as gr
from functools import lru_cache
from ultralytics import YOLO
import imageio

# -------- Model loading (cached) --------
@lru_cache(maxsize=2)
def load_model(weights: str = "yolov8n.pt"):
    model = YOLO(weights)
    try:
        model.fuse()  # small speed boost
    except Exception:
        pass
    return model

# -------- Drawing utils --------
def draw_label(img, text, x, y):
    """Draw text with a filled background for readability."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.5
    thickness = 1
    (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)
    cv2.rectangle(img, (x, y - th - 6), (x + tw + 6, y), (0, 0, 0), -1)
    cv2.putText(img, text, (x + 3, y - 3), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)

def draw_detections(frame: np.ndarray, result) -> np.ndarray:
    names = result.names
    if result.boxes is None or len(result.boxes) == 0:
        return frame
    for b in result.boxes:
        x1, y1, x2, y2 = b.xyxy[0].tolist()
        x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))
        cls = int(b.cls.item()) if b.cls is not None else -1
        conf = float(b.conf.item()) if b.conf is not None else 0.0
        label = f"{names.get(cls, str(cls))} {conf:.2f}"
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        draw_label(frame, label, x1, y1)
    return frame

# -------- Video processing --------
def _make_writer(w: int, h: int, fps: float):
    """
    Create a browser-playable MP4 (H.264, yuv420p, faststart) using imageio/ffmpeg,
    but expose a .write(...) and .release() API to keep the rest of the code unchanged.
    """
    out_path = tempfile.mkstemp(suffix=".mp4")[1]
    writer = imageio.get_writer(
        out_path,
        fps=max(fps, 1.0),
        codec="libx264",
        format="FFMPEG",
        macro_block_size=None,
        ffmpeg_params=["-pix_fmt", "yuv420p", "-movflags", "+faststart"],
    )

    class _Wrapper:
        def __init__(self, wtr):
            self._w = wtr
        def write(self, frame):
            # imageio expects RGB
            self._w.append_data(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        def release(self):
            self._w.close()
        def isOpened(self):  # for API symmetry; not used here
            return True

    return _Wrapper(writer), out_path

def detect_video(video_path, weights, conf, iou, max_side, progress=gr.Progress(track_tqdm=True)):
    """
    Process an uploaded video and return the path of an annotated video.
    """
    if video_path is None:
        return None

    model = load_model(weights)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError("Could not open the uploaded video.")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    fps   = float(cap.get(cv2.CAP_PROP_FPS)) or 30.0
    w     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Optional downscale for speed
    scale = 1.0
    if max(w, h) > max_side > 0:
        scale = max_side / float(max(w, h))
        w_s, h_s = int(w * scale), int(h * scale)
    else:
        w_s, h_s = w, h

    # Ensure even dims for yuv420p (browser-friendly)
    if w_s % 2: w_s -= 1
    if h_s % 2: h_s -= 1

    writer, out_path = _make_writer(w_s, h_s, fps)

    i = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if scale != 1.0:
                frame = cv2.resize(frame, (w_s, h_s), interpolation=cv2.INTER_AREA)

            # YOLO expects BGR numpy; returns list of Results
            results = model.predict(frame, conf=conf, iou=iou, verbose=False)
            annotated = draw_detections(frame, results[0])
            writer.write(annotated)

            i += 1
            if total > 0:
                progress(i / total, desc=f"Processing frame {i}/{total}")
            elif i % 30 == 0:
                progress(desc=f"Processed {i} frames...")
    finally:
        cap.release()
        writer.release()

    return out_path

# -------- Gradio UI --------
def build_ui():
    with gr.Blocks() as demo:
        gr.Markdown("## YOLO Video Detector (Upload a video ➜ Get an annotated video)")

        with gr.Row():
            in_video  = gr.Video(label="Upload video", sources=["upload"])
            out_video = gr.Video(label="Detections")

        with gr.Row():
            weights = gr.Dropdown(
                ["yolov8n.pt", "yolov8s.pt", "yolov8m.pt", "yolov8l.pt", "yolov8x.pt"],
                value="yolov8n.pt", label="Model weights"
            )
            conf = gr.Slider(0.05, 0.95, value=0.25, step=0.05, label="Confidence")
            iou  = gr.Slider(0.10, 0.90, value=0.45, step=0.05, label="IoU")
            max_side = gr.Slider(480, 1920, value=960, step=16, label="Max output side (downscale for speed)")

        run_btn = gr.Button("Run detection", variant="primary")

        run_btn.click(
            fn=detect_video,
            inputs=[in_video, weights, conf, iou, max_side],
            outputs=[out_video]
        )

    return demo

if __name__ == "__main__":
    ui = build_ui()
    ui.launch(server_name="0.0.0.0", server_port=8081, share=False)