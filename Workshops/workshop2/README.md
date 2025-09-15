# YOLOv8 Fine-tuning & Deployment Workshop

## Overview
In this workshop, you will:
1. Fine-tune a **YOLOv8** object detection model on the [Logistics dataset](https://universe.roboflow.com/large-benchmark-datasets/logistics-sz9jr/browse).
2. Deploy your trained model using **LitServe** (Lightning AI).

---

## Part 0 — Setup (15 min)

```bash
# create and activate virtual environment (optional)
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate

# install dependencies
pip install ultralytics==8.* litserve==0.* fastapi uvicorn pillow opencv-python

# optional extras
pip install supervision roboflow
```

---

## Part 1 — Get the Dataset (20 min)

### Option A — Manual Download
1. Open the dataset page.
2. Click **Download** → choose **YOLOv8 / Ultralytics** format.
3. Unzip into `datasets/logistics/`.
4. Verify `dataset.yaml` includes correct `train/val/test` paths.

### Option B — Programmatic (with Roboflow API key)
```python
from roboflow import Roboflow
rf = Roboflow(api_key="YOUR_KEY")
project = rf.workspace("large-benchmark-datasets").project("logistics-sz9jr")
dataset = project.version(2).download("yolov8")
```

---

## Part 2 — Fine-tune YOLOv8 (45 min)

### CLI
```bash
yolo detect train model=yolov8n.pt data=datasets/logistics/dataset.yaml epochs=30 imgsz=640 batch=16 project=runs/logistics name=y8n
```

### Python API
```python
from ultralytics import YOLO
model = YOLO("yolov8n.pt")
model.train(data="datasets/logistics/dataset.yaml", epochs=30, imgsz=640, batch=16, project="runs/logistics", name="y8n")
```

The best weights will be saved at:
```
runs/logistics/y8n/weights/best.pt
```

---

## Part 3 — Validate & Export (15 min)

```bash
# evaluate
yolo detect val model=runs/logistics/y8n/weights/best.pt data=datasets/logistics/dataset.yaml

# export
yolo export model=runs/logistics/y8n/weights/best.pt format=onnx
```

---

## Part 4 — Deploy with LitServe (60 min)

### server.py
```python
import io
from typing import List, Dict, Any

import litserve as ls
from fastapi import UploadFile
from PIL import Image
from ultralytics import YOLO

MODEL_PATH = "runs/logistics/y8n/weights/best.pt"

class LogisticsAPI(ls.LitAPI):
    def setup(self, device: str):
        self.model = YOLO(MODEL_PATH)

    def decode_request(self, request: Dict[str, Any]):
        if isinstance(request, dict) and "image" in request:
            f: UploadFile = request["image"]
            image = Image.open(io.BytesIO(f.file.read())).convert("RGB")
            return image
        raise ValueError("Expected form-data with 'image'")

    def predict(self, image: Image.Image) -> List[Dict[str, Any]]:
        results = self.model.predict(image, conf=0.25, imgsz=640, verbose=False)[0]
        dets = []
        for b in results.boxes:
            cls_id = int(b.cls[0].item())
            dets.append({
                "class_id": cls_id,
                "class_name": results.names[cls_id],
                "confidence": float(b.conf[0].item()),
                "xyxy": [float(x) for x in b.xyxy[0].tolist()],
            })
        return dets

    def encode_response(self, output: List[Dict[str, Any]]):
        return {"detections": output}

if __name__ == "__main__":
    server = ls.LitServer(LogisticsAPI(), accelerator="auto")
    server.run(port=8000)
```

### Run
```bash
python server.py
```

### Test
```bash
curl -X POST "http://127.0.0.1:8000/predict" \
  -H "Content-Type: multipart/form-data" \
  -F "image=@sample.jpg"
```

---

## Stretch Goals
- Export to TensorRT or CoreML.
- Visualize predictions with `supervision`.
- Compare with Roboflow’s hosted baseline model.

---
### Resources

- You can follow this [Ultralitics tutorial](https://docs.ultralytics.com/es/usage/python/#train) to fine-tune the model.

- You can follow this [ROBOFLOW tutorial](https://lightning.ai/bhimrajyadav/studios/deploy-rf-detr-a-sota-real-time-object-detection-model-using-litserve?view=public&section=featured&query=detection) to deploy the model.
---

## Agenda (3–4 hours)
1. Intro & dataset briefing — 10m  
2. Environment & dataset setup — 20m  
3. Training — 40–60m  
4. Validation & export — 15m  
5. Deployment with LitServe — 45–60m  
6. Wrap-up & discussion — 15m  

---

## Deliverables
- **Trained model weights**
- **Validation metrics**
- **Working LitServe API**
- **Short report (1–2 pages)**
