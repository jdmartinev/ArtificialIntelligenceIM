from roboflow import Roboflow
from ultralytics import YOLO

from fastapi import UploadFile
from litserve import LitAPI, LitServer
from PIL import Image

from dotenv import load_dotenv
import os


class ObjectDetectionAPI(LitAPI):
    """API for object detection using YOLO model."""
    
    def setup(self, device):
        # Load the RF-DETR model and move it to the specified device (e.g., GPU)
        self.model = YOLO("best.pt")

    def decode_request(self, request: UploadFile) -> Image:
        # Convert the uploaded image file to RGB format for processing
        with Image.open(request.file) as img:
            image = img.convert("RGB")
        return image
    
    def predict(self, image):
        # Run object detection on the input image
        return self.model.predict(image, save = True)
    
    def encode_response(self, results):
        """
        Convert Ultralytics YOLO results (list of Results) into JSON-friendly dict.
        Uses class names from the results array itself.
        """
        detections = []
        if not results:
            return {"detections": detections}

        r = results[0]   # one Results object
        boxes = r.boxes

        if boxes is None or len(boxes) == 0:
            return {"detections": detections}

        cls = boxes.cls.cpu().numpy().astype(int)
        conf = boxes.conf.cpu().numpy()
        xyxy = boxes.xyxy.cpu().numpy()

        # names mapping is in the result object
        names = r.names  

        for class_id, confidence, bbox in zip(cls, conf, xyxy):
            detections.append({
                "class_id": int(class_id),
                "class_name": names[class_id],
                "confidence": float(confidence),
                "bbox": bbox.tolist()  # [x1, y1, x2, y2]
            })

        return {"detections": detections}

if __name__ == "__main__":
    # Launch the server on port 8000
    server = LitServer(ObjectDetectionAPI())
    server.run(port=8000)