from ultralytics import  YOLO
from ultralytics import NAS
model = YOLO('yolov8n-seg.pt')
model = YOLO('molinosag.pt')

model.export(format='onnx')