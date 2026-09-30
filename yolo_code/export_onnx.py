from ultralytics import YOLO

MODEL_PATH = r"C:\Users\10650\Desktop\first-year-work\yolov11\ultralytics\runs\segment\ore_seg_project\yolov11n_ore_seg\weights\best.pt"
model = YOLO(MODEL_PATH)
# 导出onnx，固定输入640，包含后处理，simplify简化模型，方便嵌入式部署
model.export(format="onnx", imgsz=640, simplify=True, opset=17)
