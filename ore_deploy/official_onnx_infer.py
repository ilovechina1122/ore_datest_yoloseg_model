from ultralytics import YOLO
import os

# 配置
ONNX_PATH = "./best.onnx"
IMG_DIR = "./test_img"
OUT_DIR = "./official_result"
CONF_THRESH = 0.5
IOU_THRESH = 0.45

os.makedirs(OUT_DIR, exist_ok=True)

# 加载ONNX模型（官方原生支持，后处理内置）
model = YOLO(ONNX_PATH)

# 批量推理
results = model(
    source=IMG_DIR,
    conf=CONF_THRESH,
    iou=IOU_THRESH,
    save=True,        # 自动保存带框+mask的结果图
    project=OUT_DIR,
    name="imgs",
    verbose=True
)

# 打印每张图的结果
for i, res in enumerate(results):
    print(f"\n==== 图片 {res.path.split('/')[-1]} ====")
    for box in res.boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        cls_name = res.names[cls_id]
        xyxy = box.xyxy[0].tolist()
        print(f"类别:{cls_name}  置信度:{conf:.3f}  坐标:[{int(xyxy[0])},{int(xyxy[1])},{int(xyxy[2])},{int(xyxy[3])}]")
