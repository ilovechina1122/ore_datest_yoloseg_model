from ultralytics import YOLO
import cv2
import os

# ====================== 配置区，修改这里 ======================
MODEL_PATH = r"C:\Users\10650\Desktop\first-year-work\yolov11\ultralytics\runs\segment\ore_seg_project\yolov11n_ore_seg\weights\best.pt"
# 存放你下载的网上矿石图片文件夹
IMG_FOLDER = r"C:\Users\10650\Desktop\ore_dataset_open\test_online"
# 推理结果保存目录
SAVE_FOLDER = r"C:\Users\10650\Desktop\ore_dataset_open\test_online_result"
CONF_THRESH = 0.5  # 置信度阈值，低于这个不显示
# =============================================================

def preprocess_image(img_path, target_size=640):
    """
    图像预处理：读取任意格式图片，保证RGB，自动缩放+padding，返回原始图+预处理后图
    """
    # cv2读取，支持jpg/png/webp
    img = cv2.imread(img_path)
    if img is None:
        print(f"⚠️ 无法读取图片：{img_path}，跳过")
        return None, None
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = img.shape[:2]
    # YOLO原生letterbox预处理（保持比例，填充黑边，不拉伸）
    scale = min(target_size / w, target_size / h)
    new_w, new_h = int(w * scale), int(h * scale)
    img_resized = cv2.resize(img_rgb, (new_w, new_h))
    # 创建640×640画布，填充灰色padding
    canvas = 114 * np.ones((target_size, target_size, 3), dtype=np.uint8)
    canvas[:new_h, :new_w, :] = img_resized
    return img, canvas

if __name__ == '__main__':
    import numpy as np
    # 创建输出文件夹
    os.makedirs(SAVE_FOLDER, exist_ok=True)
    # 加载模型
    model = YOLO(MODEL_PATH)
    img_list = [f for f in os.listdir(IMG_FOLDER) if f.lower().endswith(('.jpg','.jpeg','png','webp','bmp'))]
    print(f"一共找到 {len(img_list)} 张待测试图片")

    for img_name in img_list:
        img_full_path = os.path.join(IMG_FOLDER, img_name)
        origin_img, _ = preprocess_image(img_full_path, target_size=640)
        if origin_img is None:
            continue

        # 推理
        results = model(img_full_path, conf=CONF_THRESH)
        res = results[0]

        # 绘制分割掩码并保存
        plotted_img = res.plot()
        save_path = os.path.join(SAVE_FOLDER, img_name)
        cv2.imwrite(save_path, plotted_img)

        # 打印检测信息
        boxes = res.boxes
        masks = res.masks
        print(f"\n==== {img_name} ====")
        if boxes is not None:
            for box in boxes:
                cls_id = int(box.cls)
                conf = float(box.conf)
                cls_name = model.names[cls_id]
                print(f"类别:{cls_name}, 置信度:{conf:.3f}")
            print(f"检测到目标数量：{len(boxes)}")
        else:
            print("未检测到矿石目标")
    print("\n✅ 全部图片推理完成！结果保存在：", SAVE_FOLDER)
