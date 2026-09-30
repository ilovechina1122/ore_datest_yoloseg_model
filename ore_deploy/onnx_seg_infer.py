import cv2
import numpy as np
import onnxruntime as ort
import os
import onnx
import time
from PIL import Image
# ========== 可视化配色（BGR），按类别数自动循环 ==========
COLORS = [
    (0, 255, 0), (0, 0, 255), (255, 0, 0),
    (0, 255, 255), (255, 0, 255), (255, 255, 0),
    (128, 0, 255), (0, 128, 255),
]

def load_class_names(model_path):
    """从 Ultralytics 导出的 ONNX 元数据里读取类别名"""
    model = onnx.load(model_path)
    props = {p.key: p.value for p in model.metadata_props}
    names_str = props.get('names', '{}')
    names = eval(names_str)
    class_names = [names[i] for i in sorted(names.keys())]
    return class_names

class YOLOv11SegRunner:
    def __init__(self, model_path, conf_thres=0.25, iou_thres=0.45):
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.input_size = 640
        # 自动读取类别
        self.class_names = load_class_names(model_path)
        self.num_classes = len(self.class_names)
        print(f"自动读取类别: {self.class_names} (共 {self.num_classes} 类)")
        available_providers = ort.get_available_providers()
        provider_priority = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        self.providers = [p for p in provider_priority if p in available_providers]
        print(f"使用推理引擎: {self.providers}")
        sess_opt = ort.SessionOptions()
        sess_opt.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_opt.intra_op_num_threads = 4
        self.session = ort.InferenceSession(
            model_path, sess_options=sess_opt, providers=self.providers
        )
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [o.name for o in self.session.get_outputs()]

    def preprocess(self, img):
        self.orig_h, self.orig_w = img.shape[:2]
        r = min(self.input_size / self.orig_h, self.input_size / self.orig_w)
        new_h, new_w = int(self.orig_h * r), int(self.orig_w * r)
        pad_h = (self.input_size - new_h) // 2
        pad_w = (self.input_size - new_w) // 2
        img_resized = cv2.resize(img, (new_w, new_h))
        img_pad = np.full((self.input_size, self.input_size, 3), 114, dtype=np.uint8)
        img_pad[pad_h:pad_h + new_h, pad_w:pad_w + new_w, :] = img_resized
        blob = img_pad[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) / 255.0
        blob = np.expand_dims(blob, axis=0)
        self.pad_params = (pad_h, pad_w, r, new_h, new_w)
        return blob

    def postprocess(self, outputs):
        det_out = outputs[0][0]       # [4+nc+32, anchors]
        mask_proto = outputs[1][0]    # [32, proto_h, proto_w]
        det_out = det_out.T           # [anchors, 4+nc+32]
        boxes_xywh = det_out[:, :4]
        cls_scores = det_out[:, 4:4 + self.num_classes]
        mask_coeffs = det_out[:, 4 + self.num_classes:]
        # 置信度过滤
        max_scores = np.max(cls_scores, axis=1)
        keep = max_scores > self.conf_thres
        boxes_xywh = boxes_xywh[keep]
        cls_scores = cls_scores[keep]
        mask_coeffs = mask_coeffs[keep]
        max_scores = max_scores[keep]
        class_ids = np.argmax(cls_scores, axis=1)
        if len(boxes_xywh) == 0:
            return [], [], [], []
        boxes_xyxy = self._xywh2xyxy(boxes_xywh)
        boxes_xyxy = self._scale_boxes(boxes_xyxy)
        # 按类别分别做 NMS
        indices = []
        for c in np.unique(class_ids):
            c_mask = class_ids == c
            c_boxes = boxes_xyxy[c_mask]
            c_scores = max_scores[c_mask]
            idxs = cv2.dnn.NMSBoxes(c_boxes.tolist(), c_scores.tolist(),
                                    self.conf_thres, self.iou_thres)
            if len(idxs) > 0:
                idxs = idxs.flatten()
                global_idx = np.where(c_mask)[0][idxs]
                indices.extend(global_idx.tolist())
        if len(indices) == 0:
            return [], [], [], []
        indices = np.array(indices)
        final_boxes = boxes_xyxy[indices]
        final_scores = max_scores[indices]
        final_cls = class_ids[indices]
        final_coeffs = mask_coeffs[indices]
        final_masks = self._generate_masks(mask_proto, final_coeffs)
        return final_boxes, final_scores, final_cls, final_masks

    def _generate_masks(self, protos, coeffs):
        pad_h, pad_w, r, new_h, new_w = self.pad_params
        n = coeffs.shape[0]
        masks = coeffs @ protos.reshape(protos.shape[0], -1)
        masks = masks.reshape(n, protos.shape[1], protos.shape[2])
        masks = 1 / (1 + np.exp(-masks))
        mask_list = []
        for m in masks:
            m640 = cv2.resize(m, (self.input_size, self.input_size))
            mcrop = m640[pad_h:pad_h + new_h, pad_w:pad_w + new_w]
            morig = cv2.resize(mcrop, (self.orig_w, self.orig_h))
            mask_list.append((morig > 0.5).astype(np.uint8))
        return mask_list

    def _xywh2xyxy(self, x):
        y = np.zeros_like(x)
        y[:, 0] = x[:, 0] - x[:, 2] / 2
        y[:, 1] = x[:, 1] - x[:, 3] / 2
        y[:, 2] = x[:, 0] + x[:, 2] / 2
        y[:, 3] = x[:, 1] + x[:, 3] / 2
        return y

    def _scale_boxes(self, boxes):
        pad_h, pad_w, r, _, _ = self.pad_params
        boxes[:, [0, 2]] -= pad_w
        boxes[:, [1, 3]] -= pad_h
        boxes /= r
        return boxes

# ========== 读取图片函数：支持webp/jpg/png ==========
def read_img(img_path):
    t_start = time.time()
    try:
        pil_img = Image.open(img_path).convert("RGB")
        img_np = np.array(pil_img)
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        t_end = time.time()
        read_ms = (t_end - t_start) * 1000
        return img_bgr, read_ms
    except Exception as e:
        print(f"读取失败 {img_path}: {e}")
        return None, 0.0

def draw_results(img, boxes, scores, cls_ids, masks, class_names, alpha=0.25):
    overlay = img.copy()
    for box, score, cls_id, mask in zip(boxes, scores, cls_ids, masks):
        color = COLORS[int(cls_id) % len(COLORS)]
        if alpha > 0:
            color_mask = np.zeros_like(img, dtype=np.uint8)
            color_mask[mask > 0] = color
            overlay = cv2.addWeighted(color_mask, alpha, overlay, 1 - alpha, 0)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1, color, 2)
        x1, y1, x2, y2 = box.astype(int)
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)
        name = class_names[int(cls_id)] if int(cls_id) < len(class_names) else f"cls{int(cls_id)}"
        label = f"{name} {score:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(overlay, (x1, y1 - th - 8), (x1 + tw, y1), color, -1)
        cv2.putText(overlay, label, (x1, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    return overlay

if __name__ == "__main__":
    MODEL_PATH = "/home/sunweining/ore_deploy/best.onnx"
    IMG_DIR = "/home/sunweining/ore_deploy/test_img"
    OUT_DIR = "/home/sunweining/ore_deploy/output"
    CONF_THRES = 0.25
    ALPHA = 0.25
    os.makedirs(OUT_DIR, exist_ok=True)

    model = YOLOv11SegRunner(MODEL_PATH, conf_thres=CONF_THRES)

    img_formats = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')

    # 统计累加器
    total_read = 0.0
    total_prep = 0.0
    total_infer = 0.0
    total_post = 0.0
    img_cnt = 0

    for fn in sorted(os.listdir(IMG_DIR)):
        if not fn.lower().endswith(img_formats):
            continue
        img_path = os.path.join(IMG_DIR, fn)

        # 1 读图计时
        img, read_ms = read_img(img_path)
        if img is None:
            print(f"[{fn}] 读取失败，跳过")
            continue
        img_cnt += 1
        total_read += read_ms

        # 2 预处理计时
        t2 = time.time()
        blob = model.preprocess(img)
        t3 = time.time()
        prep_ms = (t3 - t2) * 1000
        total_prep += prep_ms

        # 3 推理Run计时
        t4 = time.time()
        outputs = model.session.run(model.output_names, {model.input_name: blob})
        t5 = time.time()
        infer_ms = (t5 - t4) * 1000
        total_infer += infer_ms

        # 4 后处理：NMS + mask解码 + 绘图 + 保存
        t6 = time.time()
        boxes, scores, cls_ids, masks = model.postprocess(outputs)
        vis = draw_results(img, boxes, scores, cls_ids, masks, model.class_names, alpha=ALPHA)
        out_name = os.path.splitext(fn)[0] + ".jpg"
        out_file = os.path.join(OUT_DIR, out_name)
        cv2.imwrite(out_file, vis, [cv2.IMWRITE_JPEG_QUALITY, 95])
        t7 = time.time()
        post_ms = (t7 - t6) * 1000
        total_post += post_ms

        obj_num = len(boxes)
        print(f"[{img_cnt}] {fn} | read:{read_ms:.2f} ms | prep:{prep_ms:.2f} ms | infer:{infer_ms:.2f} ms | post:{post_ms:.2f} ms | obj:{obj_num}")

    # 打印数据集平均耗时，用于和C++对比
    if img_cnt > 0:
        print("\n====== 数据集平均耗时（Python‑onnxruntime）======")
        print(f"图片总数量:{img_cnt}")
        print(f"平均读图:    {total_read/img_cnt:.4f} ms")
        print(f"平均预处理:  {total_prep/img_cnt:.4f} ms")
        print(f"平均推理Run: {total_infer/img_cnt:.4f} ms")
        print(f"平均后处理:  {total_post/img_cnt:.4f} ms")
        print(f"平均全套总:  {(total_read+total_prep+total_infer+total_post)/img_cnt:.4f} ms")

