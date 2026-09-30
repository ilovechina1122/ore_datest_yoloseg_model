# -*- coding: utf-8 -*-
"""
VIA JSON -> YOLO 格式转换脚本
用法：
    python convert_via_to_yolo.py
配置（改下面这几行即可）：
    JSON_FILE : VIA 导出的标注 json 文件名
    IMG_DIR   : 原图所在文件夹
    OUT_DIR   : 输出的 labels 文件夹
    MODE      : "det" = 目标检测(外接框)，"seg" = 实例分割(多边形点)
    CLASS_MAP : 类别名 -> id
依赖：
    pip install pillow
"""
import json
import os
from PIL import Image

# ============ 配置区 ============
JSON_FILE = "11111.json"
IMG_DIR = "images"
OUT_DIR = "labels"
MODE = "seg"                # "det" 或 "seg"
CLASS_MAP = {"pyrite": 1}   # 类别 -> id
# ================================

os.makedirs(OUT_DIR, exist_ok=True)

with open(JSON_FILE, "r", encoding="utf-8") as f:
    data = json.load(f)

# 生成 classes.txt（类别顺序与 id 对应）
with open("classes.txt", "w", encoding="utf-8") as fc:
    for name, cid in sorted(CLASS_MAP.items(), key=lambda x: x[1]):
        fc.write(name + "\n")

converted, skipped = 0, 0

for file_key, item in data.items():
    filename = item["filename"]
    img_path = os.path.join(IMG_DIR, filename)
    if not os.path.exists(img_path):
        print(f"跳过: 找不到图片 {filename}")
        skipped += 1
        continue

    # VIA json 里没有宽高，必须读真实图片
    im = Image.open(img_path)
    W, H = im.size

    lines = []
    for r in item["regions"].values():
        label = r["region_attributes"].get("label")
        if label not in CLASS_MAP:
            print(f"跳过未知类别 {label} in {filename}")
            continue
        cid = CLASS_MAP[label]
        xs = r["shape_attributes"]["all_points_x"]
        ys = r["shape_attributes"]["all_points_y"]

        if MODE == "det":
            # 检测：多边形取外接矩形 -> x_center y_center w h
            xmin, xmax = min(xs), max(xs)
            ymin, ymax = min(ys), max(ys)
            xc = ((xmin + xmax) / 2) / W
            yc = ((ymin + ymax) / 2) / H
            bw = (xmax - xmin) / W
            bh = (ymax - ymin) / H
            lines.append(f"{cid} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")
        else:
            # 分割：直接归一化多边形点
            pts = " ".join(f"{x / W:.6f} {y / H:.6f}" for x, y in zip(xs, ys))
            lines.append(f"{cid} {pts}")

    out_txt = os.path.join(OUT_DIR, os.path.splitext(filename)[0] + ".txt")
    with open(out_txt, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    converted += 1

print(f"\n完成: 转换 {converted} 张, 跳过 {skipped} 张")
print(f"标签输出目录: {os.path.abspath(OUT_DIR)}")
