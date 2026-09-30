import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

# 下面再写 from ultralytics import YOLO ...


from ultralytics import YOLO
import matplotlib.pyplot as plt
import numpy as np

# ========== 字体配置 ==========
import matplotlib
# 不用SimHei中文，直接用英文绘图，彻底规避中文字体损坏问题
matplotlib.use('Agg')

if __name__ == '__main__':
    # -------------------------- 配置区域 --------------------------
    model_path = r"C:\Users\10650\Desktop\first-year-work\yolov11\ultralytics\runs\segment\ore_seg_project\yolov11n_ore_seg\weights\best.pt"
    data_yaml = r"C:\\Users\\10650\\Desktop\\first-year-work\\dataset001\\quartz_dataset\\ore_data.yaml"
    # -------------------------------------------------------------

    model = YOLO(model_path)
    # 重点：plots=False 关闭YOLO内置绘图，防止内部字体报错！
    metrics = model.val(data=data_yaml, split="test", batch=4, plots=False)

    # 打印指标
    print("="*50)
    print("【Box 边界框指标】")
    print(f"box mAP@0.5:   {metrics.box.map50:.4f}")
    print(f"box mAP@0.5:0.95: {metrics.box.map:.4f}")
    print(f"box Precision: {metrics.box.p[0]:.4f}")
    print(f"box Recall:    {metrics.box.r[0]:.4f}")

    print("\n【Mask 分割指标】")
    print(f"mask mAP@0.5:   {metrics.seg.map50:.4f}")
    print(f"mask mAP@0.5:0.95: {metrics.seg.map:.4f}")
    print(f"mask Precision: {metrics.seg.p[0]:.4f}")
    print(f"mask Recall:    {metrics.seg.r[0]:.4f}")

    # 逐类别指标
    class_names = metrics.names
    print("\n===== 每一类分割AP@0.5 =====")
    for cls_id, cls_name in class_names.items():
        ap50 = metrics.seg.ap50[cls_id]
        print(f"{cls_name:<12} AP@0.5 = {ap50:.4f}")

    # ========== 自己单独绘图（英文标题，避免字体问题） ==========
    # ========== 手动填入本次测试得到的指标，绘图（稳定不会出错） ==========
    try:
        # 从你刚才控制台输出复制来的真实数据
        cls_list = ["quartz", "pyrite", "malachite", "bornite", "basalt"]
        ap50_list = [0.8439, 0.9314, 0.9412, 0.9950, 0.9255]

        plt.figure(figsize=(8, 4))
        bars = plt.bar(cls_list, ap50_list, color="#4472C4")
        plt.ylim(0, 1.05)
        plt.title("Mask AP@0.5 for each ore class")
        plt.ylabel("AP@0.5")
        plt.xlabel("Ore Category")
        plt.grid(axis="y", alpha=0.3)
        # 在柱子顶部标注数值，论文图更好看
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width() / 2., height + 0.01,
                     f'{height:.3f}', ha="center")
        plt.tight_layout()
        plt.savefig(r"C:\Users\10650\Desktop\class_ap_plot.png", dpi=300)
        plt.close()
        print("\n✅ 柱状图已保存到桌面 class_ap_plot.png")
    except Exception as e:
        print(f"\n⚠️  Custom plot failed: {e}")
