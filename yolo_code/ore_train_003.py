from ultralytics import YOLO
import os


def main():
    # 1. 加载预训练分割模型
    # 可选 yolov11n-seg.pt / yolov11s-seg.pt / yolov11m-seg.pt
    # n/s/m/l/x 模型越大精度越高，速度越慢，显存需求越大
    model = YOLO('yolo b11n-seg.pt')

    # 2. 开始训练
    results = model.train(
        # ===== 数据集配置 =====
        data='C:\\Users\\10650\\Desktop\\first-year-work\\dataset001\\quartz_dataset\\ore_data.yaml',  # 你的数据集yaml配置文件路径
        imgsz=640,  # 输入图片分辨率
        batch=16,  # 批次大小，显存不够就改小（8/4）

        # ===== 训练轮次与优化 =====
        epochs=100,  # 总训练轮数
        patience=20,  # 20轮验证集mAP不提升就自动早停，防止过拟合
        optimizer='AdamW',  # 优化器
        lr0=0.001,  # 初始学习率
        weight_decay=0.0005,  # 权重衰减

        # ===== 可视化与保存 =====
        project='ore_seg_project',  # 项目根文件夹
        name='yolov11n_ore_seg',  # 本次实验名称，所有结果存在这个文件夹里
        exist_ok=True,  # 覆盖之前同名的实验结果
        plots=True,  # 训练结束自动绘制所有曲线图（默认开启）
        save=True,  # 保存模型权重
        save_period=10,  # 每10轮保存一次中间模型，方便看不同阶段效果
        val=True,  # 每轮训练后跑验证集，计算精度
        device=0,  # GPU编号，单卡写0；没有GPU就写 'cpu'

        # ===== 数据增强（矿石分割建议开启） =====
       # augment=True,
        #mosaic=1.0,  # 马赛克增强
        #mixup=0.1,  # 混合增强
        #flipud=0.5,  # 上下翻转概率
        #fliplr=0.5,  # 左右翻转概率
    )

    # 3. 训练结束后，打印最佳精度
    print("\n===== 训练完成 =====")
    print(f"最佳验证集 mAP50(检测框): {results.results_dict['metrics/mAP50(B)']:.4f}")
    print(f"最佳验证集 mAP50(分割掩码): {results.results_dict['metrics/mAP50(M)']:.4f}")
    print(f"所有结果保存在: {os.path.abspath('ore_seg_project/yolov11n_ore_seg')}")


if __name__ == '__main__':
    main()
