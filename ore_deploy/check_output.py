def read_img(img_path):
    try:
        print(f"正在读取: {img_path}")   # 加这行
        pil_img = Image.open(img_path).convert("RGB")
        img_np = np.array(pil_img)
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        print(f"  -> 成功, 尺寸: {img_bgr.shape}")   # 加这行
        return img_bgr
    except Exception as e:
        print(f"  -> 读取失败: {e}")
        return None

