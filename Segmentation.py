import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
import sys
import os

sys.path.append("..")
from segment_anything import sam_model_registry, SamPredictor

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)

def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)

sam_checkpoint = "./models/sam_vit_b_01ec64.pth"
model_type = "vit_b"
device = "cuda"  # or "cuda"

sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)
predictor = SamPredictor(sam)

# 文件夹路径
folder_path = 'C:\\Users\\c8703\\Desktop\\data\\R'

# 输出文件夹路径
output_folder_path = 'C:\\Users\\c8703\\Desktop\\test\\masked_R'

# 确保输出文件夹存在
if not os.path.exists(output_folder_path):
    os.makedirs(output_folder_path)

# 遍历文件夹内的所有文件
for filename in os.listdir(folder_path):
    if filename.endswith(".jpg"): 
        # 读取图像
        image_path = os.path.join(folder_path, filename)
        image = cv2.imread(image_path)

        # 创建窗口并绑定回调函数
        points = []
        def mouse_callback(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                points.append((x, y))
                print(f"Clicked at coordinates: ({x}, {y})")

        cv2.namedWindow('Image')
        cv2.setMouseCallback('Image', mouse_callback)

        # 显示图像
        cv2.imshow('Image', image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        # 将选中的点转换为 NumPy 数组
        points_array = np.array(points)

        # 单点 prompt 输入格式为 (x, y) 并表示出点所带有的标签 1 (前景点)或 0 (背景点)。
        input_point = points_array  # 标记点
        input_label = np.ones(input_point.shape[0])  # 点所对应的标签（全部为 1）

        # 设置图像
        predictor.set_image(image)

        # 执行掩码预测
        masks, scores, logits = predictor.predict(
            point_coords=input_point,
            point_labels=input_label,
            multimask_output=True,
        )

        # 获取最佳掩码
        best_mask = masks[np.argmax(scores)]
        h, w= best_mask.shape[-2:]

        # 创建黑色背景图像
        black_image = np.zeros((h, w, 3), dtype=np.uint8)

        # 将掩码部分复制到黑色背景图像上
        masked_image = black_image.copy()
        masked_image[best_mask] = image[best_mask]

        # 找到掩码的边界框
        y_indices, x_indices = np.where(best_mask)
        x_min, x_max = np.min(x_indices), np.max(x_indices)
        y_min, y_max = np.min(y_indices), np.max(y_indices)

        # 使用边界框裁剪图像
        cropped_image = masked_image[y_min:y_max, x_min:x_max]

        # 保存裁剪后的图像
        output_path = os.path.join(output_folder_path, filename)
        cv2.imwrite(output_path, cropped_image)
