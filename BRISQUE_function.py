import cv2
import torch
import numpy as np
from torch import Tensor

def extract_brisque_features(inputs: Tensor):
    """
    Extract BRISQUE features from a batch of images using OpenCV.

    Args:
        inputs (Tensor): Batch of images, shape (B, C, H, W), values in [0, 1].

    Returns:
        Tensor: BRISQUE feature vectors, shape (B, 36).
    """
    brisque_features_batch = []
    for img in inputs:
        # 将图像从 Tensor 转换为 NumPy，并调整为 BGR 格式
        img_np = img.permute(1, 2, 0).cpu().numpy() * 255  # 从 [0, 1] 转换到 [0, 255]
        img_np = img_np.astype(np.uint8)
        img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

        # 计算 BRISQUE 特征
        features = cv2.quality.QualityBRISQUE_computeFeatures(img_np)
        features = features.flatten()  # 将 [1, 36] 变为 [36]
        if np.isnan(features).any() or np.isinf(features).any():
            features = np.where(np.isfinite(features), features, 0.0)
        brisque_features_batch.append(features)
    
    # 将特征转换为张量
    return torch.tensor(brisque_features_batch)

if __name__ == "__main__":
    # 随机生成示例输入
    inputs = torch.randn(8, 3, 512, 512)

    # 提取 BRISQUE 特征
    q = extract_brisque_features(inputs)

    q = q.unsqueeze(2).unsqueeze(3)

    print("BRISQUE feature shape:", q.shape)  # 输出: torch.Size([8, 36, 1, 1])