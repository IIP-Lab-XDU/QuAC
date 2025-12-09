import torch.nn as nn
import torch
import torchvision

class QualityAcon(nn.Module):
    """Quality-Adaptive Activation Module.
    
    This module adaptively modulates feature activations based on input quality features.
    
    Args:
        q_channels (int): Number of channels in quality features.
        x_channels (int): Number of channels in input features.
        r (int): Reduction ratio. Default: 16.

    Returns:
        torch.Tensor: Feature map after quality-adaptive activation.
    """
    def __init__(self,
                 q_channels=36,
                 x_channels=None,
                 r=16):
        super().__init__()
        width = q_channels + x_channels
        self.fc1 = nn.Conv2d(width, max(r, width // r), kernel_size=1, stride=1, bias=True)
        self.bn1 = nn.BatchNorm2d(max(r, width // r))
        # self.ln1 = nn.LayerNorm([max(r, width // r), 1, 1]) # if batch size = 1
        self.fc2 = nn.Conv2d(max(r, width // r), width, kernel_size=1, stride=1, bias=True)
        self.bn2 = nn.BatchNorm2d(width)
        # self.ln2 = nn.LayerNorm([width, 1, 1])

        self.p1 = nn.Parameter(torch.randn(1, x_channels, 1, 1))
        self.p2 = nn.Parameter(torch.randn(1, x_channels, 1, 1))

        self.sigmoid = nn.Sigmoid()

        self.fc3 = nn.Conv2d(width, x_channels, kernel_size=1, stride=1)

    def forward(self, x, q):
        # x (torch.Tensor): Input feature map with shape (B, x_channels, H, W).
        # q (torch.Tensor): Quality features with shape (B, q_channels, 1, 1).
        assert q.dim() == 4 and q.shape[2] == 1 and q.shape[3] == 1, \
                f"Expected q shape (B, q_channels, 1, 1), but got {q.shape}"
        x_avg = x.mean(dim=2, keepdims=True).mean(dim=3, keepdims=True)
        z = torch.cat((x_avg, q), dim=1)
        beta = torch.sigmoid(self.bn2(self.fc2(self.bn1(self.fc1(z)))))
        beta = self.fc3(beta)
        x = (self.p1 * x - self.p2 * x) * torch.sigmoid(beta * (self.p1 * x - self.p2 * x)) + self.p2 * x
        return x

if __name__ == "__main__":
    # 随机生成示例输入
    image = torch.randn(8, 512, 64, 64)
    quality = torch.randn(8, 32, 1, 1)

    # 初始化 CONTRIQUE 模型
    activation = QualityAcon(36, 512)

    x = activation(image, quality)

    print("after quality-adaptive activation feature shape:", x.shape)  # 输出: torch.Size([8, 512, 64, 64])