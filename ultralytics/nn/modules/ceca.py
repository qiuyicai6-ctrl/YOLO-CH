import torch
import torch.nn as nn

class CECA(nn.Module):
    """Coordinate and Channel-wise Efficient Attention (CECA) Module"""
    def __init__(self, c1, c2, reduction=16):
        super().__init__()
        assert c1 == c2, f"CECA requires c1 == c2, but got c1={c1}, c2={c2}"
        self.c1 = c1
        self.c2 = c2
        self.reduction = reduction

        # 1. 通道注意力（Channel Attention）
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(c1, c1 // reduction, kernel_size=1, bias=False),
            nn.ReLU(),
            nn.Conv2d(c1 // reduction, c1, kernel_size=1, bias=False),
            nn.Sigmoid()
        )

        # 2. 坐标注意力（Coordinate Attention）- 修复维度匹配问题
        self.conv1 = nn.Conv2d(c1, c1 // reduction, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(c1 // reduction)
        self.act = nn.ReLU()
        self.conv_x = nn.Conv2d(c1 // reduction, c1, kernel_size=1, stride=1, padding=0)
        self.conv_y = nn.Conv2d(c1 // reduction, c1, kernel_size=1, stride=1, padding=0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 通道注意力（无维度问题）
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        ca = self.sigmoid(avg_out + max_out)
        x = x * ca

        # 坐标注意力（修复维度匹配）
        b, c, h, w = x.size()
        # 沿x方向池化（保留宽度，压缩高度为1）
        x_avg = torch.mean(x, dim=2, keepdim=True)  # shape: (b, c, 1, w)
        # 沿y方向池化（保留高度，压缩宽度为1）
        y_avg = torch.mean(x, dim=3, keepdim=True)  # shape: (b, c, h, 1)
        # 调整y_avg维度为 (b, c, 1, h)，确保拼接后维度一致
        y_avg = y_avg.permute(0, 1, 3, 2)  # shape: (b, c, 1, h)
        # 拼接维度改为dim=3（宽度维度），确保尺寸匹配
        concat = torch.cat([x_avg, y_avg], dim=3)  # shape: (b, c, 1, w+h)
        concat = self.act(self.bn1(self.conv1(concat)))
        # 分离x和y方向注意力（按宽度维度拆分）
        x_att, y_att = torch.split(concat, [w, h], dim=3)
        # 调整y_att维度回 (b, c, h, 1) 并广播
        y_att = y_att.permute(0, 1, 3, 2)  # shape: (b, c, h, 1)
        # 生成注意力权重并应用
        x_att = self.sigmoid(self.conv_x(x_att))  # (b, c, 1, w) → 广播为 (b, c, h, w)
        y_att = self.sigmoid(self.conv_y(y_att))  # (b, c, h, 1) → 广播为 (b, c, h, w)
        x = x * x_att * y_att

        return x