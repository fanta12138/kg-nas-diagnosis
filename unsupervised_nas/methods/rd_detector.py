"""
RD-KG-NAS 检测器：Reverse Distillation 风格的教师-学生差异检测

教师（冻结）: ImageNet 预训练 ResNet-18，提取 layer1/layer2/layer3 多尺度特征
学生: NAS 搜索的 encoder + 1x1 bottleneck + 反卷积 decoder
      decoder 逐步重建教师的 layer3 → layer2 → layer1 特征
训练: MSE 多尺度重建损失（仅正常样本）
推理: 各层教师-学生特征差异上采样求和 → 异常分数

参考: "Anomaly Detection via Reverse Distillation from One-Class Embedding" (CVPR 2022)
"""
import os

os.environ.setdefault('HF_HUB_OFFLINE', '1')

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from torchvision import models
from typing import Dict, List, Tuple


class TeacherFeatureExtractor(nn.Module):
    """冻结的 ResNet 教师，输出 layer1/2/3 特征"""

    def __init__(self, arch: str = 'resnet18'):
        super().__init__()
        if arch == 'resnet18':
            self.model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
            self.dims = (64, 128, 256)
        elif arch == 'resnet34':
            self.model = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)
            self.dims = (64, 128, 256)
        elif arch == 'resnet50':
            self.model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
            self.dims = (256, 512, 1024)
        elif arch == 'wide_resnet50_2':
            self.model = models.wide_resnet50_2(weights=models.Wide_ResNet50_2_Weights.IMAGENET1K_V1)
            self.dims = (256, 512, 1024)
        else:
            raise ValueError(f'Unsupported teacher: {arch}')
        for p in self.model.parameters():
            p.requires_grad = False
        self.model.eval()

        self.features: Dict[str, torch.Tensor] = {}
        self.handles = []

        def hook(module, inp, out, name):
            self.features[name] = out

        self.handles.append(self.model.layer1.register_forward_hook(lambda m, i, o: hook(m, i, o, 'l1')))
        self.handles.append(self.model.layer2.register_forward_hook(lambda m, i, o: hook(m, i, o, 'l2')))
        self.handles.append(self.model.layer3.register_forward_hook(lambda m, i, o: hook(m, i, o, 'l3')))

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        self.model(x)
        return self.features['l1'], self.features['l2'], self.features['l3']


class StudentEncoder(nn.Module):
    """NAS 搜索的学生 encoder：任意骨干 → 14x14 特征图 → bottleneck 256d"""

    def __init__(self, backbone_name: str, bottleneck_dim: int = 256, pretrained: bool = True):
        super().__init__()
        self.backbone_name = backbone_name
        try:
            # CNN 类：features_only 返回各 stage 特征图
            self.model = timm.create_model(backbone_name, pretrained=pretrained,
                                           features_only=True)
            self.mode = 'features'
        except Exception:
            # ViT 类：返回 token/patch tensor
            self.model = timm.create_model(backbone_name, pretrained=pretrained,
                                           num_classes=0, global_pool='')
            self.mode = 'flat'

        # 输入通道数在第一次 forward 时推断
        self.in_channels = None
        self.bottleneck = None

    def _init_bottleneck(self, in_c: int, bottleneck_dim: int, device):
        self.in_channels = in_c
        self.bottleneck = nn.Sequential(
            nn.Conv2d(in_c, bottleneck_dim, kernel_size=1),
            nn.BatchNorm2d(bottleneck_dim),
            nn.ReLU(inplace=True),
        ).to(device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.mode == 'features':
            f = self.model(x)[-1]
        else:
            f = self.model(x)
            if f.dim() == 3:  # [B, N, D] ViT tokens
                B, N, D = f.shape
                h = w = int(N ** 0.5)
                # 去掉 cls token（若存在），reshape 为特征图
                if N == h * w + 1:
                    f = f[:, 1:, :]
                    N = h * w
                f = f.transpose(1, 2).reshape(B, D, h, w)
        if f.dim() != 4:
            raise RuntimeError(f'Unexpected student feature shape: {f.shape}')
        f = F.adaptive_avg_pool2d(f, (14, 14))
        if self.bottleneck is None:
            self._init_bottleneck(f.shape[1], 256, f.device)
        return self.bottleneck(f)


class Decoder(nn.Module):
    """从 14x14 bottleneck 逐步重建教师 layer3(14x14) → layer2(28x28) → layer1(56x56)
    通道数根据教师自适应：resnet18/34=(64,128,256)，resnet50/wrn=(256,512,1024)
    """

    def __init__(self, bottleneck_dim: int = 256, teacher_dims=(64, 128, 256)):
        super().__init__()
        t1_dim, t2_dim, t3_dim = teacher_dims
        self.dec3 = nn.Sequential(
            nn.Conv2d(bottleneck_dim, t3_dim, 3, padding=1),
            nn.BatchNorm2d(t3_dim), nn.ReLU(inplace=True),
            nn.Conv2d(t3_dim, t3_dim, 3, padding=1),
        )
        self.dec2 = nn.Sequential(
            nn.Conv2d(t3_dim, t2_dim, 3, padding=1),
            nn.BatchNorm2d(t2_dim), nn.ReLU(inplace=True),
            nn.Conv2d(t2_dim, t2_dim, 3, padding=1),
        )
        self.dec1 = nn.Sequential(
            nn.Conv2d(t2_dim, t1_dim, 3, padding=1),
            nn.BatchNorm2d(t1_dim), nn.ReLU(inplace=True),
            nn.Conv2d(t1_dim, t1_dim, 3, padding=1),
        )

    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        s3 = self.dec3(z)                                   # 14x14
        s2 = self.dec2(F.interpolate(s3, scale_factor=2))   # 28x28
        s1 = self.dec1(F.interpolate(s2, scale_factor=2))   # 56x56
        return s1, s2, s3


class RDDetector(nn.Module):
    """RD-KG-NAS 完整检测器"""

    def __init__(self, student_backbone: str, teacher_arch: str = 'resnet18',
                 bottleneck_dim: int = 256, pretrained: bool = True):
        super().__init__()
        self.teacher = TeacherFeatureExtractor(teacher_arch)
        self.encoder = StudentEncoder(student_backbone, bottleneck_dim, pretrained)
        self.decoder = Decoder(bottleneck_dim, teacher_dims=self.teacher.dims)

    def forward_student(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        z = self.encoder(x)
        return self.decoder(z)

    def training_loss(self, x: torch.Tensor) -> torch.Tensor:
        """多尺度重建损失（只用正常样本训练）"""
        with torch.no_grad():
            t1, t2, t3 = self.teacher(x)
        s1, s2, s3 = self.forward_student(x)
        loss = F.mse_loss(s3, t3) + F.mse_loss(s2, t2) + F.mse_loss(s1, t1)
        return loss

    @torch.no_grad()
    def anomaly_score(self, x: torch.Tensor, img_size: int = 224) -> torch.Tensor:
        """图像级异常分数 [B]：各层差异上采样到统一分辨率求和"""
        t1, t2, t3 = self.teacher(x)
        s1, s2, s3 = self.forward_student(x)

        def diff_map(s, t):
            d = torch.norm(s - t, dim=1, keepdim=True)
            return F.interpolate(d, size=(img_size, img_size), mode='bilinear', align_corners=False)

        m1 = diff_map(s1, t1)
        m2 = diff_map(s2, t2)
        m3 = diff_map(s3, t3)
        anomaly_map = (m1 + m2 + m3).squeeze(1)          # [B, 224, 224]
        return anomaly_map.reshape(x.shape[0], -1).mean(dim=1)  # [B] 全局均值

    @torch.no_grad()
    def prototype_score(self, x: torch.Tensor, prototype: torch.Tensor) -> torch.Tensor:
        """原型距离分数 [B]：bottleneck 特征全局池化后与正常原型的距离"""
        z = self.encoder(x)                      # [B, 256, 14, 14]
        z_pool = z.reshape(z.shape[0], z.shape[1], -1).mean(dim=2)  # [B, 256]
        return torch.norm(z_pool - prototype, dim=1)
