"""Small sequential CNN for HW1.

Architecture (input 3 x S x S, S multiple of 16, 100 classes, FP32, eval mode):

    Conv7x7 s2 3->32   -> ReLU          (res S/2)
    MaxPool 3x3 s2 p1                   (res S/4)
    Conv5x5 32->64     -> ReLU          (res S/4)
    Conv3x3 s2 64->128 -> ReLU          (res S/8)
    Conv1x1 128->256   -> ReLU          (res S/8)
    Conv3x3 s2 256->256 -> ReLU         (res S/16)
    Conv1x1 256->512   -> ReLU          (res S/16)
    GlobalAvgPool
    Linear 512->256 -> ReLU
    Linear 256->100

Conventions: every conv uses padding = k // 2 and bias=False; ReLU is inplace.
"""

import torch
import torch.nn as nn


class SmallCNN(nn.Module):
    def __init__(self, num_classes: int = 100) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=7, stride=2, padding=3, bias=False),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=1, stride=1, padding=0, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 512, kernel_size=1, stride=1, padding=0, bias=False),
            nn.ReLU(inplace=True),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x).flatten(1)
        return self.classifier(x)


def build_model(num_classes: int = 100) -> SmallCNN:
    """Model on the default device, in eval() mode, with the HW1 backend flags set."""
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    model = SmallCNN(num_classes).to("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    return model


if __name__ == "__main__":
    model = build_model()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"parameters (incl. linear biases): {n_params}")
    for s in (32, 64, 128, 224, 256, 384, 512):
        x = torch.randn(2, 3, s, s, device=next(model.parameters()).device)
        with torch.inference_mode():
            y = model(x)
        assert y.shape == (2, 100), y.shape
        print(f"S={s:4d} -> output {tuple(y.shape)} ok")
