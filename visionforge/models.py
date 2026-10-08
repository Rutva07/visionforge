"""Compact CIFAR-scale residual classifier and conditional DDPM epsilon predictor."""
import math
import torch
from torch import nn
from torch.nn import functional as F


class Residual(nn.Module):
    def __init__(self, cin, cout, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(cin, cout, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(cout)
        self.conv2 = nn.Conv2d(cout, cout, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(cout)
        self.shortcut = (nn.Identity() if cin == cout and stride == 1 else
                         nn.Sequential(nn.Conv2d(cin, cout, 1, stride, bias=False), nn.BatchNorm2d(cout)))

    def forward(self, x):
        return F.relu(self.bn2(self.conv2(F.relu(self.bn1(self.conv1(x))))) + self.shortcut(x))


class Classifier(nn.Module):
    def __init__(self, in_channels=3, classes=100, width=32):
        super().__init__()
        w = width
        self.stem = nn.Sequential(nn.Conv2d(in_channels, w, 3, 1, 1, bias=False),
                                  nn.BatchNorm2d(w), nn.ReLU(inplace=True))
        self.blocks = nn.Sequential(Residual(w, w), Residual(w, w * 2, 2),
                                    Residual(w * 2, w * 2), Residual(w * 2, w * 4, 2),
                                    Residual(w * 4, w * 4))
        self.head = nn.Linear(w * 4, classes)

    def forward(self, x):
        return self.head(self.blocks(self.stem(x)).mean(dim=(2, 3)))


class TimeBlock(nn.Module):
    def __init__(self, ci, co, emb_dim):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, ci)
        self.conv1 = nn.Conv2d(ci, co, 3, padding=1)
        self.emb = nn.Linear(emb_dim, co)
        self.norm2 = nn.GroupNorm(8, co)
        self.conv2 = nn.Conv2d(co, co, 3, padding=1)
        self.skip = nn.Conv2d(ci, co, 1) if ci != co else nn.Identity()

    def forward(self, x, emb):
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.emb(F.silu(emb))[:, :, None, None]
        return self.conv2(F.silu(self.norm2(h))) + self.skip(x)


class ConditionalUNet(nn.Module):
    """32x32 class-conditional U-Net with timestep embeddings and skip connections."""
    def __init__(self, channels=3, classes=100, width=32):
        super().__init__()
        self.width = width
        self.emb_dim = width * 4
        self.class_emb = nn.Embedding(classes, self.emb_dim)
        self.time_mlp = nn.Sequential(nn.Linear(width, self.emb_dim), nn.SiLU(), nn.Linear(self.emb_dim, self.emb_dim))
        self.input = nn.Conv2d(channels, width, 3, padding=1)
        self.down1 = TimeBlock(width, width, self.emb_dim)
        self.down2 = TimeBlock(width, width * 2, self.emb_dim)
        self.down3 = TimeBlock(width * 2, width * 4, self.emb_dim)
        self.mid = TimeBlock(width * 4, width * 4, self.emb_dim)
        self.up2 = TimeBlock(width * 6, width * 2, self.emb_dim)
        self.up1 = TimeBlock(width * 3, width, self.emb_dim)
        self.output = nn.Sequential(nn.GroupNorm(8, width), nn.SiLU(), nn.Conv2d(width, channels, 3, padding=1))

    def forward(self, x, t, labels):
        half = self.width // 2
        frequencies = torch.exp(-math.log(10000) * torch.arange(half, device=t.device, dtype=torch.float32) / max(half - 1, 1))
        phase = t.float()[:, None] * frequencies[None, :]
        positional = torch.cat((torch.sin(phase), torch.cos(phase)), dim=1)
        emb = self.time_mlp(positional) + self.class_emb(labels)
        h0 = self.input(x)
        h1 = self.down1(h0, emb)
        h2 = self.down2(F.avg_pool2d(h1, 2), emb)
        h3 = self.down3(F.avg_pool2d(h2, 2), emb)
        mid = self.mid(h3, emb)
        u2 = self.up2(torch.cat([F.interpolate(mid, size=h2.shape[-2:], mode="nearest"), h2], dim=1), emb)
        u1 = self.up1(torch.cat([F.interpolate(u2, size=h1.shape[-2:], mode="nearest"), h1], dim=1), emb)
        return self.output(u1)


class Diffusion:
    def __init__(self, steps=200, device="cpu"):
        self.steps = steps
        self.betas = torch.linspace(1e-4, 0.02, steps, device=device)
        self.alphas = 1. - self.betas
        self.abar = torch.cumprod(self.alphas, dim=0)

    def loss(self, model, clean, labels):
        t = torch.randint(0, self.steps, (clean.shape[0],), device=clean.device)
        noise = torch.randn_like(clean)
        at = self.abar[t][:, None, None, None]
        noisy = at.sqrt() * clean + (1 - at).sqrt() * noise
        return F.mse_loss(model(noisy, t, labels), noise)

    @torch.no_grad()
    def sample(self, model, labels, channels, side=32):
        x = torch.randn(len(labels), channels, side, side, device=labels.device)
        model.eval()
        for step in reversed(range(self.steps)):
            t = torch.full((len(labels),), step, dtype=torch.long, device=labels.device)
            predicted = model(x, t, labels)
            mean = (x - self.betas[step] / (1 - self.abar[step]).sqrt() * predicted) / self.alphas[step].sqrt()
            if step:
                abar_prev = self.abar[step - 1]
                var = self.betas[step] * (1 - abar_prev) / (1 - self.abar[step])
                x = mean + var.sqrt() * torch.randn_like(x)
            else:
                x = mean
        return x.clamp(-1, 1)
