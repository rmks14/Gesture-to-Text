from typing import Optional, Tuple

import torch
from torch import nn
from torchvision import models
from torchvision.models import ResNet18_Weights


class FrameCNN(nn.Module):
    def __init__(self, embedding_dim: int = 256) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.projection = nn.Linear(256, embedding_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = x.flatten(1)
        x = self.projection(x)
        return x


class ResNet18FrameEncoder(nn.Module):
    def __init__(self, embedding_dim: int = 256, pretrained: bool = True) -> None:
        super().__init__()
        weights = ResNet18_Weights.DEFAULT if pretrained else None
        backbone = models.resnet18(weights=weights)
        self.features = nn.Sequential(*list(backbone.children())[:-1])  # (N, 512, 1, 1)
        self.projection = nn.Linear(512, embedding_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = x.flatten(1)
        x = self.projection(x)
        return x


class LandmarkFrameEncoder(nn.Module):
    def __init__(self, input_dim: int, embedding_dim: int = 256, dropout: float = 0.2) -> None:
        super().__init__()
        self.projection = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, embedding_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim, embedding_dim),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.projection(x)


class TemporalConvRefiner(nn.Module):
    """Lightweight depthwise-separable temporal refinement for sequence embeddings."""

    def __init__(self, channels: int, layers: int = 2, kernel_size: int = 5, dropout: float = 0.15) -> None:
        super().__init__()
        if layers <= 0:
            raise ValueError("layers must be > 0 for TemporalConvRefiner")
        if kernel_size <= 1 or kernel_size % 2 == 0:
            raise ValueError("kernel_size must be odd and > 1 for TemporalConvRefiner")

        blocks = []
        padding = kernel_size // 2
        for _ in range(layers):
            block = nn.Sequential(
                nn.Conv1d(channels, channels, kernel_size=kernel_size, padding=padding, groups=channels, bias=False),
                nn.BatchNorm1d(channels),
                nn.GELU(),
                nn.Conv1d(channels, channels, kernel_size=1, bias=False),
                nn.BatchNorm1d(channels),
                nn.GELU(),
                nn.Dropout(dropout),
            )
            blocks.append(block)
        self.blocks = nn.ModuleList(blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C)
        y = x.transpose(1, 2).contiguous()  # (B, C, T)
        for block in self.blocks:
            y = y + block(y)
        return y.transpose(1, 2).contiguous()


class SequenceClassifier(nn.Module):
    def __init__(
        self,
        num_classes: int,
        input_type: str = "rgb",
        input_dim: Optional[int] = None,
        embedding_dim: int = 256,
        hidden_size: int = 256,
        num_layers: int = 2,
        dropout: float = 0.3,
        bidirectional: bool = False,
        encoder_type: str = "auto",
        pretrained_encoder: bool = False,
        use_attention: bool = True,
        temporal_conv: bool = False,
        temporal_conv_layers: int = 2,
        temporal_conv_kernel_size: int = 5,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.input_type = input_type
        self.input_dim = input_dim
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.encoder_type = encoder_type
        self.pretrained_encoder = pretrained_encoder
        self.use_attention = use_attention
        self.temporal_conv_enabled = temporal_conv
        self.temporal_conv_layers = temporal_conv_layers
        self.temporal_conv_kernel_size = temporal_conv_kernel_size

        if input_type == "rgb":
            resolved_encoder_type = "simple_cnn" if encoder_type == "auto" else encoder_type
            if resolved_encoder_type == "simple_cnn":
                self.encoder = FrameCNN(embedding_dim=embedding_dim)
            elif resolved_encoder_type == "resnet18":
                self.encoder = ResNet18FrameEncoder(embedding_dim=embedding_dim, pretrained=pretrained_encoder)
            else:
                raise ValueError(f"Unknown encoder_type for rgb input: {resolved_encoder_type}")
            self.encoder_type = resolved_encoder_type
        elif input_type == "landmarks":
            if input_dim is None or input_dim <= 0:
                raise ValueError("input_dim must be set for landmark input.")
            self.encoder = LandmarkFrameEncoder(
                input_dim=input_dim,
                embedding_dim=embedding_dim,
                dropout=dropout,
            )
            self.encoder_type = "landmark_mlp"
        else:
            raise ValueError(f"Unknown input_type: {input_type}")

        self.temporal_conv = (
            TemporalConvRefiner(
                channels=embedding_dim,
                layers=max(1, int(temporal_conv_layers)),
                kernel_size=int(temporal_conv_kernel_size),
                dropout=min(0.4, max(0.0, dropout * 0.6)),
            )
            if temporal_conv
            else None
        )
        self.temporal = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )
        direction_scale = 2 if bidirectional else 1
        self.attention = nn.Linear(hidden_size * direction_scale, 1) if use_attention else None
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size * direction_scale, num_classes),
        )

    def encode_frames(self, x: torch.Tensor) -> torch.Tensor:
        if self.input_type == "rgb":
            batch_size, seq_len, channels, height, width = x.shape
            x = x.view(batch_size * seq_len, channels, height, width)
            x = self.encoder(x)
            x = x.view(batch_size, seq_len, self.embedding_dim)
            return x
        # Landmark features: x shape (B, T, F)
        batch_size, seq_len, feat_dim = x.shape
        x = x.view(batch_size * seq_len, feat_dim)
        x = self.encoder(x)
        x = x.view(batch_size, seq_len, self.embedding_dim)
        return x

    def _final_hidden(self, h_n: torch.Tensor) -> torch.Tensor:
        if self.bidirectional:
            return torch.cat([h_n[-2], h_n[-1]], dim=1)
        return h_n[-1]

    def _pool_sequence(self, seq_out: torch.Tensor, h_n: torch.Tensor) -> torch.Tensor:
        if self.attention is None:
            return self._final_hidden(h_n)
        scores = self.attention(seq_out).squeeze(-1)
        weights = torch.softmax(scores, dim=1).unsqueeze(-1)
        return torch.sum(seq_out * weights, dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        frame_embeddings = self.encode_frames(x)
        if self.temporal_conv is not None:
            frame_embeddings = self.temporal_conv(frame_embeddings)
        seq_out, (h_n, _) = self.temporal(frame_embeddings)
        pooled = self._pool_sequence(seq_out, h_n)
        logits = self.classifier(pooled)
        return logits

    @torch.no_grad()
    def predict_proba(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        logits = self.forward(x)
        probs = torch.softmax(logits, dim=1)
        confidences, indices = torch.max(probs, dim=1)
        return indices, confidences


class CNNLSTMClassifier(SequenceClassifier):
    """Backward-compatible alias used by existing scripts."""
