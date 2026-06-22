from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class LightweightCNNBackbone(nn.Module):
    """Small fallback backbone used when InternImage/timm backbones are unavailable."""

    def __init__(self, in_channels: int = 3):
        super().__init__()
        self.stem = ConvBlock(in_channels, 64)
        self.stage2 = ConvBlock(64, 128)
        self.stage3 = ConvBlock(128, 256)
        self.stage4 = ConvBlock(256, 512)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        f1 = self.stem(x)
        f2 = self.stage2(self.pool(f1))
        f3 = self.stage3(self.pool(f2))
        f4 = self.stage4(self.pool(f3))
        return [f1, f2, f3, f4]


class MultiScaleTransformerDecoderLayer(nn.Module):
    """Self-attention over compact image tokens, then road queries cross-attend to them."""

    def __init__(self, d_model: int = 256, nhead: int = 8, dropout: float = 0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query: torch.Tensor, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        feature_context, _ = self.self_attn(features, features, features)
        features = self.norm1(features + self.dropout(feature_context))

        query_context, attn_weights = self.cross_attn(query, features, features)
        query = self.norm2(query + self.dropout(query_context))
        query = self.norm3(query + self.ffn(query))
        return query, attn_weights


class TransformerDecoder(nn.Module):
    """Small Mask2Former-style query decoder for road-context queries."""

    def __init__(self, d_model: int = 256, nhead: int = 8, num_layers: int = 3, dropout: float = 0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            MultiScaleTransformerDecoderLayer(d_model, nhead, dropout)
            for _ in range(num_layers)
        ])

    def forward(self, query: torch.Tensor, features: torch.Tensor) -> list[torch.Tensor]:
        queries = []
        for layer in self.layers:
            query, _ = layer(query, features)
            queries.append(query)
        return queries


class TransformerFPN(nn.Module):
    """
    Projects multi-scale features to uniform channels.

    The transformer receives pooled context tokens for memory safety.
    The prediction head receives fine spatial tokens for detailed masks.
    """

    def __init__(self, in_channels: list[int], d_model: int = 256, context_sizes: tuple[int, ...] = (32, 16, 8, 4)):
        super().__init__()
        self.context_sizes = context_sizes
        self.projections = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(channels, d_model, kernel_size=1, bias=False),
                nn.GroupNorm(32, d_model),
                nn.GELU(),
            )
            for channels in in_channels
        ])

    def forward(self, features: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, tuple[int, int]]:
        projected_features = [projection(feature) for projection, feature in zip(self.projections, features)]
        fine_feature = projected_features[0]
        _, _, fine_h, fine_w = fine_feature.shape
        fine_tokens = fine_feature.flatten(2).transpose(1, 2)

        context_tokens = []
        for index, feature in enumerate(projected_features):
            pooled_size = self.context_sizes[min(index, len(self.context_sizes) - 1)]
            pooled = F.adaptive_avg_pool2d(feature, output_size=(pooled_size, pooled_size))
            context_tokens.append(pooled.flatten(2).transpose(1, 2))

        return torch.cat(context_tokens, dim=1), fine_tokens, fine_feature, (fine_h, fine_w)


class RoadPredictionHead(nn.Module):
    """Turns road query states into a dense spatial logit map."""

    def __init__(self, d_model: int = 256, num_queries: int = 4):
        super().__init__()
        self.query_proj = nn.Linear(d_model, d_model)
        self.pixel_proj = nn.Linear(d_model, d_model)
        self.score_head = nn.Linear(num_queries, 1)

    def forward(self, queries: torch.Tensor, pixel_tokens: torch.Tensor, spatial_shape: tuple[int, int]) -> torch.Tensor:
        batch_size, token_count, channels = pixel_tokens.shape
        height, width = spatial_shape
        query_features = self.query_proj(queries)
        pixel_features = self.pixel_proj(pixel_tokens)
        attention_maps = torch.einsum("bqc,bnc->bnq", query_features, pixel_features) / (channels ** 0.5)
        logits = self.score_head(attention_maps)
        return logits.transpose(1, 2).reshape(batch_size, 1, height, width)


def _ensure_nchw(feature: torch.Tensor, expected_channels: int) -> torch.Tensor:
    if feature.ndim != 4:
        raise ValueError(f"Expected 4D feature map, got shape {tuple(feature.shape)}")
    if feature.shape[1] == expected_channels:
        return feature
    if feature.shape[-1] == expected_channels:
        return feature.permute(0, 3, 1, 2).contiguous()
    return feature


def build_internimage_backbone(backbone_name: str = "internimage_tiny"):
    """
    Tries official InternImage first, then timm fallbacks, then a small local CNN.
    """
    try:
        from mmdet_custom.models.backbones.intern_image import InternImage

        channel_map = {
            "internimage_tiny": [64, 128, 256, 512],
            "internimage_small": [80, 160, 320, 640],
            "internimage_base": [112, 224, 448, 896],
        }
        channels = channel_map.get(backbone_name, channel_map["internimage_tiny"])
        model = InternImage(
            core_op="DCNv3",
            channels=channels[0],
            depths=[4, 4, 18, 4] if "tiny" in backbone_name else [4, 4, 21, 4],
            groups=[4, 8, 16, 32],
            mlp_ratio=4.0,
            drop_path_rate=0.2,
            out_indices=(0, 1, 2, 3),
            with_cp=False,
        )
        print(f"Using official InternImage backbone: {backbone_name}")
        return model, channels
    except ImportError:
        pass

    try:
        import timm
    except ImportError:
        print("timm is not installed. Using lightweight CNN fallback backbone.")
        return LightweightCNNBackbone(), [64, 128, 256, 512]

    candidates = []
    if backbone_name == "internimage_tiny":
        candidates.extend(["internimage_t_1k_224", "internimage_tiny"])
    else:
        candidates.append(backbone_name)
    candidates.extend(["convnext_tiny", "resnet34", "swin_tiny_patch4_window7_224"])

    for candidate in candidates:
        try:
            model = timm.create_model(candidate, pretrained=True, features_only=True, out_indices=(0, 1, 2, 3))
            channels = model.feature_info.channels()
            print(f"Using custom hybrid backbone: {candidate}")
            return model, channels
        except Exception:
            continue

    print("No timm backbone could be created. Using lightweight CNN fallback backbone.")
    return LightweightCNNBackbone(), [64, 128, 256, 512]


class EdgeAwareHybridRoadFormer(nn.Module):
    """
    InternImage-inspired backbone + query transformer decoder + edge-aware head.

    This is a practical Mask2Former-style binary road model:
    multi-scale backbone features -> compact context tokens -> road queries -> dense mask logits.
    """

    def __init__(
        self,
        backbone_name: str = "internimage_tiny",
        d_model: int = 256,
        nhead: int = 8,
        num_decoder_layers: int = 3,
        num_queries: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.backbone, self.in_channels = build_internimage_backbone(backbone_name)
        self.neck = TransformerFPN(self.in_channels, d_model=d_model)
        self.road_queries = nn.Parameter(torch.randn(1, num_queries, d_model) * 0.02)
        self.decoder = TransformerDecoder(d_model=d_model, nhead=nhead, num_layers=num_decoder_layers, dropout=dropout)
        self.seg_head = RoadPredictionHead(d_model=d_model, num_queries=num_queries)
        self.edge_head = RoadPredictionHead(d_model=d_model, num_queries=num_queries)
        self.refine_input = nn.Sequential(
            nn.Conv2d(d_model + 2, d_model // 2, kernel_size=1, bias=False),
            nn.BatchNorm2d(d_model // 2),
            nn.GELU(),
        )
        self.continuity_refine = nn.Sequential(
            ConvBlock(d_model // 2, d_model // 2),
            nn.Conv2d(d_model // 2, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.GELU(),
        )
        self.seg_refine = nn.Conv2d(32, 1, kernel_size=1)
        self.edge_refine = nn.Conv2d(32, 1, kernel_size=1)
        self.centerline_refine = nn.Conv2d(32, 1, kernel_size=1)
        self.refine_gate = nn.Parameter(torch.tensor(0.5))
        self.edge_gate = nn.Parameter(torch.tensor(0.5))
        self.centerline_gate = nn.Parameter(torch.tensor(0.25))
        self.connectivity_smooth = nn.Sequential(
            nn.Conv2d(1, 1, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm2d(1),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        for module in [
            self.neck,
            self.decoder,
            self.seg_head,
            self.edge_head,
            self.refine_input,
            self.continuity_refine,
            self.seg_refine,
            self.edge_refine,
            self.centerline_refine,
        ]:
            for parameter in module.parameters():
                if parameter.dim() > 1:
                    nn.init.xavier_uniform_(parameter)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        batch_size, _, height, width = x.shape
        features = self.backbone(x)
        features = [
            _ensure_nchw(feature, expected_channels)
            for feature, expected_channels in zip(features, self.in_channels)
        ]

        context_tokens, fine_tokens, fine_feature, fine_shape = self.neck(features)
        queries = self.road_queries.expand(batch_size, -1, -1)
        final_queries = self.decoder(queries, context_tokens)[-1]

        seg_logits = self.seg_head(final_queries, fine_tokens, fine_shape)
        edge_logits = self.edge_head(final_queries, fine_tokens, fine_shape)

        refine_features = torch.cat([fine_feature, seg_logits, edge_logits], dim=1)
        refine_features = self.refine_input(refine_features)
        refine_features = self.continuity_refine(refine_features)
        refined_seg_logits = self.seg_refine(refine_features)
        refined_edge_logits = self.edge_refine(refine_features)
        centerline_logits = self.centerline_refine(refine_features)

        smooth_logits = self.connectivity_smooth(seg_logits)
        seg_logits = seg_logits + self.refine_gate * refined_seg_logits + 0.25 * smooth_logits + self.centerline_gate * centerline_logits
        edge_logits = edge_logits + self.edge_gate * refined_edge_logits

        seg_logits = F.interpolate(seg_logits, size=(height, width), mode="bilinear", align_corners=False)
        edge_logits = F.interpolate(edge_logits, size=(height, width), mode="bilinear", align_corners=False)
        centerline_logits = F.interpolate(centerline_logits, size=(height, width), mode="bilinear", align_corners=False)
        return {
            "seg_logits": seg_logits,
            "edge_logits": edge_logits,
            "centerline_logits": centerline_logits,
        }
