import torch
import torch.nn as nn


def build_cnn_layers(config, conv_type="2d"):
    layers = []
    in_channels = config.get("in_channels", 1)
    for layer_cfg in config.get("layers", []):
        out_channels = layer_cfg.get("out_channels", 32)
        kernel_size = layer_cfg.get("kernel_size", 3)
        stride = layer_cfg.get("stride", 1)
        padding = layer_cfg.get("padding", 1)
        pool = layer_cfg.get("pool", None)

        Conv = nn.Conv3d if conv_type == "3d" else nn.Conv2d
        Pool = nn.MaxPool3d if conv_type == "3d" else nn.MaxPool2d

        layers.append(Conv(in_channels, out_channels, kernel_size, stride, padding))
        layers.append(nn.ReLU())
        if pool:
            layers.append(Pool(pool))

        in_channels = out_channels

    if conv_type == "3d":
        layers.append(nn.AdaptiveAvgPool3d(1))
    else:
        layers.append(nn.AdaptiveAvgPool2d(1))

    return nn.Sequential(*layers), in_channels


class VideoCNNEncoder(nn.Module):
    def __init__(self, in_channels=3, embed_dim=256, layers=[]):
        super().__init__()
        self.encoder, final_channels = build_cnn_layers(
            {"in_channels": in_channels, "layers": layers}, conv_type="3d"
        )
        self.proj = nn.Linear(final_channels, embed_dim)

    def forward(self, x):
        x = self.encoder(x)
        x = x.view(x.size(0), -1)
        return self.proj(x)


class AudioCNNEncoder(nn.Module):
    def __init__(self, in_channels=1, embed_dim=256, layers=[]):
        super().__init__()
        self.encoder, final_channels = build_cnn_layers(
            {"in_channels": in_channels, "layers": layers}, conv_type="2d"
        )
        self.proj = nn.Linear(final_channels, embed_dim)

    def forward(self, x):
        x = self.encoder(x)
        x = x.view(x.size(0), -1)
        return self.proj(x)


class BinarySyncClassifier(nn.Module):
    def __init__(self, input_dim=512, hidden_dim=128):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x):
        return self.classifier(x)


class SyncAV(nn.Module):
    def __init__(self, config):
        super().__init__()
        vid_cfg = config.get("video_cnn", {})
        aud_cfg = config.get("audio_cnn", {})
        pred_cfg = config.get("predictor", {})

        self.video_encoder = VideoCNNEncoder(
            in_channels=vid_cfg.get("in_channels", 3),
            embed_dim=vid_cfg.get("embed_dim", 256),
            layers=vid_cfg.get("layers", []),
        )
        self.audio_encoder = AudioCNNEncoder(
            in_channels=aud_cfg.get("in_channels", 1),
            embed_dim=aud_cfg.get("embed_dim", 256),
            layers=aud_cfg.get("layers", []),
        )

        total_embed_dim = vid_cfg.get("embed_dim", 256) + aud_cfg.get("embed_dim", 256)
        self.classifier = BinarySyncClassifier(
            input_dim=pred_cfg.get("input_dim", total_embed_dim),
            hidden_dim=pred_cfg.get("hidden_dim", 128),
        )

    def forward(self, video, audio, return_embedding=False):
        v_emb = self.video_encoder(video)
        a_emb = self.audio_encoder(audio)
        fused = torch.cat([v_emb, a_emb], dim=1)

        if return_embedding:
            hidden = self.classifier.classifier[0](fused)  # Linear(input_dim → 128)
            hidden = self.classifier.classifier[1](hidden)  # ReLU
            return hidden  # shape: [B, 128]

        return self.classifier(fused)


if __name__ == "__main__":
    config = {
        "video_cnn": {
            "in_channels": 3,
            "embed_dim": 512,
            "layers": [
                {
                    "out_channels": 64,
                    "kernel_size": 3,
                    "stride": 1,
                    "padding": 1,
                    "pool": 2,
                },
                {
                    "out_channels": 128,
                    "kernel_size": 3,
                    "stride": 1,
                    "padding": 1,
                    "pool": 2,
                },
                {
                    "out_channels": 256,
                    "kernel_size": 3,
                    "stride": 1,
                    "padding": 1,
                    "pool": 2,
                },
                {
                    "out_channels": 512,
                    "kernel_size": 3,
                    "stride": 1,
                    "padding": 1,
                    "pool": 2,
                },
            ],
        },
        "audio_cnn": {
            "in_channels": 1,
            "embed_dim": 512,
            "layers": [
                {
                    "out_channels": 64,
                    "kernel_size": 3,
                    "stride": 1,
                    "padding": 1,
                    "pool": 2,
                },
                {
                    "out_channels": 128,
                    "kernel_size": 3,
                    "stride": 1,
                    "padding": 1,
                    "pool": 2,
                },
                {
                    "out_channels": 256,
                    "kernel_size": 3,
                    "stride": 1,
                    "padding": 1,
                    "pool": 2,
                },
                {
                    "out_channels": 512,
                    "kernel_size": 3,
                    "stride": 1,
                    "padding": 1,
                    "pool": 2,
                },
            ],
        },
        "predictor": {"hidden_dim": 256},
    }

    model = SyncAV(config)
    dummy_video = torch.randn(2, 3, 50, 144, 144)
    dummy_audio = torch.randn(2, 1, 256, 256)
    out = model(dummy_video, dummy_audio)
    print("Logit output shape:", out.shape)
