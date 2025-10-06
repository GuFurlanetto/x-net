import torch
import torch.nn as nn
import numpy as np
import json

from transformers import (
    ViTModel,
    VideoMAEModel,
    AutoImageProcessor,
)

from x_net.controller import ControllerAV


class FiLMBlock(nn.Module):
    def __init__(self, input_dim, cond_dim):
        """
        input_dim: dimension of the feature to be modulated (e.g., backbone output)
        cond_dim: dimension of the conditioning vector (e.g., video_emb or audio_emb)
        """
        super().__init__()
        self.gamma_fc = nn.Linear(cond_dim, input_dim)
        self.beta_fc = nn.Linear(cond_dim, input_dim)

    def forward(self, x, cond):
        """
        x: [B, D] - feature to modulate
        cond: [B, D_cond] - modulating embedding
        Returns: modulated x
        """
        gamma = self.gamma_fc(cond)
        beta = self.beta_fc(cond)
        return gamma * x + beta


class XNet(nn.Module):
    def __init__(self, model_cfg):
        super().__init__()

        # Extract configuratios
        video_model_path = model_cfg["video_backbone_path"]
        video_output_size = model_cfg["video_feature_dim"]
        audio_model_path = model_cfg["audio_backbone_path"]
        audio_output_size = model_cfg["audio_feature_dim"]
        cond_dim = model_cfg["cond_dim"]
        num_classes = model_cfg["num_classes"]
        freeze_video = model_cfg["freeze_video"]
        freeze_audio = model_cfg["freeze_audio"]

        self.video_model = VideoMAEModel.from_pretrained(
            video_model_path, num_labels=num_classes
        )

        self.audio_model = ViTModel.from_pretrained(
            audio_model_path, num_labels=num_classes
        )

        # Remove backbones head
        self.video_model.classifier = nn.Identity()
        self.audio_model.classifier = nn.Identity()

        self.use_controler = model_cfg["use_controller_av"]
        self.hidden_size = model_cfg["hidden_size"]

        if self.use_controler:
            self.controler_AV_blcok = ControllerAV(model_cfg["controller_cfg"])
            self.film_video = FiLMBlock(input_dim=video_output_size, cond_dim=cond_dim)
            self.film_audio = FiLMBlock(input_dim=audio_output_size, cond_dim=cond_dim)

        if freeze_video:
            for param in self.video_model.parameters():
                param.requires_grad = False

        if freeze_audio:
            for param in self.audio_model.parameters():
                param.requires_grad = False

        self.classifier = nn.Linear(self.hidden_size, num_classes)

    def forward(
        self, video_pixel_values, audio_pixel_values, frames=None, spectogram=None
    ):
        video_out = self.video_model(pixel_values=video_pixel_values).last_hidden_state[
            :, 0, :
        ]
        audio_out = self.audio_model(pixel_values=audio_pixel_values).last_hidden_state[
            :, 0, :
        ]

        if self.use_controler:
            assert frames is not None and spectogram is not None
            ctrl_out = self.controler_AV_blcok(frames, spectogram)

            video_emb = ctrl_out["video_emb"]  # [B, cond_dim]
            audio_emb = ctrl_out["audio_emb"]  # [B, cond_dim]

            video_out = self.film_video(video_out, video_emb)  # FiLM modulation
            audio_out = self.film_audio(audio_out, audio_emb)

            fused = torch.cat([video_out, audio_out], dim=-1)
        else:
            fused = torch.cat([video_out, audio_out], dim=-1)

        logits = self.classifier(fused)
        return logits


if __name__ == "__main__":
    # Set seed and device
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load config and model
    config = json.load(open("src/x_net/model_configs/x_net.json", "r"))
    model = XNet(config["model_parameters"]).to(device)
    model.eval()

    # Load processors
    video_processor = AutoImageProcessor.from_pretrained("MCG-NJU/videomae-base")
    audio_processor = AutoImageProcessor.from_pretrained(
        "google/vit-base-patch16-224-in21k"
    )

    # Dummy video input: 16 frames of 224x224 RGB
    dummy_video = [torch.rand(3, 224, 224) for _ in range(16)]
    video_inputs = video_processor(dummy_video, return_tensors="pt")["pixel_values"].to(
        device
    )  # shape: [1, 16, 3, 224, 224]

    # Dummy audio spectrogram input: single image
    dummy_audio = torch.rand(3, 224, 224)
    audio_inputs = audio_processor(dummy_audio, return_tensors="pt")["pixel_values"].to(
        device
    )  # shape: [1, 3, 224, 224]

    # Forward pass
    with torch.no_grad():
        import time

        t0 = time.time()
        logits = model(
            video_pixel_values=video_inputs,
            audio_pixel_values=audio_inputs,
            frames=torch.from_numpy(np.array(dummy_video)).unsqueeze(0).to(device),
            spectogram=torch.from_numpy(np.array(dummy_audio)).unsqueeze(0).to(device),
        )

        t1 = time.time()

    print(f"✅ XNet test passed. Output shape: {logits.shape}\nElapsed Time: {t1 - t0}")
    assert logits.shape == (1, config["model_parameters"]["num_classes"])
