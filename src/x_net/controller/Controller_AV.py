import torch
from torch import nn
from x_net.controller.Sync_AV import SyncAV
from x_net.controller.Processor_av import ProcessorAV
from utils.common import read_json


class ControllerAV(nn.Module):
    def __init__(self, controller_config):
        super().__init__()

        self.config = controller_config

        # Load configs
        self.sync_av_config = read_json(controller_config["syncav"])["model_parameters"]
        self.audio2audio_config = read_json(controller_config["audio2audio"])[
            "model_parameters"
        ]
        self.video2video_config = read_json(controller_config["video2video"])[
            "model_parameters"
        ]

        # Load SyncAV and two AV processors
        self.sync_av = SyncAV(self.sync_av_config)
        self.audio2video = ProcessorAV(self.audio2audio_config)
        self.video2audio = ProcessorAV(self.video2video_config)

        # Load weights
        sync_av_weights = controller_config["syncav_weights"]
        audio2audio_weights = controller_config["audio2audio_weights"]
        video2video_weights = controller_config["video2video_weights"]

        self.sync_av.load_state_dict(torch.load(sync_av_weights))
        self.audio2video.load_state_dict(torch.load(audio2audio_weights))
        self.video2audio.load_state_dict(torch.load(video2video_weights))

        # Swap decoder logic
        self.audio2video.decoder, self.video2audio.decoder = (
            self.video2audio.decoder,
            self.audio2video.decoder,
        )

        # Freeze models
        for param in self.sync_av.parameters():
            param.requires_grad = False
        for param in self.audio2video.parameters():
            param.requires_grad = False
        for param in self.video2audio.parameters():
            param.requires_grad = False

        self.video_fusion_cnn = nn.Sequential(
            nn.Conv2d(
                in_channels=3 + self.sync_av_config["predictor"]["hidden_dim"],
                out_channels=64,
                kernel_size=3,
                padding=1,
            ),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
        )

        self.audio_fusion_cnn = nn.Sequential(
            nn.Conv2d(
                in_channels=3 + self.sync_av_config["predictor"]["hidden_dim"],
                out_channels=64,
                kernel_size=3,
                padding=1,
            ),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
        )

    def process_input(self, raw_frames, raw_spectrogram):
        """
        Prepare input for SyncAV and ProcessorAV modules.
        Applies normalization to frames as needed by ProcessorAV.

        Args:
            raw_frames: Tensor [B, T, C, H, W] — assumed to be already resized
            raw_spectrogram: Tensor [B, C, H, W]

        Returns:
            Tuple of:
            - frames_for_syncav: Tensor [B, T, C, H, W]
            - spec_for_syncav: Tensor [B, C, H, W]
            - frames_for_processor: Tensor [B, T, C, H, W]
            - spec_for_processor: Tensor [B, C, H, W]
        """
        # For SyncAV: use raw directly
        frames_for_syncav = raw_frames.permute(0, 4, 1, 2, 3)
        spec_for_syncav = raw_spectrogram.permute(0, 3, 1, 2)
        # For ProcessorAV: normalize frames (mean=0.5, std=0.5)

        frames_for_processor = raw_frames / 255.0
        frames_for_processor = (frames_for_processor - 0.5) / 0.5

        spec_for_processor = raw_spectrogram.permute(0, 3, 1, 2)
        spec_for_processor /= 255.0
        spec_for_processor = (spec_for_processor - 0.5) / 0.5

        return (
            frames_for_syncav,
            spec_for_syncav,
            frames_for_processor,
            spec_for_processor,
        )

    def av_sync_block(self, frames, spectrogram):
        """
        Run SyncAV model and return synchronization embedding.

        Args:
            frames: Tensor [B, T, C, H, W]
            spectrogram: Tensor [B, C, H, W]

        Returns:
            Tensor [B, D] — Sync embedding
        """
        return self.sync_av(frames, spectrogram, return_embedding=True)

    def av_processor_block(self, frames, spectrogram):
        """
        Generate:
        - Single video frame from full spectrogram
        - Spectrogram from center video frame
        """

        # Generate one frame from full spectrogram
        gen_frame = self.audio2video(spectrogram)[0]  # [B, C, H, W]

        # Use center frame for spectrogram generation
        T = frames.size(1)
        center_frame = frames[:, T // 2].permute(0, 3, 1, 2)  # [B, C, H, W]
        generated_spec = self.video2audio(center_frame)[0]  # [B, C, H, W]

        return gen_frame, generated_spec

    def process_output(self, sync_embedding, video_frame, generated_spec):
        """
        Fuse sync embedding with generated outputs using spatial broadcast,
        followed by convolutional projection to produce modulation vectors.
        """

        B, C, H, W = generated_spec.shape
        D_sync = sync_embedding.size(-1)

        # --- Spatial broadcast of sync embedding
        sync_map = (
            sync_embedding.unsqueeze(-1).unsqueeze(-1).expand(-1, D_sync, H, W)
        )  # [B, D_sync, H, W]

        # --- Concatenate and fuse
        video_fused = torch.cat([video_frame, sync_map], dim=1)  # [B, C + D_sync, H, W]
        audio_fused = torch.cat([generated_spec, sync_map], dim=1)

        # --- Convolutional projection
        video_emb = self.video_fusion_cnn(video_fused)  # [B, D_mod, H, W]
        audio_emb = self.audio_fusion_cnn(audio_fused)

        # --- Global average pooling to get [B, D_mod]
        video_emb = video_emb.mean(dim=[-1, -2])  # [B, D_mod]
        audio_emb = audio_emb.mean(dim=[-1, -2])

        return {
            "sync_embedding": sync_embedding,  # [B, D_sync]
            "generated_frame": video_frame,  # [B, T, C, H, W] or [B, C, H, W]
            "generated_spectrogram": generated_spec,  # [B, C, H, W]
            "video_emb": video_emb,  # [B, D_mod]
            "audio_emb": audio_emb,  # [B, D_mod]
        }

    def forward(self, raw_frames, raw_spectrogram):
        # 1. Preprocess
        syncAV_frames, syncAV_spec, processor_frames, processor_spec = (
            self.process_input(raw_frames, raw_spectrogram)
        )

        # 2. Get Sync Embedding
        sync_embedding = self.av_sync_block(syncAV_frames, syncAV_spec)

        # 3. Generate Cross-modal representations
        gen_image, gen_spec = self.av_processor_block(processor_frames, processor_spec)

        # 4. Postprocess & return
        return self.process_output(sync_embedding, gen_image, gen_spec)
