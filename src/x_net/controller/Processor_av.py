import torch
import torch.nn as nn
import torch.nn.functional as F


class ProcessorAV(nn.Module):
    def __init__(self, model_cfg):
        super(ProcessorAV, self).__init__()

        # === Retrieve model args ===
        self.layers = model_cfg["layers"]
        self.img_channels = model_cfg["img_channels"]
        self.latent_dim = model_cfg["latent_dim"]
        self.kernel_sizes = model_cfg["kernel_sizes"]
        self.kernel_strides = model_cfg["kernel_strides"]
        self.kernel_padding = model_cfg["kernel_paddings"]
        self.input_size = model_cfg.get("input_size", (256, 256))  # Default to 256x256

        self.use_batchnorm = model_cfg.get("use_batchnorm", True)
        self.dropout_prob = model_cfg.get("dropout_prob", 0.0)

        self._check_params()

        # === Build Encoder ===
        self.encoder = nn.ModuleList()
        in_ch = self.img_channels
        h, w = self.input_size
        self.feature_shapes = []  # Track shapes after each layer

        for i, out_ch in enumerate(self.layers):
            block = [
                nn.Conv2d(
                    in_ch,
                    out_ch,
                    kernel_size=self.kernel_sizes[i],
                    stride=self.kernel_strides[i],
                    padding=self.kernel_padding[i],
                )
            ]

            if self.use_batchnorm:
                block.append(nn.BatchNorm2d(out_ch))
            block.append(nn.ReLU())

            if self.dropout_prob > 0:
                block.append(nn.Dropout2d(self.dropout_prob))

            self.encoder.append(nn.Sequential(*block))

            h = (
                h + 2 * self.kernel_padding[i] - self.kernel_sizes[i]
            ) // self.kernel_strides[i] + 1
            w = (
                w + 2 * self.kernel_padding[i] - self.kernel_sizes[i]
            ) // self.kernel_strides[i] + 1
            self.feature_shapes.append((h, w))
            in_ch = out_ch

        self.final_feature_shape = self.feature_shapes[-1]
        h, w = self.final_feature_shape
        self.fc_enc = nn.Linear(self.layers[-1] * h * w, self.latent_dim)

        # === Build Decoder ===
        self.fc_dec = nn.Linear(self.latent_dim, self.layers[-1] * h * w)
        self.decoder = nn.ModuleList()

        # ConvTranspose layers
        for i in range(len(self.layers) - 1, 0, -1):
            in_ch = self.layers[i]
            out_ch = self.layers[i - 1]
            target_shape = self.feature_shapes[i - 1]
            stride = self.kernel_strides[i]
            padding = self.kernel_padding[i]
            kernel_size = self.kernel_sizes[i]

            out_h = (self.feature_shapes[i][0] - 1) * stride - 2 * padding + kernel_size
            out_w = (self.feature_shapes[i][1] - 1) * stride - 2 * padding + kernel_size
            output_padding = (target_shape[0] - out_h, target_shape[1] - out_w)

            block = [
                nn.ConvTranspose2d(
                    in_ch,
                    out_ch,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                    output_padding=output_padding,
                )
            ]

            if self.use_batchnorm:
                block.append(nn.BatchNorm2d(out_ch))
            block.append(nn.ReLU())

            if self.dropout_prob > 0:
                block.append(nn.Dropout2d(self.dropout_prob))

            self.decoder.append(nn.Sequential(*block))

        # Final output layer (no BN or dropout, with Sigmoid)
        final_stride = self.kernel_strides[0]
        final_padding = self.kernel_padding[0]
        final_kernel = self.kernel_sizes[0]
        out_h = (
            (self.feature_shapes[0][0] - 1) * final_stride
            - 2 * final_padding
            + final_kernel
        )
        out_w = (
            (self.feature_shapes[0][1] - 1) * final_stride
            - 2 * final_padding
            + final_kernel
        )
        final_output_padding = (self.input_size[0] - out_h, self.input_size[1] - out_w)

        self.decoder.append(
            nn.ConvTranspose2d(
                self.layers[0],
                self.img_channels,
                kernel_size=final_kernel,
                stride=final_stride,
                padding=final_padding,
                output_padding=final_output_padding,
            )
        )

    def _compute_output_size(self):
        h, w = self.input_size
        for k, s, p in zip(self.kernel_sizes, self.kernel_strides, self.kernel_padding):
            h = (h + 2 * p - k) // s + 1
            w = (w + 2 * p - k) // s + 1
        return h, w

    def _check_params(self):
        """
        Validates the parameters for configuring a neural network architecture.
        Parameters:
            img_channels (int): Number of input image channels.
            latent_dim (int): Dimensionality of the latent space.
            layers (list): List specifying the number of units or filters in each layer.
            kernel_sizes (list): List specifying the kernel size for each layer.
            kernel_strides (list): List specifying the stride for each layer.
            kernel_padding (list): List specifying the padding for each layer.
        Raises:
            AssertionError: If any parameter does not meet the expected type or if the lengths of
                            layers, kernel_sizes, kernel_strides, and kernel_padding do not match.
        """

        assert isinstance(
            self.img_channels, int
        ), "Image Channels parameter must be an integer"
        assert isinstance(
            self.latent_dim, int
        ), "Latent dim parameter must be an integer"
        for name, var in zip(
            ["layers", "kernel_sizes", "kernel_strides", "kernel_padding"],
            [self.layers, self.kernel_sizes, self.kernel_strides, self.kernel_padding],
        ):
            assert isinstance(var, list), f"{name} must be a list"
        assert (
            len(self.layers)
            == len(self.kernel_padding)
            == len(self.kernel_sizes)
            == len(self.kernel_strides)
        ), "All kernel parameters should match layer sizes"

    def encode(self, x):
        for layer in self.encoder:
            x = F.relu(layer(x))
        x = x.contiguous().view(x.size(0), -1)
        z = self.fc_enc(x)
        return z

    def decode(self, z):
        # Use the actual final shape stored in self.final_feature_shape (h, w)
        h, w = self.final_feature_shape
        x = self.fc_dec(z)
        x = x.view(-1, self.layers[-1], h, w)

        for i, layer in enumerate(self.decoder):
            if i == len(self.decoder) - 1:
                x = torch.tanh(layer(x))  # final layer
            else:
                x = F.relu(layer(x))
        return x

    def forward(self, x):
        z = self.encode(x)
        recon_x = self.decode(z)
        return recon_x, z


if __name__ == "__main__":
    import torch

    # Sample config
    config = {
        "img_channels": 3,
        "layers": [32, 64, 128],
        "kernel_sizes": [2, 4, 2],
        "kernel_strides": [2, 2, 2],
        "kernel_paddings": [0, 1, 1],
        "latent_dim": 256,
        "input_size": (244, 244),
        "use_batchnorm": True,
        "dropout_prob": 0.1,
    }

    # Create model
    model = ProcessorAV(config)
    model.eval()

    # Dummy input [B, C, H, W]
    x = torch.rand(2, 3, 224, 224)

    # Forward pass
    with torch.no_grad():
        recon, z = model(x)

    # Checks
    print("Input shape:       ", x.shape)
    print("Reconstructed shape:", recon.shape)
    print("Latent shape:       ", z.shape)

    assert recon.shape == x.shape, "Reconstruction shape mismatch"
    assert z.shape == (2, config["latent_dim"]), "Latent vector shape mismatch"

    print("✅ ProcessorAV test passed.")
