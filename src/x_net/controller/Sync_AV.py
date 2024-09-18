from torch import nn

import torch
import json


class Sync_AV(nn.Module):
    def __init__(self, config_path, inference_mode=False) -> None:
        super(Sync_AV, self).__init__()

        self.config_path = config_path
        self.inference_mode = inference_mode
        self.activation_funcs = {"l_relu": nn.LeakyReLU, "relu": nn.ReLU}
        self.pooling_layer = {"max": nn.MaxPool2d, "avg": nn.AvgPool2d}

        # Load config
        self.config = json.load(open(self.config_path, "r"))

        # Build CNN
        cnn_config = self.config["cnn"]

        conv_layers = [
            nn.Conv2d(
                cnn_config["input_channels"],
                cnn_config["kernel_qty"][0],
                cnn_config["kernel_size"],
            )
        ]

        for i in range(1, cnn_config["number_of_layers"]):
            new_conv_layer = nn.Conv2d(
                cnn_config["kernel_qty"][i - 1],
                cnn_config["kernel_qty"][i],
                cnn_config["kernel_size"],
            )

            conv_layers.append(new_conv_layer)
            conv_layers.append(self.activation_funcs[cnn_config["activation_func"]]())
            conv_layers.append(
                self.pooling_layer[cnn_config["pooling_layer"]](
                    cnn_config["pooling_kernel_size"]
                )
            )

        self.conv_blcok = nn.Sequential(*conv_layers)

        # Build MLP
        mlp_config = self.config["mlp"]
        dense_layers = [
            nn.Linear(mlp_config["input_size"], mlp_config["neurons_per_layer"][0])
        ]

        for i in range(1, mlp_config["number_of_layers"]):
            new_dense_layer = nn.Linear(
                mlp_config["neurons_per_layer"][i - 1],
                mlp_config["neurons_per_layer"][i],
            )

            dense_layers.append(new_dense_layer)
            dense_layers.append(self.activation_funcs[mlp_config["activation_func"]]())

        dense_layers.append(
            nn.Linear(mlp_config["neurons_per_layer"][-1], mlp_config["output_size"])
        )

        self.dense_block = nn.Sequential(*dense_layers)

    def forward(self, x):
        y = self.conv_blcok(x)
        flatten = nn.Flatten()
        embedding = flatten(y)

        if self.inference_mode:
            output = embedding
        else:
            output = self.dense_block(embedding)

        return output


if __name__ == "__main__":
    sync_av = Sync_AV("src/config_files/sync_av.json", inference_mode=False)

    print(sync_av(torch.ones(1, 2, 128, 128)))
