import torch
from monai.networks.nets import SwinUNETR

def build_swinunetr_model(config=None):
    model = SwinUNETR(
        img_size=config["model_parameters"]["img_size"],
        in_channels=config["model_parameters"]["in_channels"],
        out_channels=config["model_parameters"]["num_classes"],
        feature_size=config["model_parameters"]["feature_size"],
        use_checkpoint=config["model_parameters"]["use_checkpoint"],
        spatial_dims=3,
        use_v2=True,
        downsample="mergingv2"
    )
    return model 