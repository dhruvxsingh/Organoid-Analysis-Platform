import typing
import torch
from dilipredict.image_loader import ImageLoader


class DILIPredict:
    def __init__(self, image_loader: ImageLoader, device: torch.device = torch.device('cpu')) -> None:
        self.image_loader = image_loader
        self.device = device

    def __call__(self, sample_img_files: typing.Sequence) -> torch.Tensor:
        return self.image_loader.encoder(sample_img_files)
