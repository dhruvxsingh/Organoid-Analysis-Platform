import torch
import typing
import torch.nn as nn
from dilipredict.image_loader import ImageLoader


class DILIPredict:
    def __init__(self, image_loader: ImageLoader, model: nn.Module, device: torch.device = torch.device('cpu')) -> None:
        self.image_loader = image_loader
        model = model.to(device)
        model.eval()
        model.requires_grad_(False)
        self.model = model
        self.device = device

    def __call__(self, img_files: typing.Sequence) -> int:
        '''
        img_files: img_files: [time_0_list, ..., time_n_list]
            time_0_list: [img_0_path, ..., img_n_path]
        '''
        img_x = self.image_loader.encoder(img_files).unsqueeze(0)
        out = self.model(img_x)
        out_label = out.argmax(dim=-1).item()
        probability = out.cpu().numpy().tolist()[0]
        return out_label, probability
