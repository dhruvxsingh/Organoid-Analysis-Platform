import typing
import torch
import numpy as np
from PIL import Image
from einops import rearrange
from torchvision import transforms


class ImageLoader:
    CELL_IMG_MEAN = (0.7408901, 0.74743044, 0.75368834)
    CELL_IMG_STD = (0.064860135, 0.06883157, 0.073834345)

    def __init__(
        self,
        img_mean: typing.Sequence[float] = CELL_IMG_MEAN,
        img_std: typing.Sequence[float] = CELL_IMG_STD,
        img_size: int = 224,
    ) -> None:
        self.img_size = img_size
        self.common_transform = transforms.Compose([
            transforms.ColorJitter(0.4, 0.4, 0.4),
            transforms.Resize((img_size, img_size))])
        self.patch_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=torch.tensor(img_mean),
                std=torch.tensor(img_std))
        ])

    def resize_image(self, path):
        img = Image.open(path)
        img = self.common_transform(img.convert('RGB'))
        img = self.patch_transform(img)
        return img

    def encoder(self, img_files: typing.Sequence) -> torch.Tensor:
        '''
        img_files: [time_0_list, ..., time_n_list]
            time_0_list: [img_0_path, ..., img_n_path]
        '''
        imgs = []
        time_num = len(img_files)
        for paths in img_files:
            for path in paths:
                img = self.resize_image(path)
                img = img.reshape(self.img_size, self.img_size, 3, 1)
                imgs.append(img)
        img = np.concatenate(imgs, axis=3)
        img = torch.Tensor(img)
        img = img.permute(2, 3, 0, 1)
        img = rearrange(img, 'c (t s) h w -> c t s h w', t=time_num)
        return img
