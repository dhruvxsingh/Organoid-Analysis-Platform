from dilipredict.nets import STViT, STViTConfig, beitv2


class DILIPredict(STViT):
    DEFAULT_CONF = STViTConfig(
        dim=768,
        num_classes=3,
        pool='mean',
        img_encoder_conf=beitv2.BeitV2.DEFAULT_CONF
    )

    def __init__(self, conf: STViTConfig = DEFAULT_CONF) -> None:
        super().__init__(conf)
