import torch
import torch.nn as nn

from references.EVIT_UNET.unet.eff_unet import Eff_Unet, stem


class EfficientVitUNet(Eff_Unet):
    def __init__(
        self,
        layers,
        embed_dims=None,
        downsamples=None,
        vit_num=6,
        drop_path_rate=0.1,
        resolution=224,
        num_classes=9,
        act_layer=nn.GELU,
        in_channels=1,
        **kwargs
    ):
        super().__init__(
            layers,
            embed_dims=embed_dims,
            downsamples=downsamples,
            vit_num=vit_num,
            drop_path_rate=drop_path_rate,
            resolution=resolution,
            num_classes=num_classes,
            act_layer=act_layer,
            **kwargs
        )

        self.patch_embed = stem(in_channels, embed_dims[0], act_layer=act_layer)


if __name__ == "__main__":
    model = EfficientVitUNet(
        layers=[5, 5, 15, 10],
        embed_dims=[40, 80, 192, 384],
        downsamples=[True, True, True, True],
        resolution=256,
        num_classes=1,
        in_channels=3,
    ).cuda()

    x = torch.randn(1, 3, 256, 256).cuda()
    y = model(x)

    print(y.shape)
