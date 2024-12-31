from mmpose.registry import MODELS
from .base_backbone import BaseBackbone
from .mobilenet_v4_timm import _gen_mobilenet_v4

@MODELS.register_module()
class MobileNetV4(BaseBackbone):
    def __init__(self,model_name='mobilenetv4_conv_medium', channel_multiplier=1.0,init_cfg=[
                     dict(type='Kaiming', layer=['Conv2d']),
                     dict(type='Constant', val=1, layer=['_BatchNorm'])
                 ],**kwargs):  
        super().__init__(init_cfg=init_cfg)
        self.model=_gen_mobilenet_v4(model_name,channel_multiplier,features_only=True,**kwargs)
        # self.model=_gen_mobilenet_v4(model_name,channel_multiplier,pretrained,num_classes=1000,**kwargs)
    
    def forward(self, x):
        return tuple(self.model(x))
    
if __name__ == '__main__':
    import torch
    from torchinfo import summary
    model=MobileNetV4()
    x=torch.rand(1,3,256,192)
    model.eval()
    out=model(x)
    for i in out:
        print(i.shape)
    summary(model.model, input_size=(1, 3, 256, 192))