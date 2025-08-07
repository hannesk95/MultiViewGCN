import torch
from PIL import Image
from torchvision import transforms
from Franca.franca.hub.backbones import _make_franca_model

# Franca -- In21k
# franca_vitb14 = torch.hub.load('valeoai/Franca', 'franca_vitb14')
# franca_vitl14 = torch.hub.load('valeoai/Franca', 'franca_vitl14')
# franca_vitg14 = torch.hub.load('valeoai/Franca', 'franca_vitg14')

# Franca -- Laion600M
# franca_vitl14 = torch.hub.load('valeoai/Franca', 'franca_vitl14', weights='LAION')
# franca_vitg14 = torch.hub.load('valeoai/Franca', 'franca_vitg14', weights='LAION')

# Dinov2 baseline -- In21k
# franca_vitb14 = torch.hub.load('valeoai/Franca', 'franca_vitb14', weights='Dinov2_In21k')
# franca_vitl14 = torch.hub.load('valeoai/Franca', 'franca_vitl14', weights='Dinov2_In21k')


class FrancaExtractor(torch.nn.Module):
    def __init__(self):
        super(FrancaExtractor, self).__init__()

        # Franca -- In21k
        # franca_vitb14 
        ckpt_path = torch.hub.load('valeoai/Franca', 'franca_vitb14')
        # franca_vitl14 
        ckpt_path = torch.hub.load('valeoai/Franca', 'franca_vitl14')
        # franca_vitg14 
        ckpt_path = torch.hub.load('valeoai/Franca', 'franca_vitg14')

        # Franca -- Laion600M
        # franca_vitl14 
        ckpt_path = torch.hub.load('valeoai/Franca', 'franca_vitl14', weights='LAION')
        # franca_vitg14 
        ckpt_path = torch.hub.load('valeoai/Franca', 'franca_vitg14', weights='LAION')

        # Dinov2 baseline -- In21k
        # franca_vitb14 
        ckpt_path = torch.hub.load('valeoai/Franca', 'franca_vitb14', weights='Dinov2_In21k')
        # franca_vitl14 
        ckpt_path = torch.hub.load('valeoai/Franca', 'franca_vitl14', weights='Dinov2_In21k')

        self.ckpt = ckpt_path
        self.model = _make_franca_model(arch_name=arch_name,
                                        img_size=img_size,
                                        pretrained=True,
                                        local_state_dict=ckpt_path    )

    def forward(self, image):
        with torch.no_grad():
            feats = self.model.forward_features(x)
            cls_token = feats["x_norm_clstoken"]
            patch_tokens = feats["x_norm_patchtokens"]



        