from mri_foundation.models.sam import sam_model_registry  
import mri_foundation.cfg as cfg
import torch

class MRICoreExtractor(torch.nn.Module):
    def __init__(self):
        super(MRICoreExtractor, self).__init__()
        self.args = cfg.parse_args()
        self.checkpoint_path = "./mri_foundation/mri_foundation.pth"

        self.model = sam_model_registry['vit_b'](self.args, checkpoint=self.checkpoint_path, num_classes=self.args.num_cls, image_size=self.args.image_size, pretrained_sam=True)

    def forward(self, image):

        img_emb = self.model.image_encoder(image)
        
        return img_emb
