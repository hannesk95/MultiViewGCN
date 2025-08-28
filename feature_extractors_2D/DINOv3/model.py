import PIL.Image
from transformers import AutoImageProcessor, AutoModel
import PIL
import torch

class DINOv3Extractor(torch.nn.Module):
    def __init__(self):
        super(DINOv3Extractor, self).__init__()
        self.processor = AutoImageProcessor.from_pretrained("facebook/dinov3-vits16-pretrain-lvd1689m")
        self.model = AutoModel.from_pretrained("facebook/dinov3-vits16-pretrain-lvd1689m", device_map="auto")

    def forward(self, image):

        assert isinstance(image, PIL.Image.Image), "Input must be a PIL Image"

        inputs = self.processor(images=image, return_tensors="pt")
        inputs.to("cuda" if torch.cuda.is_available() else "cpu")
        outputs = self.model(**inputs)
        
        return outputs.pooler_output
