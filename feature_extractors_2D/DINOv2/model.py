import PIL.Image
from transformers import AutoImageProcessor, AutoModel
import PIL
import torch

class DINOv2Extractor(torch.nn.Module):
    def __init__(self):
        super(DINOv2Extractor, self).__init__()
        self.processor = AutoImageProcessor.from_pretrained('facebook/dinov2-small')
        self.model = AutoModel.from_pretrained('facebook/dinov2-small')

    def forward(self, image):

        assert isinstance(image, PIL.Image.Image), "Input must be a PIL Image"

        inputs = self.processor(images=image, return_tensors="pt")
        inputs.to("cuda" if torch.cuda.is_available() else "cpu")
        outputs = self.model(**inputs)
        
        return outputs.pooler_output
