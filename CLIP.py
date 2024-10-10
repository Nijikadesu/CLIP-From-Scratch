import torch
import torch.nn as nn
import torch.nn.functional as F
from structure.TextEncoder import TextEncoder
from structure.ImageEncoder import ImageEncoderResNet

class CLIP(nn.Module):
    """
    CLIP model.
    """
    def __init__(self, word_dict=None):
        super().__init__()
        self.image_encoder = ImageEncoderResNet()
        self.text_encoder = TextEncoder(word_dict=word_dict)

    def forward(self, image, text):
        image = self.image_encoder(image)
        text = self.text_encoder(text)

        logits = torch.matmul(image, torch.transpose(text, 0, 1))
        logits = F.softmax(logits, dim=-1)

        return logits

if __name__ == '__main__':
    clip = CLIP()
    image = torch.rand(5, 1, 28, 28)
    text = torch.rand(5, 19, 16)
    logits = clip(image, text)
    print(logits.shape)