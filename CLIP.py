import torch
import torch.nn as nn
import torch.nn.functional as F
from data.data_config import get_word_dict
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
    word_dict = get_word_dict()
    clip = CLIP(word_dict=word_dict)
    image = torch.rand(10, 1, 28, 28)
    text = ['a photo of number 0',
            'a photo of number 1',
            'a photo of number 2',
            'a photo of number 3',
            'a photo of number 4',
            'a photo of number 5',
            'a photo of number 6',
            'a photo of number 7',
            'a photo of number 8',
            'a photo of number 9',]
    logits = clip(image, text)
    print(logits.shape)