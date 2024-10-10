import torchvision
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor, ToPILImage

class MNIST(Dataset):
    """
    Dataset.
    """
    def __init__(self):
        super().__init__()
        self.dataset = torchvision.datasets.MNIST(root="", train=True, download=True)
        self.transform = ToTensor()

    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        image = self.transform(image)

        prompt_template = "a photo of number "
        label = prompt_template + str(label)


        return image, label

    def __len__(self):
        return len(self.dataset)

if __name__ == '__main__':
    dataset = MNIST()
    image, label = dataset[0]
    print(image.shape)

    image = ToPILImage()(image)

    plt.figure(figsize=(32, 32))
    plt.imshow(image)
    plt.show()

    print(label)