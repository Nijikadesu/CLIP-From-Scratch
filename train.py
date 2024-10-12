import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from CLIP import CLIP
from data.dataset import MNIST
import os
from torch.utils.tensorboard import SummaryWriter
from data.data_config import get_word_dict

class TrainConfig:
    """
    Settings of Environment and Hyper Parameter.
    """
    max_iter = 10000
    learning_rate = 1e-3
    batch_size = 64
    save_step = 1000
    target_count = 10
    log_dir = './logs'
    model_dir = './models'

    for item in (log_dir, model_dir):
        if not os.path.exists(item):
            os.makedirs(item)

class Trainer:
    """
    Class to train CLIP model.
    """
    def __init__(self, cfg):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.max_iter = cfg.max_iter
        self.learning_rate = cfg.learning_rate
        self.batch_size = cfg.batch_size
        self.save_step = cfg.save_step
        self.target_count = cfg.target_count
        self.log_dir = cfg.log_dir
        self.model_dir = cfg.model_dir

        self.word_dict = get_word_dict()
        self.dataset = MNIST()
        self.CLIP = CLIP(self.word_dict).to(self.device)

        self.optimizer = torch.optim.Adam(filter(lambda x: x.requires_grad, self.CLIP.parameters()))

    def train(self):
        self.CLIP.train()
        writer = SummaryWriter(self.log_dir)
        trainloader = DataLoader(
            dataset=self.dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
            drop_last=True
        )

        for i in range(self.max_iter):
            if i % 500 == 0:
                print('-' * 10, f'iteration {i:04d}', '-' * 10)
            while True:
                batch = next(iter(trainloader))
                image, label = batch
                if torch.unique(label).shape[0] < self.target_count:
                    continue
                target = set()
                indexes = []
                for j in range(self.batch_size):
                    if label[j] in target:
                        continue
                    target.add(label[j])
                    indexes.append(j)

                    if len(target) == self.target_count:
                        break
                batch_image = image[indexes]
                batch_text = label[indexes]
                break

            logits = self.CLIP(batch_image.to(self.device), batch_text.to(self.device))
            targets = torch.arange(0, self.target_count).to(self.device)

            loss_i = F.cross_entropy(logits, targets)
            loss_t = F.cross_entropy(logits.permute(1, 0), targets)
            loss = (loss_i + loss_t) / 2

            print(f'train loss: {loss.item():.3f}')

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            writer.add_scalar('loss/train', loss.item(), i)

        model_path = os.path.join(self.model_dir, '/model.pt')
        torch.save(self.CLIP.state_dict(), model_path)

        writer.close()

if __name__ == '__main__':
    cfg = TrainConfig()
    trainer = Trainer(cfg)
    trainer.train()