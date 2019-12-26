import torch
import torchvision
from torch.utils.data import Dataset
from PIL import Image
from IPython.display import display
import numpy as np

class MNIST_dataset(Dataset):
    
    def __init__(self, data):
        self.data = data[0] / 255.0
        self.labels = data[1]
        
    def __getitem__(self, idx):
        
        return self.data[idx].view(1, 28, 28), self.labels[idx]
    
    def __len__(self):
        return self.data.size(0)


def show_elem(in_batch):
    N, H, L = in_batch.size()
    s = 112
    in_batch*=255
    out = np.zeros((s, s*N), dtype='uint8')
    for n in range(N):
        x = torch.clamp(in_batch[n], 0, 255)
        x = x.detach().numpy().astype(np.uint8)
        im = Image.fromarray(x)
        im = im.resize((s, s), resample=0)
        out[:, s*n:(n+1)*s] = np.asarray(im)
    display(Image.fromarray(out))