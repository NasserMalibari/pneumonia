import torchvision.transforms as transforms
from PIL import Image
import random

from torch.utils.data import TensorDataset, DataLoader
from torch.utils.data import Dataset, DataLoader
import torch

base_dir = 'chest_xray/'

train_dir = base_dir + 'train/'
test_dir = base_dir + 'test/'
val_dir = base_dir + 'val/'

train_neg = train_dir + 'NORMAL'
train_pos = train_dir + 'PNEUMONIA'
test_neg = test_dir + 'NORMAL'
test_pos = test_dir + 'PNEUMONIA'
val_neg = val_dir + 'NORMAL'
val_pos = val_dir + 'PNEUMONIA'

train_pos = [train_pos+'/'+i  for i in os.listdir(train_pos) ]
train_neg = [train_neg + '/' + i for i in os.listdir(train_neg) ]


test_pos = [test_pos + '/' + i for i in os.listdir(test_pos) ]
test_neg = [test_neg + '/' + i for i in os.listdir(test_neg)]

val_pos = [val_pos + '/' + i for i in os.listdir(val_pos)]
val_neg = [val_neg + '/' + i for i in os.listdir(val_neg)]

train_paths = train_pos + train_neg
random.seed(42) # Answer to life the universe and everything
random.shuffle(train_paths)

#######################################################
#               Define Dataset Class
#######################################################

class pneumonia_dataset(Dataset):
    def __init__(self, image_paths):
        self.image_paths = train_paths
        
    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_filepath = self.image_paths[idx]
        img = Image.open(image_filepath)
        newsize = (127, 127)
        img = img.resize(newsize)
        
        transform = transforms.Compose([
            transforms.PILToTensor()
        ])
        img_tensor = transform(img)

        label = 0
        if "PNEUMONIA" in image_filepath:
            label = 1
        
        return img_tensor, torch.tensor([label])

#######################################################
#                  Create Dataset
#######################################################

train_dataset = pneumonia_dataset(train_paths)

# Do as above but for test set:
test_paths = test_pos + test_neg
random.shuffle(test_paths)
test_dataset = pneumonia_dataset(test_paths)

def load_datasets():
    return (train_dataset, test_dataset)