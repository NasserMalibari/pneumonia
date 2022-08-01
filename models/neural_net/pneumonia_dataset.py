import torchvision.transforms as transforms
from PIL import Image, ImageOps
import random
import os

from torch.utils.data import TensorDataset, DataLoader
from torch.utils.data import Dataset, DataLoader
import torch
from tqdm import tqdm


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


# train_pos: list of pathnames to positive images
train_pos = [train_pos+'/'+i  for i in os.listdir(train_pos) ]

# train_neg: list of pathnames to negative images
train_neg = [train_neg + '/' + i for i in os.listdir(train_neg) ]

# test_pos: list of pathnames to positive images
test_pos = [test_pos + '/' + i for i in os.listdir(test_pos) ]

# test_neg: list of pathnames to negative images
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
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_filepath = self.image_paths[idx]
        
        # grayscale and resize all images
        transform = transforms.Grayscale()
        img = Image.open(image_filepath)
        img = transform(img)
        newsize = (127, 127)
        img = img.resize(newsize)
        
        # change image object to tensor for training
        transform = transforms.Compose([
            transforms.PILToTensor()
        ])
        img_tensor = transform(img)

        # assign label
        label = 0
        if "bacteria" in image_filepath:
            label = 1
        if "virus" in image_filepath:
            label = 2
        
        return img_tensor.float().to(self.device), torch.tensor(label).long().to(self.device)

#######################################################
#                  Create Dataset
#######################################################

# labels:
# NORMAL = 0
# BACTERIA = 1
# VIRUS = 2

directories = os.listdir('chest_xray/train/NORMAL')
mirror_exists = False
for directory in directories:
    if "mirror" in directory:
        mirror_exists = True
        break

# if mirroring hasnt occured:
# mirror all images to get more examples
# generating synthetic data
if mirror_exists == False:
    for i, img in tqdm(enumerate(train_neg)):
        im = Image.open(img)
        im_mirror = ImageOps.mirror(im)
        im_mirror.save(f'chest_xray/train/NORMAL/mirror_{i}.jpeg')

    # update train_paths list to include new mirrored images
    train_neg = [train_neg + '/' + i for i in os.listdir(train_neg) ]
    train_paths = train_pos + train_neg


# CREATE train dataset
train_dataset = pneumonia_dataset(train_paths)

# Create test dataset:
test_paths = test_pos + test_neg
random.shuffle(test_paths)
test_dataset = pneumonia_dataset(test_paths)

def load_datasets():
    return (train_dataset, test_dataset)