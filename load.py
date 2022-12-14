import glob
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler


# create custom dataset
class FingerprintDataset(Dataset):
    # initialize the variables
    def __init__(self):
        # contains the base path to the dataset directory
        self.images_path = "DB1_B/"
        # search for all subdirectories
        file_list = glob.glob(self.images_path + "*")
        print(file_list)
        self.data = []
        for class_path in file_list:
            class_name = class_path.split("/")[-1]
            for img_path in glob.glob(class_path + "/*.tif"):
                self.data.append([img_path, class_name])
        print(self.data)
        self.class_map = {}
        for i in range(1, 11):
            self.class_map[f'{i}'] = i
        self.img_dim = (300, 300)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, class_name = self.data[idx]
        img = cv2.imread(img_path)
        img = cv2.resize(img, self.img_dim)
        class_id = self.class_map[class_name]
        # convert variables to the torch tensor format for gradient calculating
        img_tensor = torch.from_numpy(img)
        # Torch convolutions require images to be in a channel first format
        # Channels=2, Width=0, Height=1
        img_tensor = img_tensor.permute(2, 0, 1)
        # convert the integer value of class_id to a torch tensor
        class_id = torch.tensor([class_id])
        return img_tensor, class_id

    def __len__(self):
        return len(self.class_map)


dataset = FingerprintDataset()
data_loader = DataLoader(dataset, batch_size=16, shuffle=True)
for images, labels in data_loader:
    print("Batch of images has shape: ", images.shape)
    print("Batch of labels has shape: ", labels.shape)

batch_size = 16
validation_split = .2
shuffle_dataset = True
random_seed = 42

# Creating data indices for training and validation splits:
dataset_size = len(dataset)
indices = list(range(dataset_size))
split = int(np.floor(validation_split * dataset_size))
if shuffle_dataset:
    np.random.seed(random_seed)
    np.random.shuffle(indices)
train_indices, val_indices = indices[split:], indices[:split]


# Creating PT data samplers and loaders:
train_sampler = SubsetRandomSampler(train_indices)
valid_sampler = SubsetRandomSampler(val_indices)

train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                           sampler=train_sampler)
validation_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                                sampler=valid_sampler)
