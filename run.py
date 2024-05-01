#!/usr/bin/env python3

import time
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchinfo import summary
import torchvision.io as io
import os
import json
from torchvision.io.video import re
from tqdm import tqdm
import csv
import timm
import wandb

from PIL import Image
import torchvision.transforms.v2 as transforms

# UTILITIES

def extract_frames(video_path, nb_frames=10, delta=1, timeit=False):
    # use time to measure the time it takes to resize a video
    t1 = time.time()
    reader = io.VideoReader(video_path)
    # take 10 frames uniformly sampled from the video
    frames = []
    for i in range(nb_frames):
        reader.seek(delta)
        frame = next(reader)
        frames.append(frame['data'])
    t2 = time.time()     
    video = torch.stack(frames)
    if timeit:
        print(f"read: {t2-t1}")
    return video

def smart_resize(data, size): # kudos louis
    # Prends un tensor de shape [...,C,H,W] et le resize en [...C,size,size]
    # x, y, height et width servent a faire un crop avant de resize

    full_height = data.shape[-2]
    full_width = data.shape[-1]

    if full_height > full_width:
        alt_height = size
        alt_width = int(full_width / (full_height / size))
    elif full_height < full_width:
        alt_height = int(full_height / (full_width / size))
        alt_width = size
    else:
        alt_height = size
        alt_width = size
    tr = transforms.Compose([
        transforms.Resize((alt_height, alt_width)),
        transforms.CenterCrop(size)
    ])
    return tr(data)

def resize_data(data, new_height, new_width, x=0, y=0, height=None, width=None):
    # Prends un tensor de shape [...,C,H,W] et le resize en [C,new_height,new_width]
    # x, y, height et width servent a faire un crop avant de resize

    full_height = data.shape[-2]
    full_width = data.shape[-1]
    height = full_height - y if height is None else height
    width = full_width -x if width is None else width

    ratio = new_height/new_width
    if height/width > ratio:
        expand_height = height
        expand_width = int(height / ratio)
    elif height/width < ratio:
        expand_height = int(width * ratio)
        expand_width = width
    else:
        expand_height = height
        expand_width = width
    tr = transforms.Compose([
        transforms.CenterCrop((expand_height, expand_width)),
        transforms.Resize((new_height, new_width))
    ])
    x = data[...,y:min(y+height, full_height), x:min(x+width, full_width)].clone()
    return tr(x)


# SETUP DATASET

dataset_dir = "/raid/datasets/hackathon2024"
root_dir = os.path.expanduser("~/automathon-2024")
nb_frames = 10

## MAKE RESIZED DATASET
resized_dir = os.path.join(dataset_dir, "resized_dataset")
"""
create_small_dataset = False
errors = []
if not os.path.exists(resized_dir) or create_small_dataset:
    os.mkdir(resized_dir)
    os.mkdir(os.path.join(resized_dir, "train_dataset"))
    os.mkdir(os.path.join(resized_dir, "test_dataset"))
    os.mkdir(os.path.join(resized_dir, "experimental_dataset"))
    train_files = [f for f in os.listdir(os.path.join(dataset_dir, "train_dataset")) if f.endswith('.mp4')]
    test_files = [f for f in os.listdir(os.path.join(dataset_dir, "test_dataset")) if f.endswith('.mp4')]
    experimental_files = [f for f in os.listdir(os.path.join(dataset_dir, "experimental_dataset")) if f.endswith('.mp4')]
    def resize(in_video_path, out_video_path, nb_frames=10):
        video = extract_frames(in_video_path, nb_frames=nb_frames)
        t1 = time.time()
        #video, audio, info = io.read_video(in_video_path, pts_unit='sec', start_pts=0, end_pts=10, output_format='TCHW')
        video = smart_resize(video, 256)
        t2 = time.time()
        torch.save(video, out_video_path)
        t3 = time.time()
        print(f"resize: {t2-t1}\nsave: {t3-t2}")
        #video = video.permute(0,2,3,1)
        #io.write_video(video_path, video, 15, video_codec='h264')

    
    for f in tqdm(train_files):
        in_video_path = os.path.join(dataset_dir, "train_dataset", f)
        out_video_path = os.path.join(resized_dir, "train_dataset", f[:-3] + "pt")
        try:
            resize(in_video_path, out_video_path)
        except Exception as e:
            errors.append((f, e))
        print(f"resized {f} from train")
    
    for f in tqdm(test_files):
        in_video_path = os.path.join(dataset_dir, "test_dataset", f)
        out_video_path = os.path.join(resized_dir, "test_dataset", f[:-3] + "pt")
        try:
            resize(in_video_path, out_video_path)
        except Exception as e:
            errors.append((f, e))
        print(f"resized {f} from test")
    for f in tqdm(experimental_files):
        in_video_path = os.path.join(dataset_dir, "experimental_dataset", f)
        out_video_path = os.path.join(resized_dir, "experimental_dataset", f[:-3] + "pt")
        try:
            resize(in_video_path, out_video_path)
        except Exception as e:
            errors.append((f, e))
        print(f"resized {f} from experimental")
    os.system(f"cp {os.path.join(dataset_dir, 'train_dataset', 'metadata.json')} {os.path.join(resized_dir, 'train_dataset', 'metadata.json')}")
    os.system(f"cp {os.path.join(dataset_dir, 'dataset.csv')} {os.path.join(resized_dir, 'dataset.csv')}")
    os.system(f"cp {os.path.join(dataset_dir, 'experimental_dataset', 'metadata.json')} {os.path.join(resized_dir, 'experimental_dataset', 'metadata.json')}")
    if errors:
        print(errors)
"""
use_small_dataset = True
if use_small_dataset:
    dataset_dir = resized_dir

nb_frames = 10

class VideoDataset(Dataset):
    """
    This Dataset takes a video and returns a tensor of shape [10, 3, 256, 256]
    That is 10 colored frames of 256x256 pixels.
    """
    def __init__(self, root_dir, dataset_choice="train", nb_frames=10):
        super().__init__()
        self.dataset_choice = dataset_choice
        if  self.dataset_choice == "train":
            self.root_dir = os.path.join(root_dir, "train_dataset")
        elif  self.dataset_choice == "test":
            self.root_dir = os.path.join(root_dir, "test_dataset")
        elif  self.dataset_choice == "experimental":
            self.root_dir = os.path.join(root_dir, "experimental_dataset")
        else:
            raise ValueError("choice must be 'train', 'test' or 'experimental'")

        with open(os.path.join(root_dir, "dataset.csv"), 'r') as file:
            reader = csv.reader(file)
            # read dataset.csv with id,label columns to create
            # a dict which associated label: id
            self.ids = {row[1][:-3] + "pt" : row[0] for row in reader}

        if self.dataset_choice == "test":
            self.data = None
        else:
            with open(os.path.join(self.root_dir, "metadata.json"), 'r') as file:
                self.data= json.load(file)
                self.data = {k[:-3] + "pt" : (torch.tensor(float(1)) if v == 'fake' else torch.tensor(float(0))) for k, v in self.data.items()}

        #self.video_files = [f for f in os.listdir(self.root_dir) if f.endswith('.mp4')]
        self.video_files = [f for f in os.listdir(self.root_dir) if f.endswith('.pt')]

    def __len__(self):
        return len(self.video_files)

    def __getitem__(self, idx):
        video_path = os.path.join(self.root_dir, self.video_files[idx])
        #video, audio, info = io.read_video(video_path, pts_unit='sec')
        #video = extract_frames(video_path)
        
        '''
        Ici on va lire la vidéo et extraire les têtes, puis les redimensionner en 64x64
        '''
        
        
        
        video = torch.load(video_path)

        """
        video = video.permute(0,3,1,2)
        length = video.shape[0]
        video = video[[i*(length//(nb_frames)) for i in range(nb_frames)]]
        """
        # resize the data into a reglar shape of 256x256 and normalize it
        #video = smart_resize(video, 256) / 255
        video = video / 255

        ID = self.ids[self.video_files[idx]]
        if self.dataset_choice == "test":
            return video, ID
        else:
            label = self.data[self.video_files[idx]]
            return video, label, ID



train_dataset = VideoDataset(dataset_dir, dataset_choice="train", nb_frames=nb_frames)
test_dataset = VideoDataset(dataset_dir, dataset_choice="test", nb_frames=nb_frames)
experimental_dataset = VideoDataset(dataset_dir, dataset_choice="experimental", nb_frames=nb_frames)


# MODELE

#On va équilibrer le dataset
#print(list(train_dataset.data.values()))
print("Nombre de deepake:",sum([int(torch.unsqueeze(k,dim=0)) for k in list(train_dataset.data.values())]))
print("Nombre d'images:",len(train_dataset.data))

#On va vérifier qu'on a bien des images différentes
video1 = train_dataset[0][0]
for k in range(len(train_dataset)):
    video2 = train_dataset[k][0]
    if torch.all(video1 == video2):
        print("Vidéos identiques")
        
    
        


class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0):
        """
        Args:
            patience (int): Combien d'époques à attendre après la dernière fois où la performance a été améliorée.
                            Le compteur d'attente est remis à zéro après une amélioration.
            verbose (bool): Si vrai, imprime un message à chaque amélioration.
            delta (float): Seuil minimal d'amélioration pour considérer qu'il y a une amélioration.
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = float('inf')
        self.delta = delta

    def __call__(self, val_loss, model):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0
            self.val_loss_min = val_loss
            # Sauvegardez le meilleur modèle
            torch.save(model.state_dict(), 'checkpoint.pt')
            if self.verbose:
                print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')

class CNNVideoClassifier(nn.Module):
    def __init__(self, num_classes):
        super(CNNVideoClassifier, self).__init__()
        self.conv1 = nn.Conv3d(10, 16, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
        self.bn1 = nn.BatchNorm3d(16)
        self.pool1 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))
        
        self.conv2 = nn.Conv3d(16, 32, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
        self.bn2 = nn.BatchNorm3d(32)
        self.pool2 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))
        
        
        self.conv3 = nn.Conv3d(32, 64, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
        self.bn3 = nn.BatchNorm3d(64)
        self.pool3 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))

        # The output from the last pooling layer will have dimensions [batch_size, 64, 10, 32, 32]
        self.fc1 = nn.Linear(64*32*32*3, 512)
        self.fc2 = nn.Linear(512, num_classes-1)

        self.relu = nn.LeakyReLU()
        
        self.drop = nn.Dropout(0.2)
        
        self.sigmoid = nn.Sigmoid()
        
        self.l1_lambda = 0.0005

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.pool3(x)
        
        # Flatten the output for the fully connected layer
        x = x.view(x.size(0), -1)  # Flatten preserving the batch dimension
        x = self.relu(self.fc1(x))
        x = self.drop(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x
    
    def l1_loss(self):
        l1_reg = torch.tensor(0., requires_grad=True)
        for name, param in self.named_parameters():
            if 'weight' in name:
                l1_reg = l1_reg + torch.norm(param, p=1)
        return self.l1_lambda * l1_reg




# Assuming you have your video dataset loaded into DataLoader
# for batch_idx, (inputs, targets) in enumerate(train_loader):
#     optimizer.zero_grad()
#     outputs = model(inputs)
#     loss = criterion(outputs, targets)
#     loss.backward()
#     optimizer.step()
'''
class DeepfakeDetector(nn.Module):
    def __init__(self, nb_frames=10):
        super().__init__()
        self.dense = nn.Linear(nb_frames*3*256*256,1)
        self.flat = nn.Flatten()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.flat(x)
        y = self.dense(y)
        y = self.sigmoid(y)
        return y
'''
# LOGGING

wandb.login(key="09f8b3f5e76020d7c21ddcfc5f20869c998b143e")

run = wandb.init(
    project="PVP"
)

# ENTRAINEMENT

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Example usage:
# Define your video classification model
model = CNNVideoClassifier(num_classes=2).to(device)  # Change num_classes according to your dataset
# Define loss function and optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
batch_size = 32
loss_fn = nn.CrossEntropyLoss()#nn.MSELoss()
#model = DeepfakeDetector().to(device)
print("Training model:")
summary(model, input_size=(batch_size, 10, 3, 256, 256))
epochs = 2
loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

early_stopping = EarlyStopping(patience=3, verbose=True)

'''
for p in model.parameters():
    p.requires_grad = False'''
#loader = DataLoader(experimental_dataset, batch_size=2, shuffle=True)

print("Training...")
for epoch in range(epochs):
    for sample in tqdm(loader):
        optimizer.zero_grad()
        X, label, ID = sample
        #print("X_shape:",X.shape)
        X = X.to(device)
        label = label.to(device)
        label_pred = model(X)
        label = torch.unsqueeze(label,dim=1)
        loss = loss_fn(label, label_pred) #+ model.l1_loss()
        loss.backward()
        optimizer.step()
        run.log({"loss": loss.item(), "epoch": epoch,"L1 loss": model.l1_loss()})
        

## TEST

loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
model = model.to(device)
ids = []
labels = []
print("Testing...")
for sample in tqdm(loader):
    X, ID = sample
    #ID = ID[0]
    X = X.to(device)
    label_pred = model(X)
    ids.extend(list(ID))
    print("predicted_label:",label_pred)
    pred = (label_pred > 0.5).long()
    pred = pred.cpu().detach().numpy().tolist()
    labels.extend(pred)

### ENREGISTREMENT
print("Saving...")
tests = ["id,label\n"] + [f"{ID},{label_pred[0]}\n" for ID, label_pred in zip(ids, labels)]
with open("submission.csv", "w") as file:
    file.writelines(tests)
