import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import pandas as pd
import sklearn as sk
from PIL import Image
import os, sys
import cv2
from tqdm import tqdm
import seaborn as sns



class CustomDataset(Dataset):
    def __init__(self, imagePath, maskPath, transforms):
        self.imagePath = imagePath
        self.maskPath = maskPath
        self.transforms = transforms

    def __len__(self):
        return len(self.imagePath)
    
    def __getitem__(self, idx):
        imagePath = self.imagePath[idx]
        image = cv2.imread(imagePath)
        mask = cv2.imread(self.maskPath[idx], 0)

        if self.transforms is not None:
            image = self.transforms(image)
            mask = self.transforms(mask)
        return (image, mask)
    

# input file paths for directories, output image and mask transformed

class CustomImageDataset(Dataset):
    def __init__(self, images_dir, masks_dir, transform=None, target_transform=None):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.transform = transform
        self.target_transform = target_transform
        self.image_files = sorted(os.listdir(images_dir))
        self.mask_files = sorted(os.listdir(masks_dir))

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.images_dir, self.image_files[idx])
        mask_path = os.path.join(self.masks_dir, self.mask_files[idx])

        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")  # Assuming masks are in grayscale

        # Image.open(os.path.join(self.masks_dir, self.mask_files[idx])).convert("L")

        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            mask = self.target_transform(mask).squeeze(0) # you squeezed target/mask/label dimension to get rid of singleton dimension

        return image, mask




path_data = "/Users/ashutoshraman/Documents/repos/Tearney_BE/raw_data/images/"
path_masks = "/Users/ashutoshraman/Documents/repos/Tearney_BE/raw_data/annotations/"

img = Image.open(path_masks+"117.png").convert('L')
img.show() #not working for png files rn unless you use convert('L')


# img = Image.open(path_data+"10.tif").convert('RGB')
# img.show() #not working for png files rn unless you use convert('L')


image_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


class ToLabel:
    def __init__(self, label_mappings):
        self.label_mapping = label_mappings

    def __call__(self, pic):
        label_array = np.array(pic, dtype= np.int32)
        for k, v in self.label_mapping.items():
            label_array[label_array == v] = k
        return torch.from_numpy(label_array).long() #delete long?
    
label_mapping = {0: 0, 1: 150, 2: 76}

mask_transforms = transforms.Compose([
    # transforms.ToTensor(), # commented this out but now i  have index 150 out of range, try one hot encoding
    ToLabel(label_mapping)
])

transform = mask_transforms(img)
print(f"uniques are {torch.unique(transform)}")


dataset = CustomImageDataset(path_data, path_masks, transform=image_transforms, target_transform=mask_transforms)

train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])


train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True) #num_workers depends on num CPU cores (10 in M2Pro) and num per batch
val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)
#for some reason this need to be batch size of 2 otherwise errors out
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load pre-trained FCN model

model = torch.hub.load('pytorch/vision:v0.10.0', 'fcn_resnet50', pretrained=True) #change to false and train your own data
###FCN is an ordered dict of output loss and masks, also need to have if statement of if cuda exists
# model = torch.hub.load('pytorch/vision:v0.10.0', 'fcn_resnet101', pretrained=True)

# model.eval()
#model = FCN(pretrained=True)  # Load pre-trained model
# print(model)
# Modify the last layer of the model to match the number of classes in your dataset
num_classes = 3 # Number of classes in your dataset
model.classifier[4] = nn.Conv2d(512, num_classes, kernel_size=1)
model = model.to(device)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
num_epochs = 3

def calculate_iou(y_pred, y_real, classes):
    ious = []
    pred = y_pred.view(-1)
    target = y_real.view(-1)

    for cls in range(classes):
        pred_inds = pred == cls
        target_inds = target == cls
        intersection = (pred_inds[target_inds]).sum().float()
        union = pred_inds.sum() + target_inds.sum() - intersection
        if union == 0:
            ious.append(float('nan'))  # If there is no union, ignore this class
        else:
            ious.append(intersection / union)
    return np.nanmean(ious)


accuracy_stats = {
    "train": [],
    "val": []
}
loss_stats = {
    "train": [],
    "val": []
}
iou_stats = {
    "train": [],
    "val": []
}


# Training loop
for epoch in tqdm(range(num_epochs)):
    train_epoch_loss = 0
    train_epoch_acc = 0
    train_correct = 0
    train_total = 0
    train_iou = 0
    model.train()
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        print(f"Input shape: {inputs.shape}, Label shape: {labels.shape}")
        print(f"Unique values in label: {torch.unique(labels)}") #issue is here


        optimizer.zero_grad()
        outputs = model(inputs)['out'] #you added out and 0, just took out 0
        print('\n')
        print(f"Output shape: {outputs.shape}, Labels shape: {labels.shape}")
        
        train_loss = criterion(outputs, labels)
        # print(train_loss)
        train_loss.backward()
        optimizer.step()

        _, predicted = torch.max(outputs, 1)
        train_total += labels.numel()
        train_correct += (predicted == labels).sum().item()
        train_epoch_acc += train_correct/ train_total
        train_iou += calculate_iou(predicted, labels, num_classes)

        #calculate accuracy and loss and store
        train_epoch_loss += train_loss.item()
        
    # Validation loop
    model.eval()
    val_loss = 0.0
    val_iou = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            print(f"Input shape: {inputs.shape}, Label shape: {labels.shape}")

            outputs = model(inputs)['out'] #you added out and 0, just took out 0
            print('\n')
            print(f"Output shape: {outputs.shape}, Labels shape: {labels.shape}")
            
            val_loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs, 1)
            total += labels.numel()
            correct += (predicted == labels).sum().item()
            val_iou += calculate_iou(predicted, labels, num_classes)
    
    val_loss /= len(val_loader)
    val_acc = correct / total
    val_iou /= len(val_loader)

    #store accuracy and loss
    

    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {val_loss}, Accuracy: {val_acc}")

    # store in array
    loss_stats['train'].append(train_epoch_loss/len(train_loader))
    loss_stats['val'].append(val_loss)
    accuracy_stats['train'].append(train_epoch_acc/ len(train_loader))
    accuracy_stats['val'].append(val_acc)
    iou_stats['train'].append(train_iou / len(train_loader))
    iou_stats['val'].append(val_iou)





train_val_acc_df = pd.DataFrame.from_dict(accuracy_stats).reset_index().melt(id_vars=['index']).rename(columns={"index":"epochs"})
train_val_loss_df = pd.DataFrame.from_dict(loss_stats).reset_index().melt(id_vars=['index']).rename(columns={"index":"epochs"})
train_val_iou_df = pd.DataFrame.from_dict(iou_stats).reset_index().melt(id_vars=['index']).rename(columns={"index":"epochs"})
# Plot the dataframes
fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(20,7))
sns.lineplot(data=train_val_loss_df, x = "epochs", y="value", hue="variable", ax=axes[0]).set_title('Train-Val Loss/Epoch')
sns.lineplot(data=train_val_acc_df, x = "epochs", y="value", hue="variable",  ax=axes[1]).set_title('Train-Val Accuracy/Epoch')
sns.lineplot(data=train_val_iou_df, x = "epochs", y="value", hue="variable",  ax=axes[2]).set_title('Train-Val IoU/Epoch')

plt.show()

print(train_val_iou_df.iloc[-1])

def plot_masks(ml_model, data_loader, class_num):
    ml_model.eval()
    with torch.no_grad():
        inputs, labels = next(iter(data_loader))
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = ml_model(inputs)['out']

        _, preds = torch.max(outputs, 1)

        fig, axs = plt.subplots(3, len(inputs), figsize=(15,5))

        for i in range(len(inputs)):

            # image
            img = inputs[i].cpu().numpy().transpose((1,2,0)) # do i need to squeeze dimensions?
            img = np.clip(img * np.array([.229, .224, .225]) + np.array([.485, .456, .406]), 0, 1)
            axs[0, i].imshow(img)
            axs[0, i].set_title('Original image')
            axs[0, i].axis('off')

            # ground truth mask
            true_mask = labels[i].cpu().numpy()
            axs[1, i].imshow(true_mask, cmap='gray', vmin=0, vmax=class_num-1)
            axs[1, i].set_title('Ground Truth')
            axs[1, i].axis('off')

            # predicted mask
            pred_mask = preds[i].cpu().numpy()
            axs[2, i].imshow(pred_mask, cmap='gray', vmin=0, vmax=class_num-1)
            axs[2, i].set_title('Predictions')
            axs[2, i].axis('off')

        plt.tight_layout()
        plt.show()

plot_masks(model, train_loader, num_classes)



# calculate Dice, (over epochs?)
# plot worst masks, best masks