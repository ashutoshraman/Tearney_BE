import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.nn.functional import relu
import pandas as pd
import sklearn as sk
from PIL import Image
import os, sys
import cv2
from tqdm import tqdm
import seaborn as sns


class CustomImageDataset(Dataset):
    def __init__(self, images_dir, masks_dir, transform=None, target_transform=None):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.transform = transform
        self.target_transform = target_transform
        
        self.image_files = sorted(os.listdir(images_dir))
        self.mask_files = [f for f in sorted(os.listdir(masks_dir)) if f.endswith(('.png'))]

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

class ToLabel:
    def __init__(self, label_mappings):
        self.label_mapping = label_mappings

    def __call__(self, pic):
        label_array = np.array(pic, dtype= np.int32)
        for k, v in self.label_mapping.items():
            label_array[label_array == v] = k
        return torch.from_numpy(label_array).long() #delete long?
    
path_data = "/Users/ashutoshraman/Documents/repos/Tearney_BE/raw_data/images/"
path_masks = "/Users/ashutoshraman/Documents/repos/Tearney_BE/raw_data/annotations/"

msk = Image.open(path_masks+"11.png").convert('L')
msk.show() #not working for png files rn unless you use convert('L')

img = Image.open(path_data+"11.tif").convert('RGB')
img.show() #not working for png files rn unless you use convert('L')
def normalize_im(images, masks):
    pass


image_transforms = transforms.Compose([
    transforms.ToTensor(),
    # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    # transforms.Lambda(lambda x: x * 255.0),  # Scale to [0, 255] from [0, 1]
])

label_mapping = {0: 0, 1: 150, 2: 76} # darker is BE, lighter or whiter is healthy

mask_transforms = transforms.Compose([
    # transforms.ToTensor(), # commented this out but now i  have index 150 out of range, try one hot encoding
    ToLabel(label_mapping)
])

transform = mask_transforms(msk)
print(f"uniques are {torch.unique(transform)}")
transform_img = image_transforms(img)
print(f"uniques are {torch.unique(transform_img)}")


dataset = CustomImageDataset(path_data, path_masks, transform=image_transforms, target_transform=mask_transforms)

train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])


train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True) #num_workers depends on num CPU cores (10 in M2Pro) and num per batch
val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False)
#for some reason this need to be batch size of 2 otherwise errors out
device=torch.device("mps" if torch.backends.mps.is_available() else "cpu")
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

#create u-net here
class UNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        
        # Encoder
        # In the encoder, convolutional layers with the Conv2d function are used to extract features from the input image. 
        # Each block in the encoder consists of two convolutional layers followed by a max-pooling layer, with the exception of the last block which does not include a max-pooling layer.
        # -------
        # input: 572x572x3
        self.e11 = nn.Conv2d(in_channels, 64, kernel_size=3, padding=1) # output: 570x570x64
        self.e12 = nn.Conv2d(64, 64, kernel_size=3, padding=1) # output: 568x568x64
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2) # output: 284x284x64

        # input: 284x284x64
        self.e21 = nn.Conv2d(64, 128, kernel_size=3, padding=1) # output: 282x282x128
        self.e22 = nn.Conv2d(128, 128, kernel_size=3, padding=1) # output: 280x280x128
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2) # output: 140x140x128

        # input: 140x140x128
        self.e31 = nn.Conv2d(128, 256, kernel_size=3, padding=1) # output: 138x138x256
        self.e32 = nn.Conv2d(256, 256, kernel_size=3, padding=1) # output: 136x136x256
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2) # output: 68x68x256

        # input: 68x68x256
        self.e41 = nn.Conv2d(256, 512, kernel_size=3, padding=1) # output: 66x66x512
        self.e42 = nn.Conv2d(512, 512, kernel_size=3, padding=1) # output: 64x64x512
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2) # output: 32x32x512

        # input: 32x32x512
        self.e51 = nn.Conv2d(512, 1024, kernel_size=3, padding=1) # output: 30x30x1024
        self.e52 = nn.Conv2d(1024, 1024, kernel_size=3, padding=1) # output: 28x28x1024


        # Decoder
        self.upconv1 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.d11 = nn.Conv2d(1024, 512, kernel_size=3, padding=1)
        self.d12 = nn.Conv2d(512, 512, kernel_size=3, padding=1)

        self.upconv2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.d21 = nn.Conv2d(512, 256, kernel_size=3, padding=1)
        self.d22 = nn.Conv2d(256, 256, kernel_size=3, padding=1)

        self.upconv3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.d31 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.d32 = nn.Conv2d(128, 128, kernel_size=3, padding=1)

        self.upconv4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.d41 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.d42 = nn.Conv2d(64, 64, kernel_size=3, padding=1)

        # Output layer
        self.outconv = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        # Encoder
        xe11 = relu(self.e11(x))
        xe12 = relu(self.e12(xe11))
        xp1 = self.pool1(xe12)

        xe21 = relu(self.e21(xp1))
        xe22 = relu(self.e22(xe21))
        xp2 = self.pool2(xe22)

        xe31 = relu(self.e31(xp2))
        xe32 = relu(self.e32(xe31))
        xp3 = self.pool3(xe32)

        xe41 = relu(self.e41(xp3))
        xe42 = relu(self.e42(xe41))
        xp4 = self.pool4(xe42)

        xe51 = relu(self.e51(xp4))
        xe52 = relu(self.e52(xe51))
        
        # Decoder
        xu1 = self.upconv1(xe52)
        xu11 = torch.cat([xu1, xe42], dim=1)
        xd11 = relu(self.d11(xu11))
        xd12 = relu(self.d12(xd11))

        xu2 = self.upconv2(xd12)
        xu22 = torch.cat([xu2, xe32], dim=1)
        xd21 = relu(self.d21(xu22))
        xd22 = relu(self.d22(xd21))

        xu3 = self.upconv3(xd22)
        xu33 = torch.cat([xu3, xe22], dim=1)
        xd31 = relu(self.d31(xu33))
        xd32 = relu(self.d32(xd31))

        xu4 = self.upconv4(xd32)
        xu44 = torch.cat([xu4, xe12], dim=1)
        xd41 = relu(self.d41(xu44))
        xd42 = relu(self.d42(xd41))

        # Output layer
        out = self.outconv(xd42)

        return out

#instantiate here
num_classes = 3 # Number of classes in your dataset
model = UNet(3, 3)
model = model.to(device)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
num_epochs = 10

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
            ious.append((intersection / union).cpu())
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
        outputs = model(inputs) #you added out and 0, just took out 0
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

            outputs = model(inputs) #you added out and 0, just took out 0
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
        outputs = ml_model(inputs)

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
