#%%
import torch
import torch.nn as nn
from torchvision import datasets ,models,transforms
from torch.utils.data.sampler import SubsetRandomSampler
from torch.optim import lr_scheduler
from pathlib import Path
from matplotlib import pyplot as plt
import numpy as np
import torch.nn.functional as F
from torch.autograd import Variable
# from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from resnet import ResNet50
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()
#%%
train_on_gpu = torch.cuda.is_available()
if not train_on_gpu:
    print('CUDA is not available.  Training on CPU ...')
else:
    print('CUDA is available!  Training on GPU ...')



PATH_train="dataset\\train"
PATH_val="dataset\\val"
PATH_test="dataset\\test"
# %%
TRAIN =Path(PATH_train)
VALID = Path(PATH_val)
TEST=Path(PATH_test)
print(TRAIN)
print(VALID)
print(TEST)
# %% 
num_workers = 0
batch_size = 32
LR = 0.01

transforms.Normalize(mean=[0.5, 0.5, 0.5],std=[0.5, 0.5, 0.5])
train_transforms = transforms.Compose([transforms.Resize((224,224)),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])])

valid_transforms = transforms.Compose([transforms.Resize((64,64)),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])])

test_transforms = transforms.Compose([transforms.Resize((64,64)),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])])
# %%
train_data = datasets.ImageFolder(TRAIN, transform=train_transforms)
valid_data = datasets.ImageFolder(VALID,transform=valid_transforms)
test_data = datasets.ImageFolder(TEST, transform=test_transforms)
# %%
print(train_data.class_to_idx)
print(valid_data.class_to_idx)
# %%
train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, num_workers=num_workers,shuffle=True)
valid_loader = torch.utils.data.DataLoader(valid_data, batch_size=batch_size,  num_workers=num_workers,shuffle=True)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size,  num_workers=num_workers)
images,labels=next(iter(train_loader))
images.shape,labels.shape

# %% plot example image
model=models.resnet50()
from torchsummary import summary
summary(model.cuda(), (3, 224, 224))
# %%
from tqdm import tqdm_notebook as tqdm
if train_on_gpu:
    model.cuda()
# number of epochs to train the model
n_epochs = 10

valid_loss_min = np.Inf # track change in validation loss
optimizer=torch.optim.Adam(model.parameters(),lr=LR)
criterion = torch.nn.CrossEntropyLoss()
#train_losses,valid_losses=[],[]

for epoch in range(1, n_epochs+1):

    # keep track of training and validation loss
    train_loss = 0.0
    valid_loss = 0.0
    train_acc=0.0
    valid_acc=0.0
    print('running epoch: {}'.format(epoch))
    ###################
    # train the model #
    ###################
    model.train()
    for data, target in tqdm(train_loader):
        # move tensors to GPU if CUDA is available
        if train_on_gpu:
            data, target = data.cuda(), target.cuda()
        # clear the gradients of all optimized variables
        optimizer.zero_grad()
        # forward pass: compute predicted outputs by passing inputs to the model
        output = model(data)     
        # calculate the batch loss
        loss = criterion(output, target)
        
        # backward pass: compute gradient of the loss with respect to model parameters
        loss.backward()
        # perform a single optimization step (parameter update)
        optimizer.step()
        # update training loss
        train_loss += loss.item()*data.size(0)
        # calculate the train acc
        _, pred = torch.max(output, 1)
        num_correct = (pred == target).sum()
        train_acc += num_correct.item()

    writer.add_scalar("Loss/train", train_loss/len(train_data), epoch)
    writer.add_scalar("Acc/train", train_acc/len(train_data), epoch)
    ######################    
    # validate the model #
    ######################
    model.eval()
    for data, target in tqdm(valid_loader):
        # move tensors to GPU if CUDA is available
        if train_on_gpu:
            data, target = data.cuda(), target.cuda()
        # forward pass: compute predicted outputs by passing inputs to the model
        output = model(data)
        # calculate the batch loss
        loss = criterion(output, target)
        
        # update average validation loss 
        valid_loss += loss.item()*data.size(0)
        # calculate the train acc
        _, pred = torch.max(output, 1)
        num_correct = (pred == target).sum()
        valid_acc += num_correct.item()

    writer.add_scalar("Loss/valid", valid_loss/len(valid_data), epoch)
    writer.add_scalar("Acc/valid", valid_acc/len(valid_data), epoch)
    # calculate average losses
    #train_losses.append(train_loss/len(train_loader.dataset))
    #valid_losses.append(valid_loss.item()/len(valid_loader.dataset)
    train_loss = train_loss/len(train_loader.dataset)
    valid_loss = valid_loss/len(valid_loader.dataset)
        
    # print training/validation statistics 
    print('\tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(
        train_loss, valid_loss))
    print('\tTraining Acc: {:.6f}  \tValidation Acc: {:.6f}'.format(
        train_acc/len(train_data),valid_acc/len(valid_data)))
    
    # save model if validation loss has decreased
    if valid_loss <= valid_loss_min:
        print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(
        valid_loss_min,
        valid_loss))
        torch.save(model.state_dict(), 'model_CNN2.pth')
        valid_loss_min = valid_loss

writer.flush()
# %%
writer.close()