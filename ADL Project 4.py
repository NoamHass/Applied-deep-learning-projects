# %% [markdown]
# Question 1

# %%
import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

batch_size = 1

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                        download=True, transform=transform)

testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# %%
import matplotlib.pyplot as plt
import numpy as np

# functions to show an image


def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


# get some random training images
dataiter = iter(trainloader)
images, labels = next(dataiter)

# show images
imshow(torchvision.utils.make_grid(images))
# print labels
print(' '.join(f'{classes[labels[j]]:5s}' for j in range(batch_size)))

# %%
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


net = Net()

# %%
import torch.optim as optim

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)


# %%
for epoch in range(2):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 2000 == 1999:    # print every 2000 mini-batches
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
            running_loss = 0.0

print('Finished Training')

# %%
PATH = './cifar_net.pth'
torch.save(net.state_dict(), PATH)

# %%
dataiter = iter(testloader)
images, labels = next(dataiter)

# print images
imshow(torchvision.utils.make_grid(images))
print('GroundTruth: ', ' '.join(f'{classes[labels[j]]:5s}' for j in range(4)))

# %%
net = Net()
net.load_state_dict(torch.load(PATH))

# %%
outputs = net(images)

_, predicted = torch.max(outputs, 1)

print('Predicted: ', ' '.join(f'{classes[predicted[j]]:5s}'
                              for j in range(4)))

# %%
correct = 0
total = 0
# since we're not training, we don't need to calculate the gradients for our outputs
with torch.no_grad():
    for data in testloader:
        images, labels = data
        # calculate outputs by running images through the network
        outputs = net(images)
        # the class with the highest energy is what we choose as prediction
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Accuracy of the network on the 10000 test images: {100 * correct // total} %')

# %%
# prepare to count predictions for each class
correct_pred = {classname: 0 for classname in classes}
total_pred = {classname: 0 for classname in classes}

# again no gradients needed
with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = net(images)
        _, predictions = torch.max(outputs, 1)
        # collect the correct predictions for each class
        for label, prediction in zip(labels, predictions):
            if label == prediction:
                correct_pred[classes[label]] += 1
            total_pred[classes[label]] += 1


# print accuracy for each class
for classname, correct_count in correct_pred.items():
    accuracy = 100 * float(correct_count) / total_pred[classname]
    print(f'Accuracy for class: {classname:5s} is {accuracy:.1f} %')

# %%
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Assuming that we are on a CUDA machine, this should print a CUDA device:

print(device)

# %% [markdown]
# Question 2

# %%
#Adapting the network to include 2 deconvolutional layers 

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2, return_indices=True)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

        self.deconv1 = nn.ConvTranspose2d(16, 6, 5)
        self.deconv2 = nn.ConvTranspose2d(6, 3, 5)
        self.unpool = nn.MaxUnpool2d(2, 2)

    def forward(self, x):
        x, indeices1 = self.pool(F.relu(self.conv1(x)))
        z1 = [x.clone(), indeices1]

        x, indeices2 = self.pool(F.relu(self.conv2(x)))
        z2 = [x.clone(), indeices2]

        deconvol = x.clone()
        deconvol = self.deconv1(F.relu(self.unpool(deconvol, indeices2)))
        deconvol = self.deconv2(F.relu(self.unpool(deconvol, indeices1)))

        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x, deconvol, z1, z2
    
net = Net()


# %%
#Define the loss function to include the reconstruction loss

def criterion(outputs, labels, deconvol, inputs, lambda_weight):
    loss_ce = nn.CrossEntropyLoss()
    loss_rec = nn.MSELoss()
    return loss_ce(outputs, labels) + lambda_weight * loss_rec(deconvol, inputs)

#Define the optimizer

optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

# %%
#train the modified network

for epoch in range(2):  # loop over the dataset multiple times
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
    
            # zero the parameter gradients
            optimizer.zero_grad()
    
            # forward + backward + optimize
            outputs, deconvol, z1, z2 = net(inputs)
            loss = criterion(outputs, labels, deconvol, inputs, 1)
            loss.backward()
            optimizer.step()
    
            # print statistics
            running_loss += loss.item()
            if i % 2000 == 1999:    # print every 2000 mini-batches
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
                running_loss = 0.0

print('Finished Training')

# %%
#Reporting the classification error in accuracy of the test set

correct = 0
total = 0
# since we're not training, we don't need to calculate the gradients for our outputs
with torch.no_grad():
    for data in testloader:
        images, labels = data
        # calculate outputs by running images through the network
        outputs, _, _, _ = net(images)
        # the class with the highest energy is what we choose as prediction
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Accuracy of the network on the 10000 test images: {100 * correct // total} %')

# %%
#Displaying 3 reconstructed images along with the original images

dataiter = iter(testloader)
images, labels = next(dataiter)

outputs, deconvol, _, _ = net(images)


# print images
imshow(torchvision.utils.make_grid(images))

# print reconstructed images
imshow(torchvision.utils.make_grid(deconvol))

# %% [markdown]
# Question 3

# %%

def leave_one_channel_on(tensor, channel_idx):
    zeroed_tensor = tensor.clone()
    zeroed_tensor = torch.zeros_like(zeroed_tensor)
    zeroed_tensor[channel_idx, :, :] = tensor[channel_idx, :, :]
    return zeroed_tensor

    

# %%
#Using the trained model on one image from the train set and one image from the test set

train_dataiter = iter(trainloader)
test_dataiter = iter(testloader)

images_test, labels_test = next(test_dataiter)
images_train, labels_train = next(train_dataiter)

_, _, z1_test, z2_test = net(images_test)
_, _, z1_train, z2_train = net(images_train)

z1_test = z1_test[0][0,:,:,:]
z2_test = z2_test[0][0,:,:,:]
z1_train = z1_train[0][0,:,:,:]
z2_train = z2_train[0][0,:,:,:]


imshow(torchvision.utils.make_grid(images_test))
imshow(torchvision.utils.make_grid(images_train))





# %%
deconv2 = net.deconv1
deconv1 = net.deconv2
unpool = net.unpool

# %%
reconstructed_images_z1_train = []
for i in range(6) :
    z1_train_copy = z1_train.clone()
    z1_train_copy.data = leave_one_channel_on(z1_train_copy.data, i)
    # reconstruct
    z1_train_copy, indices = net.pool(F.relu(net.conv2(z1_train_copy)))
    deconv = deconv2(F.relu(unpool(z1_train_copy, indices)))
    deconv = deconv1(F.relu(unpool(deconv, z1_train_indices)))
    reconstructed_images_z1_train.append(deconv)

# %%
grid_images = []
for i in range(6):
    img = torchvision.utils.make_grid(reconstructed_images_z1_train[i])
    grid_images.append(img)

combined_image = torchvision.utils.make_grid(grid_images, nrow=len(reconstructed_images_z1_train))

imshow(combined_image)

# %%
reconstructed_images_z1_test = []
for i in range(6) :
    z1_test_copy = z1_test.clone()
    z1_test_copy.data = leave_one_channel_on(z1_test_copy.data, i)
    # reconstruct
    z1_test_copy, indices = net.pool(F.relu(net.conv2(z1_test_copy)))
    deconv = deconv2(F.relu(unpool(z1_test_copy, indices)))
    deconv = deconv1(F.relu(unpool(deconv, z1_test_indices)))
    reconstructed_images_z1_test.append(deconv)

# %%

grid_images = []
for i in range(6):
    img = torchvision.utils.make_grid(reconstructed_images_z1_test[i])
    grid_images.append(img)

combined_image = torchvision.utils.make_grid(grid_images, nrow=len(reconstructed_images_z1_test))

imshow(combined_image)

# %%
reconstructed_images_z2_train = []
for i in range(3) :
    z2_train_copy = z2_train.clone()
    z2_train_copy.data = leave_one_channel_on(z2_train_copy.data, i)
    # reconstruct
    deconv = deconv2(F.relu(unpool(z2_train_copy, z2_train_indices)))
    deconv = deconv1(F.relu(unpool(deconv, z1_train_indices)))
    reconstructed_images_z2_train.append(deconv)

# %%
grid_images = []
for i in range(len(reconstructed_images_z2_train)):
    img = torchvision.utils.make_grid(reconstructed_images_z2_train[i])
    grid_images.append(img)

combined_image = torchvision.utils.make_grid(grid_images, nrow=len(reconstructed_images_z2_train))

imshow(combined_image)

# %%
reconstructed_images_z2_test = []
for i in range(len(samples)) :
    z2_test_copy = z2_test.clone()
    z2_test_copy.data = leave_one_channel_on(z2_test_copy.data, i)
    # reconstruct
    deconv = deconv2(F.relu(unpool(z2_test_copy, z2_test_indices)))
    deconv = deconv1(F.relu(unpool(deconv, z1_test_indices)))
    reconstructed_images_z2_test.append(deconv)

# %%
# Create a combined grid image
grid_images = []
for i in range(len(reconstructed_images_z2_test)):
    img = torchvision.utils.make_grid(reconstructed_images_z2_test[i])
    grid_images.append(img)

combined_image = torchvision.utils.make_grid(grid_images, nrow=len(reconstructed_images_z2_test))

imshow(combined_image)


