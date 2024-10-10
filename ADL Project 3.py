
import torch 
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Subset
import numpy as np

# Setup device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# %% [markdown]
# Question 1

# %%
# Hyperparameters
input_size = 784
hidden_size = 500
num_classes = 10
num_epochs = 5
batch_size = 100
learning_rate = 0.001

# %%
# MNIST dataset
train_dataset = torchvision.datasets.MNIST(root='./data/',
										   train=True,
										   transform=transforms.ToTensor(),
										   download=True)

test_dataset = torchvision.datasets.MNIST(root='./data/',
										  train=False,
										  transform=transforms.ToTensor())

# Data loader
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
										   batch_size=batch_size,
										   shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
										  batch_size=batch_size,
										  shuffle=False)



# Using only 10% of the dataset for manageable runtime
# Calculate 10% of the dataset length
train_size = len(train_dataset)
test_size = len(test_dataset)
train_subset_size = int(0.1 * train_size)
test_subset_size = int(0.1 * test_size)

# Create subset datasets
train_subset = torch.utils.data.Subset(train_dataset, indices=range(train_subset_size))
test_subset = torch.utils.data.Subset(test_dataset, indices=range(test_subset_size))

# Create data loaders for subsets
train_loader = torch.utils.data.DataLoader(dataset=train_subset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_subset, batch_size=batch_size,shuffle=False)

# %%
#Implementing a basic 2-layer fully connected fully connected neural network
class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out
    
    
validation_size = 1000


validation_dataset = torch.utils.data.Subset(train_dataset, indices=np.random.choice(len(train_dataset), validation_size, replace=False))

validation_loader = torch.utils.data.DataLoader(dataset=validation_dataset, batch_size=batch_size, shuffle=True)

# %%
model = NeuralNet(input_size, hidden_size, num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# %%
#Implementing a training procedure
def train_procedure(model : NeuralNet, criterion, optimizer):
    model.train()
    for images , labels in train_loader:
        images = images.reshape(-1, input_size).to(device)
        labels = labels.to(device)
        
        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


# %%
#Implementing an evaluation procedure
def evaluation_procedure(model : NeuralNet , dataloader, criterion):
    model.eval()
    running_loss = 0.0
    running_accuracy = 0.0
    with torch.no_grad():
        for images, labels in dataloader :
            images = images.reshape(-1, input_size).to(device)
            labels = labels.to(device)

            output = model(images)
            loss = criterion(output,labels)

            _, predicted = torch.max(output.data, 1)
            running_loss += loss.item()
            running_accuracy += (predicted == labels).sum().item()

    epoch_loss = running_loss / len(dataloader)
    if(dataloader == train_loader ):
        epoch_accuracy = 100.0 * (running_accuracy / len(train_subset))
    if(dataloader == test_loader ):
        epoch_accuracy = 100.0 * (running_accuracy / len(test_subset))
    if(dataloader == validation_loader):
        epoch_accuracy = 100.0 * (running_accuracy / len(validation_dataset))    

    return epoch_loss , epoch_accuracy        

# %%
def run_model(model : NeuralNet, train_loss_history, train_accuracy_history, test_loss_history, test_accuracy_history, criterion, optimizer):
    train_loss , train_accuracy = evaluation_procedure(model, train_loader, criterion)
    test_loss , test_accuracy = evaluation_procedure(model, test_loader, criterion)

    train_loss_history.append(train_loss)
    train_accuracy_history.append(train_accuracy)

    test_loss_history.append(test_loss)
    test_accuracy_history.append(test_accuracy)

# %%
def question1(model : NeuralNet, citerion, optimizer):

    train_loss_history  = []
    train_accuracy_history = []
    test_loss_history = []
    test_accuracy_history = []

    for epoch in range(num_epochs):
        train_procedure(model, criterion, optimizer)
        run_model(model, train_loss_history, train_accuracy_history, test_loss_history, test_accuracy_history, criterion, optimizer)

    return train_loss_history, test_loss_history, train_accuracy_history, test_accuracy_history    


# %%
train_loss, test_loss, train_accuracy, test_accuracy = question1(model, criterion, optimizer)

for i in range(num_epochs):
    print(f'Epoch [{i+1}/{num_epochs}], Train Loss: {train_loss[i]:.4f}, Train Acc: {train_accuracy[i]:.2f}%, Test Loss: {test_loss[i]:.4f}, Test Acc: {test_accuracy[i]:.2f}%')

# %%
#Plotting the train and test error graphs
plt.plot(train_loss, label='Train Loss')
plt.plot(test_loss, label='Test Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Train and Test error')
plt.legend()
plt.show()

#Test error when training has finished
print(f'Test error when training finished: {test_loss[-1]:.4f} ')

# %%
#Finding misclassified images
def misclassified_images(num_of_images):
    misclassified_images = []
    misclassified_labels = []
    misclassified_predictions = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.reshape(-1, 28*28).to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            
            misclassified_inds = (predicted != labels).nonzero()
            for mis_ind in misclassified_inds:
                misclassified_images.append(images[mis_ind].cpu().numpy().reshape(28, 28))
                misclassified_labels.append(labels[mis_ind].item())
                misclassified_predictions.append(predicted[mis_ind].item())
                
            if len(misclassified_images) >= num_of_images:  
                break

    return misclassified_images, misclassified_labels, misclassified_predictions

# %%
# Plot the misclassified images
fig, axs = plt.subplots(2, 5, figsize=(10, 6))
fig.suptitle('Misclassified Images', fontsize=14)
fig.subplots_adjust(top=0.85)

misclassified_images, misclassified_labels, misclassified_predictions = misclassified_images(10)

for i in range(2):
    for j in range(5):
        image = misclassified_images[i*5 + j]
        true_label = misclassified_labels[i*5 + j]
        predicted_label = misclassified_predictions[i*5 + j]
        
        axs[i, j].imshow(image, cmap='gray')
        axs[i, j].set_title(f'True: {true_label}\nPredicted: {predicted_label}', fontsize=10)
        axs[i, j].axis('off')

plt.show()

# %% [markdown]
# Question 2

# %%
seeds = [2, 13, 56, 100, 523]

# %%
def question2(function):
    all_test_errors = []
    final_test_errors = []
    all_validation_history = []

    
    for i in range(5):
        torch.manual_seed(seeds[i])

        model = NeuralNet(input_size, hidden_size, num_classes).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        train_loss_history, test_loss_history, validation_loss_history,_ = function(model, criterion, optimizer)
            
        all_test_errors.append(test_loss_history)
        final_test_errors.append(test_loss_history[-1])
        all_validation_history.append(validation_loss_history)    

        

    return all_test_errors, final_test_errors, all_validation_history 




# %%
#Computing all the test errors for the different seeds
all_test_errors , final_test_errors, _ = question2(question1)

# %%
#Plotting test errors across separate runs
for i in range(5) :
    test_loss = all_test_errors[i]
    plt.plot(test_loss, label = f"Test errors - Seed #{seeds[i]}")

plt.legend()
plt.title("Test erros from the separate runs")
plt.show()

# %%
mean_value = np.mean(final_test_errors)

standard_deviation = np.std(final_test_errors)

variance = np.var(final_test_errors    )

print(f"Mean: {mean_value}")
print(f"Standard Deviation: {standard_deviation}")
print(f"variance : {variance}")

# %% [markdown]
# Question 3

# %%
validation_size = 1000

# %%
# Create validation dataset
validation_dataset = torch.utils.data.Subset(train_dataset, indices=np.random.choice(len(train_dataset), validation_size, replace=False))

# Create validation data loader
validation_loader = torch.utils.data.DataLoader(dataset=validation_dataset, batch_size=batch_size, shuffle=True)

# %%
def question3_run_model(model, criterion, optimizer, train_loss_history, validation_loss_history, test_loss_history):
    train_loss, train_accuracy = evaluation_procedure(model, train_loader, criterion)
    validation_loss, validation_accuracy = evaluation_procedure(model, validation_loader, criterion)
    test_loss, test_accuracy = evaluation_procedure(model, test_loader, criterion)

    train_loss_history.append(train_loss)
    validation_loss_history.append(validation_loss)
    test_loss_history.append(test_loss)

# %%
def question3(model, criterion, optimizer):
    train_loss_history = []
    train_accuracy_history = []
    test_loss_history = []
    test_accuracy_history = []
    validation_loss_history = []
    validation_accuracy_history = []

    for epoch in range(num_epochs):
        train_procedure(model, criterion, optimizer)
        question3_run_model(model, criterion, optimizer, train_loss_history, validation_loss_history, test_loss_history)

    return  train_loss_history, test_loss_history, validation_loss_history, []

# %%
all_test_history,_,all_validation_history = question2(question3)

# %%
print(all_validation_history)
print(all_test_history)

# %%
test_validation_errors = []

for i in range(len(all_validation_history)):
    min = np.min(all_validation_history[i])
    min_index = all_validation_history[i].index(min)
    test_validation_errors.append([min, all_test_history[i][min_index]])


for i in range(5):
    print(f"Seed number {seeds[i]} : Minimum val error = {test_validation_errors[i][0]} , correspond test error = {test_validation_errors[i][1]} ")
    

# %%
values = [sublist[0] for sublist in test_validation_errors]
min = np.min(values)
seed = seeds[values.index(min)]
print(f"The minimum val error = {min}, the correspond test error = {test_validation_errors[seeds.index(seed)][1]}")

# %%

# Plotting test errors
for i in range(len(all_test_history)):
    test_errors = all_test_history[i]
    plt.plot(test_errors, label=f"Seed {seeds[i]} Test Errors")

# Plotting validation errors
for i in range(len(all_validation_history)):
    validation_errors = all_validation_history[i]
    plt.plot(validation_errors, label=f"Seed {seeds[i]} Validation Errors")

plt.xlabel('Epoch')
plt.ylabel('Error')
plt.title('Test and Validation Errors for Each Seed')
plt.legend()
plt.show()

# %% [markdown]
# Question 4

# %%
import itertools

# Hyperparameters
batch_sizes = [100, 200, 300]
hidden_sizes = [500, 1000, 1500]
learning_rates = [0.001, 0.01, 0.1]

# Create all possible combinations of hyperparameters
combinations = list(itertools.product(batch_sizes, hidden_sizes, learning_rates))



# %%
def question4():
    all_test_history = []
    all_validation_history = []

    for i in range(len(combinations)):
        batch_size = combinations[i][0]
        hidden_size = combinations[i][1]
        learning_rate = combinations[i][2]

        model = NeuralNet(input_size, hidden_size, num_classes).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        train_loss_history, test_loss_history, validation_loss_history, _ = question3(model, criterion, optimizer)

        all_test_history.append(test_loss_history)
        all_validation_history.append(validation_loss_history)


    return all_test_history, all_validation_history

# %%
all_test_history, all_validation_history = question4()

# %%
test_validation_errors = []

for i in range(len(all_validation_history)):
    min = np.min(all_validation_history[i])
    min_index = all_validation_history[i].index(min)
    test_validation_errors.append([min, all_test_history[i][min_index]])


for i in range(len(combinations)):
    print(f"Batch size: {combinations[i][0]} , Hidden size: {combinations[i][1]} , Learning rate: {combinations[i][2]} ,  Minimum val error = {test_validation_errors[i][0]} , correspond test error = {test_validation_errors[i][1]} ")

# %%
#Finding the combination that yields the best result
values = [sublist[0] for sublist in test_validation_errors]
min = np.min(values)
index = values.index(min)
best_combination = combinations[index]
print(f"Best combination: Batch size: {best_combination[0]}, Hidden size: {best_combination[1]}, Learning rate: {best_combination[2]}")
print(f"Minimum val error = {min}, the correspond test error = {test_validation_errors[index][1]}")

# %%
import pandas as pd

#Creating a table of the results for all combinations
results = {'Batch Size': [combination[0] for combination in combinations],
           'Hidden Size': [combination[1] for combination in combinations],
           'Learning Rate': [combination[2] for combination in combinations],
           'Validation Error': [test_validation_error[0] for test_validation_error in test_validation_errors],
           'Test Error': [test_validation_error[1] for test_validation_error in test_validation_errors]}
df = pd.DataFrame(results)

print(df)

#Highlighting the best combination 
df.style.apply(lambda x: ['background: green' if x.name == index else '' for i in x], axis=1)







# %% [markdown]
# Question 5

# %%
from sklearn.manifold import TSNE

# %%
# Extract hidden features z_i and original input x_i for the train set
model.eval()
with torch.no_grad():
    all_zi = []
    all_xi = []
    all_labels = []
    for images, labels in train_loader:
        images = images.view(-1, 28*28)
        zi = model(images)
        all_zi.append(zi)
        all_xi.append(images)
        all_labels.append(labels)
    
    all_zi = torch.cat(all_zi).numpy()
    all_xi = torch.cat(all_xi).numpy()
    all_labels = torch.cat(all_labels).numpy()

# Apply t-SNE to reduce dimensionality to 2D
tsne_zi = TSNE(n_components=2, random_state=2).fit_transform(all_zi)
tsne_xi = TSNE(n_components=2, random_state=2).fit_transform(all_xi)

# Create a color map
colors = ['#ff0000', '#ff7f00', '#ffff00', '#7fff00', '#00ff00', '#00ff7f', '#00ffff', '#007fff', '#0000ff', '#7f00ff']

# Plot t-SNE of zi
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
for digit in range(10):
    idx = all_labels == digit
    plt.scatter(tsne_zi[idx, 0], tsne_zi[idx, 1], label=str(digit), color=colors[digit], alpha=0.5)
plt.title('t-SNE of Hidden Features $z_i$')
plt.legend()

# Plot t-SNE of xi
plt.subplot(1, 2, 2)
for digit in range(10):
    idx = all_labels == digit
    plt.scatter(tsne_xi[idx, 0], tsne_xi[idx, 1], label=str(digit), color=colors[digit], alpha=0.5)
plt.title('t-SNE of Original Input $x_i$')
plt.legend()

plt.show()

# %%



