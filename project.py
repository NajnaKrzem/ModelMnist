import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from model import MLP
from train import train_model
from test import test_model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

batch_size = 64
learning_rate = 0.001
num_epochs = 10


transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])
train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
test_dataset = datasets.MNIST(root='./data', train=False, transform=transform, download=True)

train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

input_size = 28 * 28
hidden_size = 128
output_size = 10
model = MLP(input_size, hidden_size, output_size).to(device)


train_model(model, train_loader, device, num_epochs, learning_rate)
test_model(model, test_loader, device)
