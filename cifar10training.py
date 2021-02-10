import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn.functional as F
from vgg import VGG

def train(epoch):
    print(f"epoch number {epoch}")
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for X, labels in train_loader:
        X, labels = X.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = net(X)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
    print(f"TRAIN | Loss: {train_loss} | Accuracy: {correct/total * 100}")

def test(epoch):
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for X, labels in test_loader:
            X, labels = X.to(device), labels.to(device)
            outputs = net(X)
            loss = criterion(outputs, labels)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    print(f"TEST | Loss: {test_loss} | Accuracy: {correct/total * 100}")

device = 'cuda' if torch.cuda.is_available() else 'cpu'

transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.RandomAffine(10),
    transforms.ToTensor(),
])

transform_test = transforms.Compose([
    transforms.ToTensor()
])

BATCH_SIZE = 128

train = torchvision.datasets.CIFAR10(
    root='./data', train=True, download=True, transform=transform_train)
train_loader = torch.utils.data.DataLoader(
    train, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

test = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=True, transform=transform_test)
test_loader = torch.utils.data.DataLoader(
    test, batch_size=100, shuffle=False, num_workers=2)

# pattern for creating vgg backbone, int is for nuber of output
# channels for 2d conv layer, 'pooling' is for MaxPool
VGG_16 = [64, 64, 'pooling', 128, 128, 'pooling', 256, 256,
          256, 'pooling', 512, 512, 512, 'pooling', 512, 512, 512, 'pooling']

net = VGG(VGG_16)
net = net.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters())
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

for epoch in range(110):
    train(epoch)
    test(epoch)
    scheduler.step()
    print("######################################################################")

if not "models_weights" in os.listdir():
    os.mkdir("models_weights")
torch.save(net.state_dict(), 'models_weights/vggModel')