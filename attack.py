import foolbox
import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np

device = 'cuda' if torch.cuda.is_available() else 'cpu'

transform = transforms.Compose([
    transforms.ToTensor()
])

test = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=True, transform=transform)

test_loader = torch.utils.data.DataLoader(
    test, batch_size=100, shuffle=False, num_workers=4)

def generate_adversarials(attack, dataloader, save_name=None):
    
    adversarials = []
    for x, y in dataloader:
        advs = attack(x.numpy(), y.numpy())
        adversarials.extend(advs)
        
    adversarials = np.array(adversarials)
    
    if save_name:
        if not "adversarials" in os.listdir():
            os.mkdir("adversarials")
        np.save(f"adversarials/{save_name}", adversarials)
        
    return adversarials

from vgg import *

VGG_16 = [64, 64, 'pooling', 128, 128, 'pooling', 256, 256,
          256, 'pooling', 512, 512, 512, 'pooling', 512, 512, 512, 'pooling']

net = VGG(VGG_16)
net = net.to(device)
net.eval()

net.load_state_dict(torch.load('models_weights/vggModel'))

preprocessing = dict(axis=-3)
fmodel = foolbox.models.PyTorchModel(net, bounds=(0, 1), num_classes=10, preprocessing=preprocessing)

fgsm = foolbox.attacks.FGSM(fmodel)

_ = generate_adversarials(fgsm, test_loader, 'fgsm')