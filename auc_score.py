import torch

from model import SimpleCNN, DeepCNN
from dataset import CatDogDataset
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Resize, ToTensor
from dataset import CatDogDataset
from sklearn.metrics import roc_auc_score


model = DeepCNN(num_class=2)

checkpoint = torch.load('train_model/classify_cnn_best_deep.pt', map_location='cpu')
model.load_state_dict(checkpoint['model'])
model.eval()

transform = Compose([Resize((512, 512)), ToTensor()])
test_dataset = CatDogDataset(root='data', train=False, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.to(device)

all_labels = []
all_probs = []

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        outputs = model(images)
        probs = torch.softmax(outputs, dim=1)[:, 1]
        all_probs.extend(probs.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

auc = roc_auc_score(all_labels, all_probs)
print(f'AUC Score = {auc:.4f}')