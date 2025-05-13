import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Resize, ToTensor, RandomAffine
from tqdm import tqdm
from dataset import CatDogDataset
from sklearn.metrics import f1_score, roc_auc_score
from torchvision.models import resnet18

# Config
device = 'cuda' if torch.cuda.is_available() else 'cpu'
batch_size = 32
epochs = 10
lr = 1e-3
num_classes = 2

# Dataloader
transform_train = Compose([
    RandomAffine(
    degrees=(-5, 5),
    translate=(0.15, 0.15),
    scale=(0.85, 1.15),
    shear=5
    ),
    Resize((512, 512)),
    ToTensor(),
])

transform_test = Compose([
    Resize((512, 512)),
    ToTensor(),
])

train_dataset = CatDogDataset(root='data', train=True, transform=transform_train)
val_dataset = CatDogDataset(root='data', train=False, transform=transform_test)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

class ResNetTransferModel(nn.Module):
    def __init__(self, num_class=2, freeze_features=True):
        super(ResNetTransferModel, self).__init__()

        self.backbone = resnet18(pretrained=True)

        if freeze_features:
            for param in self.backbone.parameters():
                param.requires_grad = False

        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Linear(in_features, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_class)
        )

    def forward(self, x):
        return self.backbone(x)

    @property
    def fc(self):
        return self.backbone.fc

    def load_checkpoint(self, checkpoint_path, device='cpu'):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        self.load_state_dict(checkpoint['model'])
        self.to(device)
        self.eval()


# Loss và Optimizer
model = ResNetTransferModel(num_class=num_classes)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.fc.parameters(), lr=lr)

# Train
def train():
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for imgs, labels in tqdm(train_loader, desc='Training'):
        imgs, labels = imgs.to(device), labels.to(device)
        outputs = model(imgs)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    acc = correct / total * 100
    print(f'Train Loss: {running_loss:.4f}, Accuracy: {acc:.2f}%')


# ========== Validation ==========
def validate():
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for imgs, labels in tqdm(val_loader, desc='Validating'):
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    acc = correct / total * 100
    print(f'Validation Accuracy: {acc:.2f}%')


if __name__ == '__main__':
    for epoch in range(epochs):
        print(f'Epoch {epoch+1}/{epochs}')
        train()
        validate()

    # Save model
    torch.save({'model': model.state_dict()}, './train_model/resnet18_transfer_best.pt')


### F1 score
    # all_labels = []
    # all_preds = []
    # model = resnet18(pretrained=False)
    # model.fc = nn.Sequential(
    #     nn.Linear(model.fc.in_features, 128),
    #     nn.ReLU(),
    #     nn.Dropout(0.5),
    #     nn.Linear(128, 2)
    # )
    # checkpoint = torch.load('./train_model/resnet18_transfer_best.pt', map_location='cpu')
    # model.load_state_dict(checkpoint['model'])
    # model = model.to(device)
    # model.eval()
    #
    # with torch.no_grad():
    #     for images, labels in val_loader:
    #         images = images.to(device)
    #         outputs = model(images)
    #         preds = torch.argmax(outputs, dim=1)
    #
    #         all_preds.extend(preds.cpu().numpy())
    #         all_labels.extend(labels.cpu().numpy())
    #
    # f1 = f1_score(all_labels, all_preds, average='binary')
    # print(f'F1 Score: {f1:.4f}')


### AUC
    # all_labels = []
    # all_probs = []
    #
    # with torch.no_grad():
    #     for images, labels in val_loader:
    #         images = images.to(device)
    #         outputs = model(images)  # logits
    #         probs = torch.softmax(outputs, dim=1)[:, 1]  # lấy xác suất class = 1 (chó)
    #         all_probs.extend(probs.cpu().numpy())
    #         all_labels.extend(labels.cpu().numpy())
    #
    # auc = roc_auc_score(all_labels, all_probs)
    # print(f'AUC Score = {auc:.4f}')