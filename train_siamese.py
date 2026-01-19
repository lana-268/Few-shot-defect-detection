import os
import random
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms, models
import torch.nn.functional as F
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

train_dir = os.path.join("Data", "train")
test_dir = os.path.join("Data", "test")

train_dataset = datasets.ImageFolder(root=train_dir, transform=transform)
test_dataset = datasets.ImageFolder(root=test_dir, transform=transform)

print("Train images:", len(train_dataset))
print("Test images:", len(test_dataset))
print("Classes:", train_dataset.classes)

num_classes = len(train_dataset.classes)


class PairDataset(Dataset):
    def __init__(self, base_dataset):
        self.base_dataset = base_dataset

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        img1, label1 = self.base_dataset[idx]

        # pick a random second image
        idx2 = random.randint(0, len(self.base_dataset) - 1)
        img2, label2 = self.base_dataset[idx2]

        # label: 1 if same class, 0 if different classes
        same = 1.0 if label1 == label2 else 0.0
        return img1, img2, torch.tensor(same, dtype=torch.float32)

pair_train_dataset = PairDataset(train_dataset)
pair_train_loader = DataLoader(pair_train_dataset, batch_size=16, shuffle=True)


class SiameseBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        # freeze backbone to make training fast
        for param in self.backbone.parameters():
            param.requires_grad = False

        self.backbone.fc = nn.Identity()
        self.fc = nn.Linear(512, 128)

    def forward(self, x):
        x = self.backbone(x)
        x = self.fc(x)
        x = F.normalize(x, p=2, dim=1)
        return x

model = SiameseBackbone().to(device)
print("Model created.")


# CONTRASTIVE LOSS
class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0):
        super().__init__()
        self.margin = margin

    def forward(self, output1, output2, label_same):
        distances = F.pairwise_distance(output1, output2)
        positive_loss = label_same * distances.pow(2)
        negative_loss = (1 - label_same) * F.relu(self.margin - distances).pow(2)
        loss = positive_loss + negative_loss
        return loss.mean()

criterion = ContrastiveLoss(margin=1.0)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# TRAINING LOOP
num_epochs = 8
epoch_losses = []

print("\nStarting training...\n")
model.train()

for epoch in range(num_epochs):
    running_loss = 0.0
    for img1, img2, label_same in pair_train_loader:
        img1 = img1.to(device)
        img2 = img2.to(device)
        label_same = label_same.to(device)

        emb1 = model(img1)
        emb2 = model(img2)

        loss = criterion(emb1, emb2, label_same)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    avg_loss = running_loss / len(pair_train_loader)
    epoch_losses.append(avg_loss)
    print(f"Epoch [{epoch+1}/{num_epochs}] - Average loss: {avg_loss:.4f}")

print("\nTraining finished.")

torch.save(model.state_dict(), "siamese_trained.pth")
print("Saved trained model to siamese_trained.pth")

plt.figure()
plt.plot(range(1, num_epochs + 1), epoch_losses, marker="o")
plt.xlabel("Epoch")
plt.ylabel("Contrastive loss")
plt.title("Training loss curve")
plt.grid(True)
plt.tight_layout()
plt.savefig("loss_curve.png")
print("Saved loss curve to loss_curve.png")

print("\nEvaluating with class prototypes...")

model.eval()

# to build prototypes from train set
train_loader_plain = DataLoader(train_dataset, batch_size=16, shuffle=False)

# sum of embeddings and counts for each class
proto_sums = [torch.zeros(128, device=device) for _ in range(num_classes)]
proto_counts = [0 for _ in range(num_classes)]

with torch.no_grad():
    for imgs, labels in train_loader_plain:
        imgs = imgs.to(device)
        labels = labels.to(device)
        emb = model(imgs)  # [B, 128]
        for i in range(len(labels)):
            c = int(labels[i].item())
            proto_sums[c] += emb[i]
            proto_counts[c] += 1

prototypes = []
for c in range(num_classes):
    prototypes.append(proto_sums[c] / proto_counts[c])
prototypes = torch.stack(prototypes)  # [C, 128]

# classify test images by nearest prototype
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

correct = 0
total = 0

with torch.no_grad():
    for imgs, labels in test_loader:
        imgs = imgs.to(device)
        labels = labels.to(device)

        emb = model(imgs)  # [B, 128]
        # [B, C] cosine similarity between each emb and each prototype
        sims = F.cosine_similarity(emb.unsqueeze(1), prototypes.unsqueeze(0), dim=2)
        preds = sims.argmax(dim=1)

        correct += (preds == labels).sum().item()
        total += labels.size(0)

accuracy = 100.0 * correct / total
print(f"Prototype-based classification accuracy on test set: {accuracy:.2f}%")

print("\nAll done.")




# train_siamese.py?
# Trains your Siamese network using contrastive loss
# Saves the trained weights to siamese_trained.pth
# Plots the training loss curve to loss_curve.png
# Evaluates the model using class prototypes and prints test accuracy

#So the full workflow is:
# train_siamese.py → learns & saves model
# evaluate_siamese.py → analyzes confusion matrix
# main.py → demos similarity on a batch