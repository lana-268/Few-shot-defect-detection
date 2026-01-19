import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F


DEVICE = "cpu"

DATA_DIR = "Data"
TRAIN_DIR = os.path.join(DATA_DIR, "train")
TEST_DIR = os.path.join(DATA_DIR, "test")

transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])


train_dataset = datasets.ImageFolder(root=TRAIN_DIR, transform=transform)
test_dataset = datasets.ImageFolder(root=TEST_DIR, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

CLASSES = train_dataset.classes
print("Train samples:", len(train_dataset))
print("Test samples:", len(test_dataset))
print("Classes:", CLASSES)


class SiameseBackbone(nn.Module):
    def __init__(self, embedding_dim=128):
        super().__init__()
        backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        num_features = backbone.fc.in_features
        backbone.fc = nn.Identity()
        self.backbone = backbone
        self.fc = nn.Linear(num_features, embedding_dim)

    def forward(self, x):
        feats = self.backbone(x)
        emb = self.fc(feats)
        emb = F.normalize(emb, p=2, dim=1)
        return emb

model = SiameseBackbone().to(DEVICE)

state_dict = torch.load("siamese_trained.pth", map_location=DEVICE)
model.load_state_dict(state_dict)
model.eval()

print("Loaded trained model from siamese_trained.pth")

def get_embedding(x: torch.Tensor) -> np.ndarray:
    """Get embedding from the model as a NumPy array."""
    with torch.no_grad():
        emb = model(x)
    return emb.cpu().numpy()

# BUILD CLASS PROTOTYPES
prototypes = {cls: [] for cls in CLASSES}

for imgs, labels in train_loader:
    imgs = imgs.to(DEVICE)
    emb = get_embedding(imgs)[0]
    cls_idx = int(labels.item())
    cls_name = CLASSES[cls_idx]
    prototypes[cls_name].append(emb)

for cls in CLASSES:
    prototypes[cls] = np.mean(np.stack(prototypes[cls], axis=0), axis=0)

print("Built class prototypes.")


y_true = []
y_pred = []

def cosine_sim(a, b):
    num = np.dot(a, b)
    den = np.linalg.norm(a) * np.linalg.norm(b)
    return num / (den + 1e-8)

for imgs, labels in test_loader:
    imgs = imgs.to(DEVICE)
    emb = get_embedding(imgs)[0]

    sims = []
    for cls in CLASSES:
        sims.append(cosine_sim(emb, prototypes[cls]))

    pred_idx = int(np.argmax(sims))

    y_true.append(int(labels.item()))
    y_pred.append(pred_idx)

cm = confusion_matrix(y_true, y_pred)
print("Confusion matrix:\n", cm)

print("\nClassification report:\n")
print(classification_report(y_true, y_pred, target_names=CLASSES))

plt.figure(figsize=(6, 5))
plt.imshow(cm, interpolation="nearest")
plt.title("Confusion Matrix")
plt.colorbar()
tick_marks = np.arange(len(CLASSES))
plt.xticks(tick_marks, CLASSES, rotation=45, ha="right")
plt.yticks(tick_marks, CLASSES)
plt.xlabel("Predicted label")
plt.ylabel("True label")
plt.tight_layout()
plt.savefig("confusion_matrix.png")
plt.close()

print("Saved confusion_matrix.png")




# evaluate_siamese.py: Its purpose is to test how good your trained model is by:

# Generating class prototypes
# (average embedding for each defect class)

# Embedding every test image
# (turning them into 128-dimensional vectors)

# Comparing each test image to the prototypes using cosine similarity
# (choosing which class it is closest to)

# Producing:
# Confusion matrix
# Classification report
# Overall accuracy
# Visual PNG file of confusion matrix

# So in the full workflow:
# train_siamese.py → teaches the model
# evaluate_siamese.py → measures how good the model is