import torch
import numpy as np
from torchvision import transforms, datasets, models
from PIL import Image
import torch.nn as nn
import torch.nn.functional as F
import os

DEVICE = "cpu"

transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

class SiameseBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        backbone.fc = nn.Identity()
        self.backbone = backbone
        self.fc = nn.Linear(512, 128)

    def forward(self, x):
        x = self.backbone(x)
        x = self.fc(x)
        x = F.normalize(x, p=2, dim=1)
        return x

model = SiameseBackbone().to(DEVICE)
model.load_state_dict(torch.load("siamese_trained.pth", map_location=DEVICE))
model.eval()

# LOAD TRAIN DATA TO BUILD PROTOTYPES
train_dir = os.path.join("Data", "train")
train_dataset = datasets.ImageFolder(train_dir, transform=transform)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=16)

CLASSES = train_dataset.classes
num_classes = len(CLASSES)

prototypes = {cls: [] for cls in CLASSES}

with torch.no_grad():
    for imgs, labels in train_loader:
        imgs = imgs.to(DEVICE)
        emb = model(imgs)
        for i in range(len(labels)):
            cls = CLASSES[labels[i]]
            prototypes[cls].append(emb[i].cpu().numpy())

for cls in CLASSES:
    prototypes[cls] = np.mean(prototypes[cls], axis=0)

# SINGLE IMAGE PREDICTION
def predict_image(image_path, threshold=0.6):
    img = Image.open(image_path)
    img = transform(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        emb = model(img)[0].cpu().numpy()

    sims = {}
    for cls in CLASSES:
        sim = np.dot(emb, prototypes[cls]) / (
            np.linalg.norm(emb) * np.linalg.norm(prototypes[cls]) + 1e-8
        )
        sims[cls] = sim

    best_class = max(sims, key=sims.get)
    best_score = sims[best_class]

    if best_score >= threshold:
        print(f"INPUT IMAGE: {image_path}")
        print(f"Prediction: DEFECTIVE")
        print(f"Defect type: {best_class}")
        print(f"Similarity score: {best_score:.2f}")
    else:
        print(f"INPUT IMAGE: {image_path}")
        print("Prediction: NOT DEFECTIVE (unknown)")
        print(f"Max similarity: {best_score:.2f}")

predict_image("Data/test/scratches/scratches_161.jpg")