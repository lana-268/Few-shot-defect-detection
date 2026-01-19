import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

train_dir = os.path.join("Data", "train")
train_dataset = datasets.ImageFolder(root=train_dir, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)

print("Number of training images:", len(train_dataset))
print("Classes:", train_dataset.classes)

class SiameseBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        # Pretrained ResNet as feature extractor.
        self.backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        self.backbone.fc = nn.Identity()
        self.fc = nn.Linear(512, 128)

    def forward(self, x):
        x = self.backbone(x)
        x = self.fc(x)
        # Normalize to unit length (for cosine similarity)
        x = nn.functional.normalize(x, p=2, dim=1)
        return x

model = SiameseBackbone().to(device)
model.eval()
print("Model ready!")

# IT TAKE ONE BATCH AND COMPUTE EMBEDDINGS
images, labels = next(iter(train_loader))
images = images.to(device)
labels = labels.to(device)

with torch.no_grad():
    embeddings = model(images)

print("\n--- Batch info ---")
print("Batch size:", images.shape[0])
print("Embedding shape:", embeddings.shape)
print("Labels in this batch:", labels.cpu().tolist())

# FIND A PAIR FROM SAME CLASS AND A PAIR FROM DIFFERENT CLASSES
same_pair = None
diff_pair = None

num_imgs = images.shape[0]

for i in range(num_imgs):
    for j in range(i + 1, num_imgs):
        if labels[i] == labels[j] and same_pair is None:
            same_pair = (i, j)
        if labels[i] != labels[j] and diff_pair is None:
            diff_pair = (i, j)
        if same_pair is not None and diff_pair is not None:
            break
    if same_pair is not None and diff_pair is not None:
        break

# COMPUTE COSINE SIMILARITY
def cosine_sim(idx1, idx2):
    emb1 = embeddings[idx1]
    emb2 = embeddings[idx2]
    sim = torch.nn.functional.cosine_similarity(emb1.unsqueeze(0), emb2.unsqueeze(0))
    return sim.item()

print("\n--- Similarity demo ---")

if same_pair is not None:
    i, j = same_pair
    sim_same = cosine_sim(i, j)
    print(f"Same-class pair: indices ({i}, {j}) with labels ({labels[i].item()}, {labels[j].item()})")
    print("Cosine similarity (same class):", sim_same)
else:
    print("Could not find a same-class pair in this batch (try running again).")

if diff_pair is not None:
    i, j = diff_pair
    sim_diff = cosine_sim(i, j)
    print(f"\nDifferent-class pair: indices ({i}, {j}) with labels ({labels[i].item()}, {labels[j].item()})")
    print("Cosine similarity (different classes):", sim_diff)
else:
    print("Could not find a different-class pair in this batch (try running again).")

print("\nDone.")



# main.py is a demonstration script that shows:
#How the trained Siamese network creates embeddings
#How images from the same class produce HIGH similarity
#How images from different classes produce LOW similarity

#In other words:
# main.py is used to visualize and understand what the Siamese model learned.
# It helps you “see” the feature space by comparing embeddings.