import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as T
from torchvision.datasets import CIFAR10
import torchvision.models as models

from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
from tsne_vis import run_tsne_analysis

# =======================
# Configurations
# =======================
class Config:
    batch_size = 512
    epochs = 50
    lr = 1e-3
    temperature = 0.5


# =======================
# Data Augmentation for SimCLR
# =======================
class SimCLRTransform:
    def __init__(self):
        self.transform = T.Compose([
            T.RandomResizedCrop(32),
            T.RandomHorizontalFlip(),
            T.ColorJitter(0.4, 0.4, 0.4, 0.1),
            T.RandomGrayscale(p=0.2),
            T.GaussianBlur(kernel_size=3),
            T.ToTensor()
        ])

    def __call__(self, x):
        return self.transform(x), self.transform(x)


# =======================
# Dataset for SimCLR
# =======================
class SimCLRDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
        self.transform = SimCLRTransform()

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        x, y = self.dataset[idx]
        x1, x2 = self.transform(x)
        return x1, x2, y


# =======================
# Model
# =======================
class SimCLRModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = models.resnet18(weights=None)
        self.encoder.fc = nn.Identity()

        self.projector = nn.Sequential(
            nn.Linear(512, 256),
            #nn.Linear(2048,512),
            nn.ReLU(),
            nn.Linear(256, 128)
            #nn.Linear(512, 128)
        )

    def forward(self, x):
        h = self.encoder(x)
        z = self.projector(h)
        return h, z


# =======================
# Loss calculation
# =======================
def contrastive_loss(z1, z2, temperature):
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)

    batch_size = z1.size(0)

    z = torch.cat([z1, z2], dim=0)  # [2N, D]

    sim = torch.matmul(z, z.T)  # [2N, 2N]

    # 去掉自身相似度
    mask = torch.eye(2 * batch_size, dtype=torch.bool).to(z.device)
    sim = sim.masked_fill(mask, -9e15)

    sim = sim / temperature

    # 正样本索引
    labels = torch.arange(batch_size).to(z.device)
    labels = torch.cat([labels + batch_size, labels])

    loss = F.cross_entropy(sim, labels)
    return loss

# =======================
# Dataloader
# =======================
def build_dataloader():
    base_dataset = CIFAR10(root='./data', download=True)

    dataset = SimCLRDataset(base_dataset)

    loader = DataLoader(
        dataset,
        batch_size=Config.batch_size,
        shuffle=True,
        num_workers=10,  
        pin_memory=True
    )

    return loader


# =======================
# Model + Optimizer
# =======================
def build_model(device):
    model = SimCLRModel().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=Config.lr)
    return model, optimizer


# =======================
# Training Loop
# =======================
def train(loader, model, optimizer, device):
    model.train()

    for epoch in range(Config.epochs):
        total_loss = 0

        for x1, x2, _ in loader:
            x1, x2 = x1.to(device), x2.to(device)

            _, z1 = model(x1)
            _, z2 = model(x2)

            loss = contrastive_loss(z1, z2, Config.temperature)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch}: Loss = {total_loss:.4f}")


# =======================
# Feature Extraction
# =======================
def extract_features(loader, model, device):
    model.eval()

    features = []
    labels = []

    with torch.no_grad():
        for x1, _, y in loader:
            x1 = x1.to(device)
            h, _ = model(x1)

            features.append(h.cpu())
            labels.append(y)

    features = torch.cat(features).numpy()
    labels = torch.cat(labels).numpy()

    return features, labels


# =======================
# KNN Cluster Evaluation
# =======================
def evaluate(features, labels):
    print("Running KMeans...")
    kmeans = KMeans(n_clusters=10)
    cluster_labels = kmeans.fit_predict(features)

    print("Running KNN evaluation...")
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(features, labels)

    acc = knn.score(features, labels)
    print("KNN Accuracy:", acc)


# =======================
# Main function
# =======================
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    loader = build_dataloader()

    model, optimizer = build_model(device)
    model.load_state_dict(torch.load("C:\\\\Users\\\\xkt04\\\\Desktop\\\\unsupervised-image-classification\\\\checkpoints\\\\simclr_model_resnet18.pth"))
    model.eval()
    #train(loader, model, optimizer, device)
    #torch.save(model.state_dict(), "C:\\\\Users\\\\xkt04\\\\Desktop\\\\unsupervised-image-classification\\\\checkpoints\\\\simclr_model_resnet18.pth")
    #print("Model saved.")
    features, labels = extract_features(loader, model, device)

    evaluate(features, labels)
    run_tsne_analysis(features, labels)


# =======================
# Run
# =======================
if __name__ == "__main__":
    main()