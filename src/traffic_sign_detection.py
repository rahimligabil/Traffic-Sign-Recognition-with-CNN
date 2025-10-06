import os
import pandas as pd
import torch                        
import torch.nn as nn              
import torch.optim as optim         
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms          
from torchvision.datasets import ImageFolder
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import Subset, DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms
import torch, numpy as np    
from CNN import SimpleCNN
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report



DATA_PATH = "./data"
TRAIN_DIR = "./data/Train"
TEST_DIR  = os.path.join(DATA_PATH, "Test")
META_CSV  = os.path.join(DATA_PATH, "Meta.csv")




IMG_SIZE = (30,30)
BATCH_SIZE = 64
EPOCH  = 30
SEED = 42
NUM_CLASSES = 43

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

data = pd.read_csv("data/Meta.csv")
print(data.head())


classes = { 0:'Speed limit (20km/h)',
            1:'Speed limit (30km/h)', 
            2:'Speed limit (50km/h)', 
            3:'Speed limit (60km/h)', 
            4:'Speed limit (70km/h)', 
            5:'Speed limit (80km/h)', 
            6:'End of speed limit (80km/h)', 
            7:'Speed limit (100km/h)', 
            8:'Speed limit (120km/h)', 
            9:'No passing', 
            10:'No passing veh over 3.5 tons', 
            11:'Right-of-way at intersection', 
            12:'Priority road', 
            13:'Yield', 
            14:'Stop', 
            15:'No vehicles', 
            16:'Veh > 3.5 tons prohibited', 
            17:'No entry', 
            18:'General caution', 
            19:'Dangerous curve left', 
            20:'Dangerous curve right', 
            21:'Double curve', 
            22:'Bumpy road', 
            23:'Slippery road', 
            24:'Road narrows on the right', 
            25:'Road work', 
            26:'Traffic signals', 
            27:'Pedestrians', 
            28:'Children crossing', 
            29:'Bicycles crossing', 
            30:'Beware of ice/snow',
            31:'Wild animals crossing', 
            32:'End speed + passing limits', 
            33:'Turn right ahead', 
            34:'Turn left ahead', 
            35:'Ahead only', 
            36:'Go straight or right', 
            37:'Go straight or left', 
            38:'Keep right', 
            39:'Keep left', 
            40:'Roundabout mandatory', 
            41:'End of no passing', 
            42:'End no passing veh > 3.5 tons' }


from torchvision.datasets import ImageFolder
import os

class NumericImageFolder(ImageFolder):
    def find_classes(self, directory):
        classes = [d.name for d in os.scandir(directory) if d.is_dir()]
        # Sayısal olarak sırala
        classes.sort(key=lambda s: int(s))
        class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        return classes, class_to_idx




classes_list = sorted(os.listdir(TRAIN_DIR))

class_count = {}

for c in classes_list:
    folder_path = os.path.join(TRAIN_DIR,c)
    class_count[c] = len(os.listdir(folder_path))


max_class = max(class_count,key = class_count.get)
min_class = min(class_count,key = class_count.get)

max_samples = max(class_count.values())
min_samples = min(class_count.values())
ratio = max_samples / min_samples


print(ratio)


train_transforms = transforms.Compose([
    transforms.Resize(IMG_SIZE),                             
    transforms.RandomRotation(10),                         
    transforms.RandomAffine(0, translate=(0.1, 0.1), scale=(0.9, 1.1)),  
    transforms.ColorJitter(brightness=0.2, contrast=0.2),   
    transforms.ToTensor(),                                   
    transforms.Normalize(mean=(0.5, 0.5, 0.5),               
                         std=(0.5, 0.5, 0.5)),
])

eval_transforms = transforms.Compose([
    transforms.Resize(IMG_SIZE),                            
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5, 0.5, 0.5),
                         std=(0.5, 0.5, 0.5)),
])


base_ds = NumericImageFolder(TRAIN_DIR)
y = np.array(base_ds.targets)  
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
train_idx, val_idx = next(sss.split(X=np.arange(len(y)), y=y))


train_full = NumericImageFolder(root=TRAIN_DIR, transform=train_transforms)
val_full   = NumericImageFolder(root=TRAIN_DIR, transform=eval_transforms)

train_ds = Subset(train_full, train_idx)
val_ds   = Subset(val_full,   val_idx)


train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0, pin_memory=True)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)

model = SimpleCNN(num_classes=NUM_CLASSES).to(device)


NUM_CLASSES = len(base_ds.classes)  


train_labels = torch.tensor(y[train_idx], dtype=torch.long)  


class_counts = torch.bincount(train_labels, minlength=NUM_CLASSES).float()

class_counts = class_counts.clamp_min(1)
total = class_counts.sum()
class_weights = total / (NUM_CLASSES * class_counts)
class_weights = class_weights.to(torch.float32).to(device)
criterion = nn.CrossEntropyLoss(weight=class_weights)


optimizer = optim.Adam(model.parameters(), lr=1e-3)   
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)



def accuracy_from_logits(logits, targets):
    preds = torch.argmax(logits, dim=1)
    correct = (preds == targets).sum().item()
    total = targets.size(0)
    return correct / total




def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, total_acc, total_samples = 0.0, 0.0, 0


    for images,labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        result = model(images)
        loss = criterion(result,labels)
        loss.backward()
        optimizer.step()

        batch_acc = accuracy_from_logits(result, labels)
        bs = labels.size(0)
        total_loss += loss.item() * bs
        total_acc  += batch_acc * bs
        total_samples += bs

    return total_loss / total_samples, total_acc / total_samples

@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, total_acc, total_samples = 0.0, 0.0, 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        logits = model(images)
        loss = criterion(logits, labels)

        batch_acc = accuracy_from_logits(logits, labels)
        bs = labels.size(0)
        total_loss += loss.item() * bs
        total_acc  += batch_acc * bs
        total_samples += bs

    return total_loss / total_samples, total_acc / total_samples


@torch.no_grad()
def evaluate_with_f1(model, loader, device):
    model.eval()
    all_true, all_pred = [], []

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        preds = torch.argmax(logits, dim=1)
        all_true.append(labels.cpu().numpy())
        all_pred.append(preds.cpu().numpy())

    y_true = np.concatenate(all_true)
    y_pred = np.concatenate(all_pred)

    precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
    recall = recall_score(y_true, y_pred, average='macro', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    print("\nDetailed classification report:")
    print(classification_report(y_true, y_pred, zero_division=0))
    print(f"Precision: {precision:.4f} | Recall: {recall:.4f} | F1: {f1:.4f}")


if __name__ == "__main__":
    best_val_acc = 0.0

    for epoch in range(1, EPOCH + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss,   val_acc   = evaluate(model, val_loader,   criterion, device)

        scheduler.step(val_acc)

        print(f"[{epoch:02d}/{EPOCH}] "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "best_model.pth")
            print("  Best model saved: best_model.pth")


    print("\n=== Final Evaluation ===")
    model.load_state_dict(torch.load("best_model.pth"))
    evaluate_with_f1(model, val_loader, device)