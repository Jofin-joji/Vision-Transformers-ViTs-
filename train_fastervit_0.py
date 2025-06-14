import os
import time
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm
from fastervit import create_model

# ----------------------------
# Configuration
# ----------------------------
data_dir = 'dataset'
batch_size = 16  # Reduce if out-of-memory
embedding_size = 512 # This is defined but not explicitly used in the model head directly, FasterViT output dim will be used
num_epochs = 20
lr = 1e-4
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ----------------------------
# Transforms
# ----------------------------
transform_train = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

transform_val = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# ----------------------------
# Triplet Dataset
# ----------------------------
class TripletDataset(Dataset):
    def __init__(self, image_folder):
        self.dataset = image_folder
        self.labels = [s[1] for s in self.dataset.samples]
        self.label_to_indices = {}
        for idx, label in enumerate(self.labels):
            self.label_to_indices.setdefault(label, []).append(idx)

        # Ensure there are at least two different classes for negative sampling
        if len(self.label_to_indices) < 2:
            raise ValueError("TripletDataset requires at least two different classes for negative sampling.")
        # Ensure each class has enough samples if we want to pick a different positive
        for label, indices in self.label_to_indices.items():
            if len(indices) < 1: # Or < 2 if you strictly want a *different* positive image
                print(f"Warning: Class {label} has only {len(indices)} sample(s). This might be an issue for positive sampling if you require a different positive sample.")


    def __getitem__(self, index):
        anchor_img_path, anchor_label = self.dataset.samples[index]
        anchor_img = self.dataset.loader(anchor_img_path) 
        if self.dataset.transform is not None:
            anchor_img = self.dataset.transform(anchor_img) # Apply transform

        # Positive sample
        possible_pos_indices = self.label_to_indices[anchor_label]
        if len(possible_pos_indices) > 1:
            pos_index = index
            while pos_index == index: # Ensure positive is different from anchor
                pos_index = random.choice(possible_pos_indices)
        else: # Only one image in this class, use anchor itself as positive (less ideal but avoids error)
            pos_index = index
            if len(possible_pos_indices) == 1 and index != possible_pos_indices[0]:
                 # This case should not happen if index is from self.dataset.samples
                 # and label_to_indices is built correctly from self.dataset.samples
                 pos_index = possible_pos_indices[0] # Fallback
            elif len(possible_pos_indices) == 0:
                 raise ValueError(f"No positive samples found for label {anchor_label}, index {index}. This should not happen.")


        pos_img_path, _ = self.dataset.samples[pos_index]
        pos_img = self.dataset.loader(pos_img_path)
        if self.dataset.transform is not None:
            pos_img = self.dataset.transform(pos_img)

        # Negative sample
        neg_label = anchor_label
        # Ensure there are other labels to choose from
        available_neg_labels = [l for l in self.label_to_indices if l != anchor_label]
        if not available_neg_labels:
            # This case should be caught by the __init__ check, but as a safeguard:
            # Fallback: if only one class, negative will be from the same class (not ideal for triplet loss)
            # Or, better, raise an error or handle as per specific requirements.
            # For now, let's assume __init__ check prevents this.
            # If it can happen, you might need to duplicate the anchor or positive as negative,
            # which would result in zero loss contribution from such triplets.
            print(f"Warning: Only one class available. Negative sampling will pick from the same class as anchor.")
            neg_label = anchor_label # Or handle differently
        else:
            neg_label = random.choice(available_neg_labels)

        neg_index = random.choice(self.label_to_indices[neg_label])
        neg_img_path, _ = self.dataset.samples[neg_index]
        neg_img = self.dataset.loader(neg_img_path)
        if self.dataset.transform is not None:
            neg_img = self.dataset.transform(neg_img)

        return anchor_img, pos_img, neg_img

    def __len__(self):
        return len(self.dataset.samples)

# ----------------------------
# Validation Accuracy
# ----------------------------
# Modified to accept model, val_loader, and device as arguments
def compute_validation_accuracy(model, val_loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        embeddings = []
        labels_list = []
        for imgs, labels in val_loader: # Use the passed val_loader
            imgs, labels = imgs.to(device), labels.to(device) # Use the passed device
            emb = F.normalize(model(imgs), p=2, dim=1) # Use the passed model
            embeddings.append(emb)
            labels_list.append(labels)

        if not embeddings: # Handle case where val_loader is empty
            model.train()
            return 0.0

        embeddings = torch.cat(embeddings)
        labels_list = torch.cat(labels_list)

        if embeddings.shape[0] < 2: # Not enough samples to compare
            model.train()
            return 0.0


        for i in range(len(embeddings)):
            anchor = embeddings[i].unsqueeze(0)
            # Create a mask to exclude the anchor itself from distance calculation
            mask = torch.ones(len(embeddings), dtype=torch.bool, device=anchor.device)
            mask[i] = False
            
            # Calculate distances only to other embeddings
            if embeddings[mask].shape[0] > 0: # Ensure there are other embeddings to compare against
                dists_to_others = torch.cdist(anchor, embeddings[mask])
                nn_idx_in_others = torch.argmin(dists_to_others[0])
                
                # Map nn_idx_in_others back to original index in embeddings
                original_indices = torch.arange(len(embeddings), device=anchor.device)[mask]
                nn_idx = original_indices[nn_idx_in_others]

                if labels_list[i] == labels_list[nn_idx]:
                    correct += 1
            # If only one embedding or no other embeddings to compare with,
            # it cannot be correct or incorrect in this KNN sense.
            # Depending on desired behavior, you might count it as incorrect or handle differently.
            # For now, it's skipped if no others to compare.
            total += 1

    model.train()
    return correct / total if total > 0 else 0.0

# ----------------------------
# Main
# ----------------------------
def main():
    print("Loading datasets...")
    # It's generally good practice to apply transforms directly when creating ImageFolder
    # The TripletDataset will then receive already transformed images if it relies on dataset[index]
    # The current TripletDataset re-applies transforms. Let's adjust TripletDataset to use transformed images.
    # The original TripletDataset __getitem__ does:
    # anchor_img, anchor_label = self.dataset[index]
    # This already returns transformed images if train_folder has transforms.
    # So, the TripletDataset __getitem__ was trying to re-access raw paths and re-apply transforms.
    # Let's simplify TripletDataset to directly use the transformed outputs from ImageFolder.

    train_image_folder = datasets.ImageFolder(os.path.join(data_dir, 'train'), transform=transform_train)
    val_image_folder = datasets.ImageFolder(os.path.join(data_dir, 'val'), transform=transform_val) # val_folder used for val_loader
    
    # Simplified TripletDataset that assumes self.dataset provides transformed images
    class TripletDatasetSimple(Dataset):
        def __init__(self, image_folder): # image_folder is an instance of datasets.ImageFolder
            self.image_folder = image_folder
            self.labels = [s[1] for s in self.image_folder.samples]
            self.label_to_indices = {}
            for idx, label in enumerate(self.labels):
                self.label_to_indices.setdefault(label, []).append(idx)

            if len(self.label_to_indices) < 2:
                raise ValueError("TripletDataset requires at least two different classes.")

        def __getitem__(self, index):
            anchor_img, anchor_label = self.image_folder[index] # This gives transformed image and label

            # Positive sample
            possible_pos_indices = self.label_to_indices[anchor_label]
            pos_index = index
            if len(possible_pos_indices) > 1:
                while pos_index == index:
                    pos_index = random.choice(possible_pos_indices)
            elif not possible_pos_indices: # Should not happen
                 raise Exception(f"No samples for label {anchor_label}")
            # else: # only one sample for this class, pos_index remains 'index'

            pos_img, _ = self.image_folder[pos_index]

            # Negative sample
            neg_label = anchor_label
            available_neg_labels = [l for l in self.label_to_indices if l != anchor_label]
            if not available_neg_labels: # Only one class in dataset
                # This should be caught by __init__ check. If not, this triplet is not useful.
                # Fallback: use a sample from the same class (makes d(A,N) potentially small)
                # This will result in a less effective triplet.
                print(f"Warning: Only one class ({anchor_label}) available. Negative sample will be from the same class.")
                neg_index = random.choice(self.label_to_indices[anchor_label])
            else:
                neg_label = random.choice(available_neg_labels)
                neg_index = random.choice(self.label_to_indices[neg_label])
            
            neg_img, _ = self.image_folder[neg_index]

            return anchor_img, pos_img, neg_img

        def __len__(self):
            return len(self.image_folder)

    train_dataset = TripletDatasetSimple(train_image_folder)


    print(f"Number of training samples (triplets): {len(train_dataset)}")
    print(f"Number of validation samples (images): {len(val_image_folder)}")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True if device.type == 'cuda' else False)
    # val_loader uses the standard ImageFolder, not TripletDataset
    val_loader = DataLoader(val_image_folder, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True if device.type == 'cuda' else False)


    print("Loading FasterViT model...")
    # Ensure the model_path is correct or remove if you want to always download/use fresh pretrained weights
    model_path = "tmp/faster_vit_0.pth.tar"
    if not os.path.exists(model_path):
        print(f"Warning: Pretrained model path {model_path} not found. `create_model` might download weights if `pretrained=True` and path is for caching.")
        model_path = None # Let timm handle download if pretrained=True

    model = create_model(
        'faster_vit_0_224',
        pretrained=True,
        model_path=model_path # model_path is for custom local weights, if None and pretrained=True, timm downloads
    )
    # Get the feature dimension from the model if possible, or set embedding_size based on model knowledge
    # For FasterViT, the output of `model.head` (if it were a Linear layer) would indicate this.
    # Since we set model.head = nn.Identity(), the output dim is the dim before the original head.
    # For 'faster_vit_0_224', it seems to be 512.
    # No need to redefine embedding_size unless you add another layer after Identity()
    model.head = nn.Identity() # Get features before classification head
    model.to(device)

    criterion = nn.TripletMarginLoss(margin=1.0)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    print("Starting training loop...")
    for epoch in range(num_epochs):
        epoch_start = time.time()
        model.train() # Set model to training mode
        epoch_loss = 0.0
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        progress_bar = tqdm(train_loader, desc="Training", unit="batch")

        for anc, pos, neg in progress_bar:
            anc, pos, neg = anc.to(device), pos.to(device), neg.to(device)

            optimizer.zero_grad()

            anc_emb = model(anc) # No F.normalize here, FasterViT outputs are not necessarily normalized
            pos_emb = model(pos)
            neg_emb = model(neg)
            
            # TripletMarginLoss expects unnormalized embeddings.
            # Normalization is often done for distance calculation/inference, not always before loss.
            # However, some triplet loss implementations work better with normalized inputs.
            # If you intend to normalize before loss, do it consistently:
            # anc_emb = F.normalize(model(anc), p=2, dim=1)
            # pos_emb = F.normalize(model(pos), p=2, dim=1)
            # neg_emb = F.normalize(model(neg), p=2, dim=1)
            # For now, let's follow the common practice of feeding raw embeddings to TripletMarginLoss.

            loss = criterion(anc_emb, pos_emb, neg_emb)

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item())

        avg_epoch_loss = epoch_loss / len(train_loader) if len(train_loader) > 0 else 0.0
        epoch_time = time.time() - epoch_start
        
        # Call compute_validation_accuracy with necessary arguments
        val_acc = compute_validation_accuracy(model, val_loader, device)
        
        print(f"Epoch [{epoch+1}/{num_epochs}] completed in {epoch_time:.2f} sec")
        print(f"Avg Loss: {avg_epoch_loss:.4f}, Validation Accuracy: {val_acc:.4f}")

    torch.save(model.state_dict(), 'fastervit_face_embeddings.pth')
    print("Model saved as 'fastervit_face_embeddings.pth'")

# ----------------------------
# Entry Point for Windows
# ----------------------------
if __name__ == "__main__":
    # The re-import of datasets is a common pattern for Windows multiprocessing,
    # but with num_workers=0, it might not be strictly necessary. Harmless to keep.
    from torchvision import datasets
    # val_loader = None # This global declaration is not needed here as val_loader is created in main and passed around.
    main()