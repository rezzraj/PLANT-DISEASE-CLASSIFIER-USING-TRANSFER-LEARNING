import os
import shutil
from collections import defaultdict
import torch
from torch.utils.data import DataLoader
from torchvision import transforms , models
from torchvision.datasets import ImageFolder
from torch import nn
import optuna
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset
from torchvision.models import MobileNet_V2_Weights
import torch.optim as optim
import multiprocessing
import json


def main():
    # check cuda is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # applying transformation

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.5, 1.0)),  # zoom in/out
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.2),  # leaves can flip
        transforms.RandomRotation(35),

        transforms.ColorJitter(
            brightness=0.6,
            contrast=0.6,
            saturation=0.6,
            hue=0.15
        ),

        transforms.RandomGrayscale(p=0.1),  # bad lighting
        transforms.GaussianBlur(kernel_size=3),

        transforms.RandomPerspective(distortion_scale=0.4, p=0.3),  # weird angles

        transforms.ToTensor(),

        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # splitting the train data further for optuna
    train_base = ImageFolder(root="PlantVillage/train", transform=train_transform)
    test_base = ImageFolder(root="PlantVillage/train", transform=val_transform)
    from collections import Counter

    counts = Counter(train_base.targets)

    print("\nCLASS DISTRIBUTION:\n")

    for idx, count in sorted(counts.items(), key=lambda x: x[1], reverse=True):
        print(f"{idx} -> {train_base.classes[idx]} : {count}")

    with open("classes.json", "w") as f:
        json.dump(train_base.classes, f)
        print("classes loaded")

    targets = train_base.targets
    indices = list(range(len(train_base)))

    train_idx, test_idx = train_test_split(
        indices, test_size=0.2, stratify=targets, random_state=42
    )

    train_dataset = Subset(train_base, train_idx)
    test_dataset = Subset(test_base, test_idx)


    val_dataset = ImageFolder(root="PlantVillage/val", transform=val_transform)
    # train dataset and dataloader for optuna
    train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True, pin_memory=True,num_workers=2)
    # validation dataset and data loader for optuna
    val_dataloader = DataLoader(val_dataset, batch_size=64, pin_memory=True,num_workers=2)
    # test dataset for final check (accuracy)
    test_dataloader = DataLoader(test_dataset, batch_size=64, pin_memory=True,num_workers=2)

    print(len(train_base.classes))
    print(len(test_base.classes))



    class MobileNetV2(nn.Module):
        def __init__(self, neurons_per_hidden_layer, dropout,num_classes):
            super().__init__()
            # loading pretrained mobileNetV2
            self.model = models.mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT)
            image_channel = self.model.last_channel
            layer = []

            for neurons in neurons_per_hidden_layer:
                layer.append(nn.Linear(image_channel, neurons))
                layer.append(nn.ReLU())
                layer.append(nn.BatchNorm1d(neurons))
                layer.append(nn.Dropout(dropout))
                image_channel = neurons
            layer.append(nn.Linear(image_channel, num_classes))
            self.model.classifier = nn.Sequential(*layer)

        def forward(self, x):
            return self.model(x)


    """""

    def objective(trial):
        no_of_hidden_layers = trial.suggest_int("no_of_hidden_layers", 1, 5)
        neurons_per_hidden_layer = []
        dropout = trial.suggest_float("dropout", 0.1, 0.5)
        for i in range(no_of_hidden_layers):
            neurons_per_hidden_layer.append(trial.suggest_int(f"neurons_per_hidden_layer{i}", 8, 128, step=8))
        learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)
        weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)

        mnv2 = MobileNetV2(neurons_per_hidden_layer, dropout,34)
        mnv2.to(device)

        freeze = True

        if freeze:
            for param in mnv2.model.features.parameters():
                param.requires_grad = False

        optimizer = optim.SGD(filter(lambda p: p.requires_grad, mnv2.parameters()), lr=learning_rate,
                              weight_decay=weight_decay, momentum=0.9)
        criterion = nn.CrossEntropyLoss()

        epochs = 5
        for epoch in range(epochs):
            print(f"Trial {trial.number} | Epoch {epoch + 1}/{epochs}")
            mnv2.train()
            for batch_idx, (image, labels) in enumerate(train_dataloader):
                print("batch_idx: ", batch_idx)
                image = image.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                optimizer.zero_grad()
                forward = mnv2(image)
                loss = criterion(forward, labels)
                loss.backward()
                optimizer.step()

        mnv2.eval()
        with torch.no_grad():
            total = 0
            correct = 0
            for image, labels in val_dataloader:
                print("printing accuracy on val set")
                image = image.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                forwardEval = mnv2(image)
                _, predicted = torch.max(forwardEval, 1)
                total += labels.shape[0]
                correct += (predicted == labels).sum().item()
            accuracy = (correct / total) * 100
            return accuracy
    
    

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=10)
    

    best_params = study.best_params
    print(best_params)
    """""



    dropout =  0.14510733657567784
    neurons_per_hidden_layer = [72,72,64]
    learning_rate = 0.007651607466474596/10
    weight_decay =  2.55740335803366e-05

    model= MobileNetV2(neurons_per_hidden_layer, dropout,34)
    model=model.to(device)
    freeze = False
    if freeze:
        for param in model.model.features.parameters():
            param.requires_grad = False
    optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate,
                          weight_decay=weight_decay, momentum=0.9)
    criterion = nn.CrossEntropyLoss()

    epochs=20
    best_accuracy = 0
    for epoch in range(epochs):
        model.train()
        print(f"epoch no:{epoch}")
        for batch_idx,(image,labels) in enumerate(train_dataloader):
            print("batch_idx: ", batch_idx)
            image= image.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            forwardPass = model(image)
            loss = criterion(forwardPass, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        with torch.no_grad():

            total = 0
            correct = 0
            model.eval()
            for batch_idx, (image, labels) in enumerate(val_dataloader):

                image = image.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                forwardPassTest = model(image)
                _, predicted = torch.max(forwardPassTest, 1)
                total += labels.shape[0]
                correct += (predicted == labels).sum().item()
        accuracy = (correct / total) * 100
        print(f"epoch {epoch} accuracy: {accuracy}")

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            torch.save(model.state_dict(), "modelWeights_dummy.pth")





    with torch.no_grad():
        model.load_state_dict(torch.load("modelWeiend.pth", map_location=device))
        model.eval()
        print("testing model now")

        total = 0
        correct = 0
        for batch_idx, (image, labels) in enumerate(test_dataloader):

            image=image.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            forwardPassTest = model(image)
            _, predicted = torch.max(forwardPassTest, 1)
            total += labels.shape[0]
            correct+= (predicted==labels).sum().item()
    accuracy = (correct / total) * 100
    print(f"final accuracy: {accuracy}")

if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()





