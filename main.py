import torch
import torchvision
import torchvision.transforms as transforms
import argparse
from torch.utils.data import DataLoader
import torch.optim as optim
import matplotlib.pyplot as plt
from torch import nn
import numpy as np
from train import train
from evaluate import evaluate_model
from visual import visual

# 设置训练设备
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Training on {device}.")

# 图片转张量，转化方法
transforms = transforms.Compose(
    [
        transforms.RandomResizedCrop(224),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


def run_alexnet_without_freeze(
    train_loader, val_loader, num_epochs, num_classes, lr, weight_decay, val_set
):
    alexnet_without_freeze = torch.hub.load(
        "pytorch/vision:v0.10.0", "alexnet", num_classes=num_classes, weights=None
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        alexnet_without_freeze.parameters(), lr=lr, weight_decay=weight_decay
    )

    train(
        train_loader=train_loader,
        val_loader=val_loader,
        name="alexnet_without_freeze",
        model=alexnet_without_freeze,
        criterion=criterion,
        optimizer=optimizer,
        num_epochs=num_epochs,
        device=device,
    )

    evaluate_model(
        name="alexnet_without_freeze",
        model=alexnet_without_freeze,
        model_path="./weights/alexnet_without_freeze.pth",
        valid_dataset=val_set,
        device=device,
    )

    visual(
        name="alexnet_without_freeze",
        model=alexnet_without_freeze,
        model_path="./weights/alexnet_without_freeze.pth",
        valid_loader=val_loader,
        device=device,
    )


def run_alexnet_with_freeze(
    train_loader, val_loader, num_epochs, num_classes, lr, weight_decay, val_set
):
    alexnet_with_freeze = torch.hub.load(
        "pytorch/vision:v0.10.0", "alexnet", pretrained=True
    ).to(device)

    for param in alexnet_with_freeze.parameters():
        param.requires_grad = False

    for param in alexnet_with_freeze.classifier.parameters():
        param.requires_grad = True

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        alexnet_with_freeze.classifier.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )

    train(
        train_loader=train_loader,
        val_loader=val_loader,
        name="alexnet_with_freeze",
        model=alexnet_with_freeze,
        criterion=criterion,
        optimizer=optimizer,
        num_epochs=num_epochs,
        device=device,
    )

    evaluate_model(
        name="alexnet_with_freeze",
        model=alexnet_with_freeze,
        model_path="./weights/alexnet_with_freeze.pth",
        valid_dataset=val_set,
        device=device,
    )

    visual(
        name="alexnet_with_freeze",
        model=alexnet_with_freeze,
        model_path="./weights/alexnet_with_freeze.pth",
        valid_loader=val_loader,
        device=device,
    )


def run_resnet101_without_freeze(
    train_loader, val_loader, num_epochs, num_classes, lr, weight_decay, val_set
):
    resnet101_without_freeze = torch.hub.load(
        "pytorch/vision:v0.10.0",
        "resnet101",
        num_classes=num_classes,
        weights=None,
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        resnet101_without_freeze.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )

    train(
        train_loader=train_loader,
        val_loader=val_loader,
        name="resnet101_without_freeze",
        model=resnet101_without_freeze,
        criterion=criterion,
        optimizer=optimizer,
        num_epochs=num_epochs,
        device=device,
    )

    evaluate_model(
        name="resnet101_without_freeze",
        model=resnet101_without_freeze,
        model_path="./weights/resnet101_without_freeze.pth",
        valid_dataset=val_set,
        device=device,
    )

    visual(
        name="resnet101_without_freeze",
        model=resnet101_without_freeze,
        model_path="./weights/resnet101_without_freeze.pth",
        valid_loader=val_loader,
        device=device,
    )


def run_resnet101_with_freeze(
    train_loader, val_loader, num_epochs, num_classes, lr, weight_decay, val_set
):
    resnet101_with_freeze = torch.hub.load(
        "pytorch/vision:v0.10.0", "resnet101", pretrained=True
    ).to(device)

    for param in resnet101_with_freeze.parameters():
        param.requires_grad = False

    for param in resnet101_with_freeze.fc.parameters():
        param.requires_grad = True

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        resnet101_with_freeze.fc.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )

    train(
        train_loader=train_loader,
        val_loader=val_loader,
        name="resnet101_with_freeze",
        model=resnet101_with_freeze,
        criterion=criterion,
        optimizer=optimizer,
        num_epochs=num_epochs,
        device=device,
    )

    evaluate_model(
        name="resnet101_with_freeze",
        model=resnet101_with_freeze,
        model_path="./weights/resnet101_with_freeze.pth",
        valid_dataset=val_set,
        device=device,
    )

    visual(
        name="resnet101_with_freeze",
        model=resnet101_with_freeze,
        model_path="./weights/resnet101_with_freeze.pth",
        valid_loader=val_loader,
        device=device,
    )


def main(args):
    # (图片数组，标签) -> 图片数组 3 * 224 * 224
    train_set = torchvision.datasets.ImageFolder(
        f"./{args.data_path}/train", transform=transforms
    )
    val_set = torchvision.datasets.ImageFolder(
        f"./{args.data_path}/val", transform=transforms
    )

    # loader
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False)

    num_epochs = args.epochs

    # # alexnet_without_freeze loss and optimizer
    # run_alexnet_without_freeze(
    #     train_loader=train_loader,
    #     val_loader=val_loader,
    #     num_epochs=num_epochs,
    #     num_classes=args.num_classes,
    #     lr=args.lr,
    #     weight_decay=args.weight_decay,
    #     val_set=val_set,
    # )

    # # alexnet_with_freeze loss and optimizer
    # run_alexnet_with_freeze(
    #     train_loader=train_loader,
    #     val_loader=val_loader,
    #     num_epochs=num_epochs,
    #     num_classes=args.num_classes,
    #     lr=args.lr,
    #     weight_decay=args.weight_decay,
    #     val_set=val_set,
    # )

    # # resnet101_without_freeze loss and optimizer
    # run_resnet101_without_freeze(
    #     train_loader=train_loader,
    #     val_loader=val_loader,
    #     num_epochs=num_epochs,
    #     num_classes=args.num_classes,
    #     lr=args.lr,
    #     weight_decay=args.weight_decay,
    #     val_set=val_set,
    # )

    # resnet101_with_freeze loss and optimizer
    run_resnet101_with_freeze(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=num_epochs,
        num_classes=args.num_classes,
        lr=args.lr,
        weight_decay=args.weight_decay,
        val_set=val_set,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_classes", type=int, default=18)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--data-path", type=str, default="./imagenet-mini")

    opt = parser.parse_args()

    main(opt)
