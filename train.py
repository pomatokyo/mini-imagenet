import torch
import matplotlib.pyplot as plt
import numpy as np
import time

def smooth_curve(points, factor=0.8):
    smoothed_points = []
    for point in points:
        if smoothed_points:
            previous = smoothed_points[-1]
            smoothed_points.append(previous * factor + point * (1 - factor))
        else:
            smoothed_points.append(point)
    return smoothed_points

def train(
    train_loader, val_loader, name, model, criterion, optimizer, num_epochs, device
):
    train_losses = []
    val_losses = []

    start_time = time.time()
    # loop epoch train and valid
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data[0].to(device), data[1].to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 200 == 199:  # 每 200 个 mini-batches 打印一次训练损失
                print(
                    f"[Epoch {epoch+1}, Mini-batch {i+1}] loss: {running_loss / 200:.3f}"
                )
                running_loss = 0.0

        print("")
        # 在验证集上评估模型
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for data in val_loader:
                inputs, labels = data[0].to(device), data[1].to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        train_losses.append(running_loss / len(train_loader))
        val_losses.append(val_loss / len(val_loader))
        print(
            f"\nTrain loss after epoch {epoch+1}: {running_loss / len(train_loader):.3f}"
        )
        print(
            f"Validation loss after epoch {epoch+1}: {val_loss / len(val_loader):.3f}"
        )
        print(
            f"Accuracy of the network on the test images: {100 * correct / total:.2f}%"
        )

    end_time = time.time()
    print(f"Train {name} completed in {end_time - start_time:.2f} seconds")
    print("Finished Training")

    # 保存模型
    torch.save(model.state_dict(), f"./weights/{name}.pth")

    # 平滑训练和验证损失曲线
    smooth_train_losses = smooth_curve(train_losses)
    smooth_val_losses = smooth_curve(val_losses)

    # 绘制训练和验证损失曲线
    plt.figure(figsize=(10, 6))
    plt.plot(smooth_train_losses, label="Smoothed Training loss", color='b')
    plt.plot(smooth_val_losses, label="Smoothed Validation loss", color='g')
    plt.plot(train_losses, label="Training loss", color='lightblue', alpha=0.4)
    plt.plot(val_losses, label="Validation loss", color='lightgreen', alpha=0.4)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.tight_layout()

    # 保存图像到本地文件
    plt.savefig(f"./{name}_final_loss_curve.png")
    plt.close()

    print(f"Loss curve saved to {name}_final_loss_curve.png")
