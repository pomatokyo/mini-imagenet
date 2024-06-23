import torch
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import umap

# 定义一个函数来获取模型输出或特征向量
def get_features(model, data_loader, device):
    features = []
    labels = []

    model.eval()
    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            features.append(outputs.cpu().numpy())
            labels.append(targets.cpu().numpy())

    # 将列表转换为 NumPy 数组，并调整形状
    features = np.vstack(features)
    labels = np.hstack(labels)

    return features, labels

# 定义一个函数来进行 UMAP 可视化
def visualize_umap(name, model, model_path, valid_loader, device='cpu', n_components=2, n_neighbors=15, min_dist=0.1):
    # 加载模型
    model.load_state_dict(torch.load(model_path))
    model.to(device)
    model.eval()

    # 获取验证集的特征向量或模型输出
    valid_features, valid_labels = get_features(model, valid_loader, device)

    # 使用 UMAP 进行降维
    umap_model = umap.UMAP(n_components=n_components, n_neighbors=n_neighbors, min_dist=min_dist)
    umap_result = umap_model.fit_transform(valid_features)

    # 绘制 UMAP 的结果
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(umap_result[:, 0], umap_result[:, 1], c=valid_labels, cmap='viridis')
    plt.colorbar(scatter)
    plt.title('UMAP Visualization of Model Outputs on Validation Set')
    plt.xlabel('UMAP Component 1')
    plt.ylabel('UMAP Component 2')
    plt.savefig(f"./{name}_umap_visualization.png")
    plt.close()
