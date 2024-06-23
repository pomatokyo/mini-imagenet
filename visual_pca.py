import torch
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


# 定义一个函数来获取模型输出或特征向量
def get_features(model, data_loader, device):
    features = []
    labels = []

    model.eval()
    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            features.extend(outputs.cpu().numpy())
            labels.extend(targets.cpu().numpy())

    return features, labels


# 定义一个函数来进行 PCA 可视化
def visualize_pca(name, model, model_path, valid_loader, device="cpu", n_components=2):
    # 加载模型
    model.load_state_dict(torch.load(model_path))
    model.to(device)
    model.eval()

    # 获取验证集的特征向量或模型输出
    valid_features, valid_labels = get_features(model, valid_loader, device)

    # 使用 PCA 进行降维
    pca = PCA(n_components=n_components)
    pca_result = pca.fit_transform(valid_features)

    # 绘制 PCA 的结果
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(
        pca_result[:, 0], pca_result[:, 1], c=valid_labels, cmap="viridis"
    )
    plt.colorbar(scatter)
    plt.title("PCA Visualization of Model Outputs on Validation Set")
    plt.xlabel(
        f"Principal Component 1 ({pca.explained_variance_ratio_[0]*100:.2f}% Variance Explained)"
    )
    plt.ylabel(
        f"Principal Component 2 ({pca.explained_variance_ratio_[1]*100:.2f}% Variance Explained)"
    )
    plt.savefig(f"./{name}_pca_visualization.png")
    plt.close()
