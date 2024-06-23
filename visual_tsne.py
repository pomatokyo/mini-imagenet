import torch
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
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
            features.append(outputs.cpu().numpy())
            labels.append(targets.cpu().numpy())

    # 将列表转换为 NumPy 数组，并调整形状
    features = np.vstack(features)
    labels = np.hstack(labels)

    return features, labels

# 定义一个函数来进行 t-SNE 可视化
def visualize_tsne(name, model, model_path, valid_loader, device='cpu', n_components=2, perplexity=30, n_iter=1000):
    # 加载模型
    model.load_state_dict(torch.load(model_path))
    model.to(device)
    model.eval()

    # 获取验证集的特征向量或模型输出
    valid_features, valid_labels = get_features(model, valid_loader, device)

    # 使用 t-SNE 进行降维
    tsne = TSNE(n_components=n_components, perplexity=perplexity, n_iter=n_iter)
    tsne_result = tsne.fit_transform(valid_features)

    # 绘制 t-SNE 的结果
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(tsne_result[:, 0], tsne_result[:, 1], c=valid_labels, cmap='viridis')
    plt.colorbar(scatter)
    plt.title('t-SNE Visualization of Model Outputs on Validation Set')
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    plt.savefig(f"./{name}_tsne_visualization.png")
    plt.close()

