import torch
from sklearn.metrics import accuracy_score, f1_score, roc_curve, auc
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

def evaluate_model(name, model, model_path, valid_dataset, device='cpu', classnum=18):
    # 加载模型参数
    model.load_state_dict(torch.load(model_path))
    model.to(device)
    model.eval()

    # 定义验证数据加载器
    valid_loader = DataLoader(valid_dataset, batch_size=64, shuffle=False)

    y_true = []
    y_pred_probs = []

    with torch.no_grad():
        for inputs, labels in valid_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            probabilities = torch.softmax(outputs, dim=1)

            y_true.extend(labels.cpu().numpy())
            y_pred_probs.extend(probabilities.cpu().numpy())

    y_true = np.array(y_true)
    y_pred_probs = np.array(y_pred_probs)

    # 计算准确率和F1分数
    y_pred = np.argmax(y_pred_probs, axis=1)
    accuracy = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='macro')

    # 计算 Macro-average ROC AUC
    n_classes = classnum
    print(n_classes)
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true == i, y_pred_probs[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute macro-average ROC AUC
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])

    mean_tpr /= n_classes

    macro_roc_auc = auc(all_fpr, mean_tpr)

    print(f'Accuracy: {accuracy:.4f}, F1 Score: {f1:.4f}, Macro-average ROC AUC: {macro_roc_auc:.4f}')

    # 绘制 ROC 曲线
    plt.figure(figsize=(8, 6))
    for i in range(n_classes):
        plt.plot(fpr[i], tpr[i], label=f'Class {i} (AUC = {roc_auc[i]:.2f})')

    plt.plot(all_fpr, mean_tpr, label=f'Macro-average ROC (AUC = {macro_roc_auc:.2f})', linestyle='--', linewidth=2)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()

    # 保存图像到本地文件
    save_path = f'{name}_roc_curve.png'
    plt.savefig(save_path)
    plt.close()

    print(f'ROC curve saved to {save_path}')

    return accuracy, f1, macro_roc_auc
