from visual_pca import visualize_pca
from visual_tsne import visualize_tsne
from visual_umap import visualize_umap


def visual(name, model, model_path, valid_loader, device):
    visualize_pca(
        name=name,
        model=model,
        model_path=model_path,
        valid_loader=valid_loader,
        device=device,
    )

    visualize_tsne(
        name=name,
        model=model,
        model_path=model_path,
        valid_loader=valid_loader,
        device=device,
    )

    visualize_umap(
        name=name,
        model=model,
        model_path=model_path,
        valid_loader=valid_loader,
        device=device,
    )
