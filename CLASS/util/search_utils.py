import torch

def retrieve_nearest_neighbors(target_latent, retrieval_latent, n=1, use_cossim=True):
    """Find n closest neighbors for the target point using either Euclidean distance or cosine similarity."""
    if use_cossim:
        normalized_target = torch.nn.functional.normalize(target_latent, p=2, dim=-1)
        normalized_retrieval = torch.nn.functional.normalize(retrieval_latent, p=2, dim=-1)
        distances = 1 - torch.matmul(normalized_retrieval, normalized_target)
    else:
        distances = torch.cdist(target_latent.unsqueeze(0), retrieval_latent, p=2)[0]

    indices = torch.topk(distances, k=n, largest=False).indices
    return indices, distances