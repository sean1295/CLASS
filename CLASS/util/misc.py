import torch
import bisect

def get_episode_bounds(episode_starts, current_index):
    """
    Given a sorted list of episode start indices, find the last and next episode start.

    Args:
        episode_starts (list): Sorted list of episode start indices.
        current_index (int): The current time index.

    Returns:
        (int, int): The last episode start and the next episode start.
    """
    pos = bisect.bisect_right(episode_starts, current_index)
    last_episode_start = episode_starts[pos - 1] if pos > 0 else None
    next_episode_start = episode_starts[pos] if pos < len(episode_starts) else None

    return last_episode_start, next_episode_start


def find_nearest_neighbors_gpu(latent_data, seq_len, k=16, use_topk=True, episode_starts=None, batch_size=2000, gamma=1.0):
    """
    Find k closest neighbors or compute full distance matrix optimized for GPU execution.
    
    Args:
        latent_data: Tensor of shape [num_samples, seq_len, feature_dim] (should be on GPU)
        seq_len: Length of each sequence
        k: Number of neighbors to find (if use_topk is True)
        use_topk: If True, return top-k neighbors; if False, return full distance matrix
        episode_starts: Optional tensor containing episode start indices
        batch_size: Size of batches to process at once (adjust based on GPU memory)
        gamma: Discount factor applied to each time step (default 1.0 = no weighting)
        
    Returns:
        If use_topk is True:
            Tuple of:
            - Tensor of shape [num_samples, k] containing indices of k nearest neighbors
            - Tensor of shape [num_samples, k] containing distances to k nearest neighbors
        If use_topk is False:
            Tensor of shape [num_samples, num_samples] containing full distance matrix
    """
    device = latent_data.device
    if not device.type == 'cuda':
        print("Warning: Data not on GPU. Moving to GPU for faster computation.")
        latent_data = latent_data.cuda()
        device = latent_data.device
    
    num_samples = latent_data.shape[0]
    
    # Time-step weights for discounted L2 (shape: [seq_len])
    weights = (gamma ** torch.arange(seq_len, device=device)).float()

    # Initialize full distance matrix
    full_distances = torch.zeros((num_samples, num_samples), device=device)
    
    for i in range(0, num_samples, batch_size):
        torch.cuda.empty_cache()
        batch_end = min(i + batch_size, num_samples)
        batch_size_actual = batch_end - i
        
        batch_distances = torch.zeros((batch_size_actual, num_samples), device=device)
        
        # Pre-compute squared norms: ||a||²
        sq_norms_all = torch.zeros(num_samples, device=device)
        for t in range(seq_len):
            sq_norms_all += weights[t] * torch.sum(latent_data[:, t] ** 2, dim=1)
        
        # ||a||² for batch samples
        batch_sq_norms = sq_norms_all[i:batch_end].unsqueeze(1)
        batch_distances += batch_sq_norms
        
        # ||b||² for all samples
        batch_distances += sq_norms_all.unsqueeze(0)
        
        # -2⟨a, b⟩ weighted
        for t in range(seq_len):
            dot_product = torch.mm(latent_data[i:batch_end, t], latent_data[:, t].T)
            batch_distances -= 2 * weights[t] * dot_product
        
        full_distances[i:batch_end, :] = batch_distances
    
    if not use_topk:
        return full_distances

    # Exclude self-distances
    for i in range(num_samples):
        full_distances[i, i] = float('inf')
    
    # Exclude same-episode neighbors if episode_starts provided
    if episode_starts is not None:
        for i in range(num_samples):
            mask_start_idx, mask_end_idx = get_episode_bounds(episode_starts, i)
            if mask_start_idx < mask_end_idx and mask_end_idx < num_samples:
                full_distances[i, mask_start_idx:mask_end_idx] = float('inf')
    
    # Get top-k neighbors
    top_values, top_indices = torch.topk(full_distances, k, dim=1, largest=False)
    return top_indices, top_values