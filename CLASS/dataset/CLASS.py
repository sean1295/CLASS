import importlib
import torch
import gc

def load_class_from_str(class_path: str):
    module_path, class_name = class_path.rsplit('.', 1)
    module = importlib.import_module(module_path)
    cls = getattr(module, class_name)
    return cls

def make_class_dataset_from_target(target: str, use_sparse: bool = False):
    base_dataset_cls = load_class_from_str(target)
    if use_sparse:
        class SparseCLASSDataset(base_dataset_cls):
            def __init__(self, num_sample, dtw_file_path, dist_quantile=0.02, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.num_sample = num_sample
                self.n = len(self)
                
                # 1. Load sparse distance matrix directly to CPU
                dtw_dist = torch.load(dtw_file_path, map_location='cpu')
                v = dtw_dist.values().half()
                idxs = dtw_dist.indices()
                shape = dtw_dist.shape
                del dtw_dist
                gc.collect()
                
                # Keep only entries within current dataset length
                valid_mask = (idxs[0] < self.n) & (idxs[1] < self.n)
                idxs = idxs[:, valid_mask]
                v = v[valid_mask]
                shape = (self.n, self.n)

                with torch.no_grad():
                    # 2. Thresholding based on dist_quantile
                    k = int(dist_quantile * shape[0] * shape[0])
                    thresh_val = torch.topk(v, k, largest=False).values[-1]
                
                    mask = v <= thresh_val
                    filtered_vals = v[mask]
                    filtered_idxs = idxs[:, mask]
                    del v, idxs, mask
                    gc.collect()
                
                    # 3. Build similarity scores efficiently
                    sample_size = min(1_000_000, filtered_vals.numel())
                    v_sample = filtered_vals[
                        torch.randperm(filtered_vals.numel(), device=filtered_vals.device)[:sample_size]
                    ]
                    sorted_sample = v_sample.sort()[0]
                    del v_sample
                    gc.collect()
                
                    num_bins = 10_000
                    bin_edges = sorted_sample[
                        torch.linspace(0, sample_size, steps=num_bins + 1, device=filtered_vals.device).long().clamp(max=sample_size-1)
                    ]
                    del sorted_sample
                    gc.collect()
                
                    # Compute similarity
                    bin_ids = torch.bucketize(filtered_vals, bin_edges, right=True)
                    del bin_edges
                    gc.collect()
                
                    similarity = 1.0 - bin_ids.half() / num_bins
                    del bin_ids
                    gc.collect()
                
                # 4. Build sparse similarity matrix
                self.dist = torch.sparse_coo_tensor(filtered_idxs, similarity.half(), shape).coalesce()
                del filtered_idxs, similarity
                gc.collect()
                
                # 5. Pre-extract for fast slicing
                i, j = self.dist.indices()
                v = self.dist.values()
                
                sorted_i, sort_idx = i.sort()
                sorted_j = j[sort_idx]
                sorted_v = v[sort_idx]
                del i, j, v, sort_idx
                gc.collect()
                
                # 6. Build row pointer
                row_counts = torch.bincount(sorted_i, minlength=self.n)
                row_ptr = torch.cat([torch.zeros(1, dtype=torch.long), row_counts.cumsum(dim=0)])
                del row_counts, sorted_i
                gc.collect()
                
                # 7. Save precomputed data
                self._row_ptr = row_ptr
                self._sorted_j = sorted_j
                self._sorted_v = sorted_v
        
            def __getitem__(self, _):
                # 1. Sample global indices for the batch (no change, already fast)
                idx = torch.randint(0, self.n, (self.num_sample,), device=self.dist.device)

                # 2. Gather observation data (e.g., images)
                # This remains the #2 bottleneck. If you can pre-process self.train_data
                # into a dictionary of stacked tensors, you can do batch = {k: self.train_data[k][idx]...}
                # for a significant speedup. For now, we focus on the main problem.
                batch = {
                    k: torch.stack([self.train_data[i][k] for i in idx.tolist()])
                    for k in self.obs_keys
                }

                # 3. Create helper mappings for vectorization (no change, already fast)
                idx_bool = torch.zeros(self.n, dtype=torch.bool, device=idx.device)
                idx_bool[idx] = True

                index_map = torch.full((self.n,), -1, dtype=torch.long, device=idx.device)
                index_map[idx] = torch.arange(self.num_sample, device=idx.device)

                # 4. OPTIMIZATION: Vectorized edge extraction using the CSR-like format
                
                # Get the start and end pointers for all batch rows at once
                starts = self._row_ptr[idx]
                ends = self._row_ptr[idx + 1]
                
                # Calculate how many neighbors each node in the batch has
                lengths = ends - starts

                # Create a tensor of local row indices (0 to num_sample-1) repeated by the number of neighbors.
                # This efficiently tells us which batch item each candidate edge belongs to.
                i_local_candidates = torch.repeat_interleave(torch.arange(self.num_sample, device=idx.device), lengths)
                
                # This is the trickiest part: gathering all the slices of columns and values.
                # We construct a single large index tensor to gather all neighbors at once.
                # This avoids the Python loop for slicing.
                all_candidate_indices = torch.cat([torch.arange(s, e, device=idx.device) for s, e in zip(starts, ends)])
                
                # Gather all candidate neighbor columns and their similarity values in one go
                j_global_candidates = self._sorted_j[all_candidate_indices]
                v_candidates = self._sorted_v[all_candidate_indices]

                # Now, from this much smaller set of candidate edges, find which ones
                # connect to another node that is ALSO in the batch.
                final_mask = idx_bool[j_global_candidates]

                # Apply the mask to get the final edges
                i_local_final = i_local_candidates[final_mask]
                v_final = v_candidates[final_mask]
                
                # Get the global column indices for the final edges
                j_global_final = j_global_candidates[final_mask]

                # Map the final global column indices to local batch indices
                j_local_final = index_map[j_global_final]

                # 5. Build the normalized distance matrix in one shot
                dist_vals = torch.zeros((self.num_sample, self.num_sample), device=idx.device, dtype=torch.float32)
                dist_vals[i_local_final, j_local_final] = v_final.float()
                
                dist_vals.fill_diagonal_(0.0)
                batch["dist"] = dist_vals
                batch["pos_mask"] = dist_vals > 0

                return batch
                
        return SparseCLASSDataset
    
    else:        
        class DenseCLASSDataset(base_dataset_cls):
            def __init__(self, num_sample, dtw_file_path, dist_quantile=0.02, *args, **kwargs):
                super().__init__(*args, **kwargs)
                dtw_dist = torch.load(dtw_file_path, weights_only = False)[:len(self),:len(self)]
                device = dtw_dist.device
                self.num_sample = num_sample
                self.all_indices = torch.arange(len(dtw_dist), device=device)
                self.dist = self.get_cdf_dist(dtw_dist, dist_quantile)
                del dtw_dist
                gc.collect()
                self.dist.fill_diagonal_(0)
            
            @staticmethod
            def get_cdf_dist(dist, quantile = 0.02):
                triu_mask = torch.triu(torch.ones_like(dist, dtype=torch.bool), diagonal=1)
                flat = dist[triu_mask]
                
                k = int(flat.numel() * quantile)
                topk_values, topk_indices = torch.topk(flat, k, largest=False, sorted=True)
                print('Threshold: ', topk_values.max())
            
                inv_cdf = torch.linspace(1.0, 0.0, steps=k, device=dist.device)
                dist_vals = torch.zeros_like(flat, dtype = inv_cdf.dtype)
                dist_vals[topk_indices] = inv_cdf
            
                # Build symmetric normalized distance matrix
                dist = torch.zeros_like(dist, dtype = dist_vals.dtype)
                dist[triu_mask] = dist_vals
                dist = dist + dist.T  # Reflect to lower triangle
                return dist
        
            def __getitem__(self, idx):        
                indices = torch.randint(0, self.all_indices.size(0), (self.num_sample,))
                batch = {
                    k: torch.stack([self.train_data[i][k] for i in indices])
                    for k in self.train_data[0].keys()
                }
                batch["dist"] = self.dist[indices][:, indices]
                return batch

        return DenseCLASSDataset

