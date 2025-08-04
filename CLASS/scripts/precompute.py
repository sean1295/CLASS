import os
import gc
import argparse

import torch
NUM_CORES = max(1, int(os.cpu_count() * 0.75))
torch.set_num_threads(NUM_CORES)

import numpy as np
from aeon.distances import distance
from multiprocessing import Pool
from hydra import compose, initialize
from hydra.utils import instantiate
from tqdm import tqdm

def compute_row_chunk(args):
    i_start, i_end, mmap_path, method, action_path, n = args
    mmap = np.memmap(mmap_path, dtype='float32', mode='r+', shape=(n, n))
    arr = np.load(action_path, mmap_mode='r')  # (n, D, T)
    for i in range(i_start, i_end):
        for j in range(i + 1, n):
            d = distance(arr[i], arr[j], method=method)
            mmap[i, j] = d
            mmap[j, i] = d
        mmap[i, i] = 0.0
    return f"Finished rows {i_start} to {i_end}"


def run_dtw_parallel(n, method, action_path, mmap_path, chunk_size=512):
    if not os.path.exists(mmap_path):
        _ = np.memmap(mmap_path, dtype='float32', mode='w+', shape=(n, n))

    args_list = [
        (i, min(i + chunk_size, n), mmap_path, method, action_path, n)
        for i in range(0, n, chunk_size)
    ]
    with Pool(processes=NUM_CORES) as pool:
        for result in tqdm(pool.imap_unordered(compute_row_chunk, args_list), total=len(args_list)):
            pass

    print("Finished computing distance matrix.")


def collect_sparse_entries_chunk(args):
    i_start, i_end, mmap_path, n, threshold = args
    mmap = np.memmap(mmap_path, dtype='float32', mode='r', shape=(n, n))

    row_idx, col_idx, vals = [], [], []
    for i in range(i_start, i_end):
        for j in range(i + 1, n):
            d = mmap[i, j]
            if d <= threshold:
                row_idx.append(i)
                col_idx.append(j)
                vals.append(d)
                row_idx.append(j)
                col_idx.append(i)
                vals.append(d)
    return row_idx, col_idx, vals


def build_sparse_tensor_from_memmap(mmap_path, n, quantile=0.05, sample_size=1_000_000, chunk_size=32):
    mmap = np.memmap(mmap_path, dtype='float32', mode='r', shape=(n, n))
    rng = np.random.default_rng()

    samples = []
    pbar = tqdm(total = sample_size)
    while len(samples) < sample_size:
        i, j = rng.integers(0, n, size=2)
        if i < j:
            samples.append(mmap[i, j])
            pbar.update(1)
    threshold = np.quantile(samples, quantile + 1e-3)
    print(f"Sparse threshold (quantile={quantile}): {threshold:.4f}")

    args_list = [
        (i, min(i + chunk_size, n), mmap_path, n, threshold)
        for i in range(0, n, chunk_size)
    ]

    row_idx_all, col_idx_all, vals_all = [], [], []
    with Pool(NUM_CORES) as pool:
        for row_idx, col_idx, vals in tqdm(pool.imap_unordered(collect_sparse_entries_chunk, args_list), total=len(args_list), desc="Collecting sparse entries"):
            row_idx_all.extend(row_idx)
            col_idx_all.extend(col_idx)
            vals_all.extend(vals)

    indices = torch.tensor([row_idx_all, col_idx_all], dtype=torch.long)
    values = torch.tensor(vals_all, dtype=torch.float16)
    sparse_tensor = torch.sparse_coo_tensor(indices, values, (n, n)).coalesce()
    return sparse_tensor


def main(args):
    with initialize(version_base=None, config_path='../configs/'):
        cfg = compose(config_name=args.config_name)
    cfg.dataset.obs_keys = [key for key in cfg.dataset.obs_keys if 'image' not in key]
    cfg.dataset.device = 'cpu'

    os.makedirs(cfg.base_dir, exist_ok=True)

    actions_file = os.path.join(cfg.base_dir, f"actions.npy")
    dat_file = os.path.join(cfg.base_dir, f"temp.dat")

    print("Loading dataset...")
    dataset = instantiate(cfg.dataset)

    print("Preparing action tensor...")
    raw_actions = torch.stack([
        dataset.train_data[i]['action'] for i in range(len(dataset))
    ]).to(dataset.device)
    actions_tensor = dataset.normalizers['action'].normalize(raw_actions)[:, :cfg.dist_horizon].swapaxes(1, 2).to('cpu')  

    if actions_tensor.shape[1] == 10:
        actions_tensor[:, 3:9] /= 2  # Rescale 6D pose

    print("Saving actions to disk...")
    np.save(actions_file, actions_tensor.numpy())
    n = len(actions_tensor)
    del actions_tensor
    gc.collect()

    print("Running symmetric DTW with multiprocessing...")
    run_dtw_parallel(n=n, method=cfg.dist_metric, action_path=actions_file, mmap_path=dat_file)

    if not cfg.use_sparse:
        print("Loading full DTW matrix...")
        mmap = np.memmap(dat_file, dtype='float32', mode='r', shape=(n, n))
        dists = torch.tensor(np.array(mmap), dtype=torch.float32)
    else:
        print("Loading sparse DTW matrix...")
        dists = build_sparse_tensor_from_memmap(
            mmap_path=dat_file,
            n=n,
            quantile=cfg.dist_quantile
        )
    torch.save(dists, cfg.dtw_file_path)
    # os.remove(dat_file)
    print("Saved sparse DTW matrix to:", cfg.dtw_file_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_name", type=str, required=True)
    parser.add_argument("--config_dir", type=str, default="configs/")
    args = parser.parse_args()
    main(args)


'''
python compute_dtw_matrix.py --config_name square_ph_dp 
'''

'''
python compute_dtw_matrix.py --config_name stack_three_d0_dp 
'''

'''
python compute_dtw_matrix.py --config_name pusht_dp 
'''