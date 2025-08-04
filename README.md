# CLASS: Contrastive Learning via Action Sequence Supervision for Robot Manipulation

![CLASS](./media/shiba.gif)
![CLASS](./media/overview.gif)


## Robomimic Fork

We additionally provide the implemention of CLASS using a [forked version](https://github.com/sean1295-robotics/CLASS_robomimic) of robomimic. This implementation currently only supports homogeneous Square task. Use the original implementation if you want to simulate dynamic camera settings. Note that the forked version uses a different Robomimic version (v0.5) from the original implementation (v0.4).

## Installation ☑️
### Clone this repo
```
git clone https://github.com/sean1295/CLASS.git
cd CLASS
```

### Installing using conda🗜

You can install the vitual conda environment using the following command:
```
conda create -n CLASS python=3.10
conda activate CLASS
pip install -r requirements.txt 
pip install -e .
```

### Installing mimicgen:
```
git clone https://github.com/NVlabs/mimicgen.git
cd mimicgen
pip install -e .
cd ..
```

## Prepare Dataset📝
### Generate Heterogeneous Dataset
Prior to generating heterogeneous datasets, you need to first download robomimic/mimicgen hdf5 datasets. See [robomimic](https://robomimic.github.io/docs/v0.4/datasets/robomimic_v0.1.html) (robomimic) and [mimicgen](https://mimicgen.github.io/docs/datasets/mimicgen_corl_2023.html) documentations for the instructions. Download --tasks square --dataset_types ph from robomimic and -dataset_type core --tasks stack_three_d0 from mimicgen. 

After the datasets are downloaded, move each dataset to dataset_dir from the config file (e.g., configs/square_ph_dp_abs). The location of the downloaded dataset should match the ori_dataset_path value from the config file. Make sure to modify dataset_dir value, which is the directory where the transformed dataset will be stored. Once complete, run
```
python CLASS/scripts/generate_dataset.py --config_name square_ph_dp_abs
```

### DTW Pre-computation
Run the following command to precompute pairwise DTW distances between the actions in the dataset. It utilizes multiprocessing, but it is expected to take a while (hours) for larger datasets.
```
python CLASS/scripts/precompute.py --config_name square_ph_dp_abs
```

## Train️‍🔥️🔥️🔥️
Set up wandb from terminal with `wandb login`
### CLASS Pre-training
You can pretrain the policy using the following command:
```
python CLASS/scripts/pretrain.py --config_name square_ph_dp_abs
```

During pre-training, it periodically reports non-parametric rollout results. 

### BC Fine-tuning
You can train the policy using the following command:
```
python CLASS/scripts/finetune.py --config_name square_ph_dp_abs
```

During fine-tuning, it periodically reports both non-parametric and parametric rollout results. If you want to train from scratch without pre-training, set finetune.enabled in the config file to false. 

