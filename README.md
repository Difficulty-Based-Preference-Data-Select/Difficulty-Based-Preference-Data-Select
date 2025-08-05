# Difficulty-based Preference Data Selection

This repository implements the difficulty-based preference data selection method proposed in the paper "Difficulty-based Preference Data Selection by DPO Implicit Reward Gap". The approach identifies challenging preference examples by computing the DPO implicit reward gap between chosen and rejected responses, then selects samples with smaller gaps (higher difficulty) for more efficient model alignment training.

Our method demonstrates that using only 10% of carefully selected preference data can achieve performance comparable to training on the full dataset. The selection process is based on the insight that examples with smaller reward gaps provide stronger learning signals and higher information content, making them more valuable for alignment training.

## Environment

```bash
conda create -n data_select python=3.10
conda activate data_select
pip install -r requirements.txt
```

## Implementation and Usage

The code is designed to work with datasets in the format of [Skywork/Skywork-Reward-Preference-80K-v0.2](https://huggingface.co/datasets/Skywork/Skywork-Reward-Preference-80K-v0.2/). If using other datasets, you may need to preprocess them to match this format (containing `prompt`, `chosen`, and `rejected` fields).

### Data Selection

Code location: `src/data_select.py`

Key variables to configure:
```python
dpo_model_path = "<dpo_model_path>"        # Path to your DPO policy model
ref_model_path = "<ref_model_path>"        # Path to your reference model  
selection_ratio = "<selection_ratio>"      # e.g., "0.1" for 10% selection
dataset_path = "<dataset_path>"            # Path to your dataset
output_path = "<output_path>"              # e.g., "./results/selected_dataset.json"
```

### Reward Model Training

Reference: https://github.com/RLHFlow/RLHF-Reward-Modeling/tree/main/bradley-terry-rm

Launch script: `src/reward_model_training.py`

Required specifications: dataset path, base model, save path

### DPO Training

Reference: https://github.com/OpenRLHF/OpenRLHF

Launch script: `src/DPO_training.sh`

Required specifications: dataset path, base model, save path