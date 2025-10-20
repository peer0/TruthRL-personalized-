## Installation

Set up the training environemnt for SFT/DPO
```bash
conda create -n truthrl-openr1 python=3.11 -y
conda activate truthrl-openr1
conda install nvidia/label/cuda-12.4.0::cuda-toolkit
pip install vllm==0.9.2
pip install setuptools && pip install flash-attn --no-build-isolation

cd open-r1
GIT_LFS_SKIP_SMUDGE=1 pip install -e ".[dev]"

huggingface-cli login
wandb login
```

Set up the training environemnt for GRPO
```bash
conda create -n truthrl-verl python=3.10 -y
conda activate truthrl-verl
conda install nvidia/label/cuda-12.4.0::cuda-toolkit

cd verl
USE_MEGATRON=0 bash scripts/install_vllm_sglang_mcore.sh

pip install numpy==1.26.1 opentelemetry-sdk==1.26.0 opentelemetry-sdk==1.26.0 click==8.2.1 tensordict==0.8.1

pip install --no-deps -e .

huggingface-cli login
wandb login
```

## Run SFT
Please change `output_dir` in `training/open-r1/recipes/Llama3.1-8B-Instruct/sft/config_sft.yaml`.

```bash
conda activate truthrl-openr1
bash train_sft.sh
```

## Run DPO
Please change `OUTPUT_DIR` in `train_dpo.sh`.

```bash
conda activate truthrl-openr1
bash train_dpo.sh
```

## Run TruthRL
The training requires a LLM verifier to judge whether the predicted answer aligns with the reference answer to produce reward signals.
Provided a LLM verifier is hosted on `localhost:8000` using:
```bash
vllm serve meta-llama/Llama-3.3-70B-Instruct --tensor-parallel-size 4 --port 8000
```

If not hosting locally, please change `OPENAI_API_BASE` in `train_grpo.sh` to the base URL where you host the verifier. Then run:

```bash
conda activate truthrl-verl
bash train_grpo.sh
```

Change reward scheme (i.e., binary/ternary reward) in `training/verl/verl/utils/reward_score/__init__.py` if needed.


After training, run the following command to get HF checkpoints:
```base
python -m verl.model_merger merge --backend fsdp --local_dir <path_to_fsdp_checkpoint>/actor --target_dir <path_to_HF_checkpoint>
```