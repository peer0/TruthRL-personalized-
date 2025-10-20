## Installation

Set up the evaluation environemnt
```bash
conda create -n truthrl-eval python=3.10 -y
conda activate truthrl-eval
pip install -r requirements.txt
```


## Run Evaluation

The evaluation requires a LLM to judge whether the predicted answer aligns with the reference answer. Provided a LLM verifier is hosted on `localhost:8000` using:
```bash
vllm serve meta-llama/Llama-3.3-70B-Instruct --tensor-parallel-size 4 --port 8000
```

If not hosting locally, please change `api_url` in `evaluate.py` to the base URL where you host the verifier. Then run:
```bash
conda activate truthrl-eval
python evaluate.py
```
