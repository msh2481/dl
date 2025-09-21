## How to run the code

```
uv sync
source .venv/bin/activate
python full_finetuning.py --lr 1e-4 --warmup-steps 1000 --fraction 0.1
python evaluate_all.py --batch-size 256 --fraction 0.1
```

## How to fix torch errors on Constructor GPU
Inside DL directory:
```
export PATH="/home/coder/project/dl/.venv/bin:$PATH"
hash -r
unset PYTHONPATH
```