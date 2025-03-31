export TRANSFORMERS_CACHE=./nas-ssd2

pip install --upgrade pip
pip install --upgrade transformers tokenizers
pip install --upgrade "accelerate>=0.26.0"

CUDA_VISIBLE_DEVICES=1 python train.py --dataset SQA --num_epochs 10 --lr 5e-6

pip install bitsandbytes

CUDA_VISIBLE_DEVICES=1 python test.py --dataset SQA --batch_size 256 --temperature 0.7 --max_new_tokens 300


