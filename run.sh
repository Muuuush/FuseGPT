export CUDA_VISIBLE_DEVICES=7

MODEL_PATH=""
SAVE_DIR=""

python run_fusion.py $MODEL_PATH wikitext2 --save --save-dir $SAVE_DIR --new-eval --prune-rate 0.05 --nsamples 32 --coef-lr 0.001 --iterative --coef-matrix --coef-lora
