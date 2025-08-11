# FuseGPT

This repository is the improved implementation for the paper [FuseGPT: Learnable Layers Fusion of Generative Pre-trained Transformers](https://arxiv.org/abs/2411.14507). 

## Dependencies

```bash
conda create -n fusegpt python=3.11
conda activate fusegpt
conda install pytorch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 pytorch-cuda=12.4 -c pytorch -c nvidia
pip install datasets==3.1.0
pip install transformers==4.47.0
pip install accelerate==1.2.0
pip install sentencepiece==0.2.0
pip install protobuf==5.29.1
```
Note: please modify the version of some packages for your own environment.

## Quick Start

### Llama

Download the models from [Huggingface](https://huggingface.co/) (models of huggingface format), then you can run the code run_fusion.py. Set model path as 'MODEL_PATH' and set the save path as 'SAVE_DIR' if you want to save the fused model.

You can run the pre-defined testing script 'run.sh' by:
```bash
bash run.sh
```

Or resetting the hyperparameters to run customized setting.
For example, run 25% sparsity with 1024 fine-tuning data on wikitext2:
```python
python run_fusion.py $MODEL_PATH wikitext2 \ 
--save --save-dir $SAVE_DIR \
--prune-rate 0.25 \
--nsamples 1024 \
--iterative --coef-matrix --coef-lora --new-eval 
```

### LLaVa

If you want to test [LLaVa](https://github.com/haotian-liu/LLaVA/tree/main), you may install LLaVA without building dependencies.
```bash
cd LLaVA
pip install --no-deps -U -e .
```

## Cite

If you found this work useful, please consider citing:

```
@article{pei2024fusegpt,
  title={FuseGPT: Learnable Layers Fusion of Generative Pre-trained Transformers},
  author={Pei, Zehua and Zhen, Hui-Ling and Yu, Xianzhi and Pan, Sinno Jialin and Yuan, Mingxuan and Yu, Bei},
  journal={arXiv preprint arXiv:2411.14507},
  year={2024}
}
```
