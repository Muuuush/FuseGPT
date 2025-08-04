import numpy as np
import torch

import datasets
from tqdm import tqdm


def set_seed(seed):
    np.random.seed(seed)
    torch.random.manual_seed(seed)


def get_QA(nsamples, seed, seqlen, model_name, model, bsz = 8):
    from datasets import load_dataset
    mmlu_full = load_dataset("cais/mmlu", "all")
    questions = mmlu_full["test"]["question"]
    choices = []
    choices_lists = mmlu_full["test"]["choices"]
    for l in choices_lists:
        choice = "Choices:\n" + "\n".join(l)
        choices.append(choice)

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    def complete_text(text, max_new = 256):
        inputs = tokenizer(text, return_tensors="pt").to("cuda")
        # generate text
        outputs = model.generate(**inputs, max_new_tokens=max_new, pad_token_id=tokenizer.eos_token_id, do_sample=False)
        return tokenizer.decode(outputs[0], skip_special_tokens=True)
    

    import random
    random.seed(seed)
    traindata = []
    file_name = "QAdata.txt"
    try:
        with open(file_name, 'r') as file:
            train_text = file.read()
            trainenc = tokenizer(train_text, return_tensors='pt')
    except FileNotFoundError:
        for i in tqdm(random.sample(range(len(questions)), 2024), desc= 'Generating QA dataset...'):
            traindata.append(complete_text(questions[i] + "\n" + choices[i]))
        train_text = "\n\n".join(traindata)
        trainenc = tokenizer(train_text, return_tensors='pt')
        with open(file_name, 'w') as file:
            file.write(train_text)

    trainloader = []
    for _ in tqdm(range(0, nsamples, bsz), desc= 'Sampling...'):
        batch_i = None
        tar_i = None
        for _ in range(bsz):

            i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
            j = i + seqlen
            inp = trainenc.input_ids[:, i:j]
            tar = inp.clone()
            tar[:, :-1] = -100
            if batch_i is None:
                batch_i = inp
                tar_i = tar
            else:
                batch_i = torch.cat((batch_i, inp), 0)
                tar = torch.cat((tar_i, tar), 0)
        trainloader.append((batch_i, tar_i))
    return trainloader, None

def get_wikitext2(nsamples, seed, seqlen, model, bsz = 8):
    from datasets import load_dataset
    traindata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')
    testdata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')

    from transformers import AutoTokenizer 
    tokenizer = AutoTokenizer.from_pretrained(model, use_fast=False)
    trainenc = tokenizer("\n\n".join(traindata['text']), return_tensors='pt')
    testenc = tokenizer("\n\n".join(testdata['text']), return_tensors='pt')

    import random
    random.seed(seed)
    trainloader = []


    for _ in range(0, nsamples, bsz):
        batch_i = None
        tar_i = None
        for _ in range(bsz):

            i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
            j = i + seqlen
            inp = trainenc.input_ids[:, i:j]
            tar = inp.clone()
            tar[:, :-1] = -100
            if batch_i is None:
                batch_i = inp
                tar_i = tar
            else:
                batch_i = torch.cat((batch_i, inp), 0)
                tar = torch.cat((tar_i, tar), 0)
        # import pdb; pdb.set_trace()
        trainloader.append((batch_i, tar_i))
    return trainloader, testenc

def get_ptb(nsamples, seed, seqlen, model):
    from datasets import load_dataset
    traindata = load_dataset('ptb_text_only', 'penn_treebank', split='train')
    valdata = load_dataset('ptb_text_only', 'penn_treebank', split='validation')

    from transformers import AutoTokenizer 
    tokenizer = AutoTokenizer.from_pretrained(model, use_fast=False)
    trainenc = tokenizer("\n\n".join(traindata['sentence']), return_tensors='pt')
    testenc = tokenizer("\n\n".join(valdata['sentence']), return_tensors='pt')

    import random
    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))
    return trainloader, testenc

def get_c4(nsamples, seed, seqlen, model, bsz = 8):
    from datasets import load_dataset

    traindata = load_dataset(
        'allenai/c4', data_files={'train': 'en/c4-train.00000-of-01024.json.gz'}, split='train'
    )
    valdata = load_dataset(
        'allenai/c4', data_files={'validation': 'en/c4-validation.00000-of-00008.json.gz'}, split='validation'
    )

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model, use_fast=False)

    import random
    random.seed(seed)
    trainloader = []
        
    for _ in range(0, nsamples, bsz):
        batch_i = None
        tar_i = None
        for _ in range(bsz):
            
            while True:
                i = random.randint(0, len(traindata) - 1)
                trainenc = tokenizer(traindata[i]['text'], return_tensors='pt')
                if trainenc.input_ids.shape[1] - seqlen - 1<= 0:
                    continue
                if trainenc.input_ids.shape[1] >= seqlen:
                    break
            i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
            j = i + seqlen
            inp = trainenc.input_ids[:, i:j]
            tar = inp.clone()
            tar[:, :-1] = -100
            if batch_i is None:
                batch_i = inp
                tar_i = tar
            else:
                batch_i = torch.cat((batch_i, inp), 0)
                tar = torch.cat((tar_i, tar), 0)

        trainloader.append((batch_i, tar_i))

    import random
    random.seed(0)
    valenc = []
    for _ in range(256):
        while True:
            i = random.randint(0, len(valdata) - 1)
            tmp = tokenizer(valdata[i]['text'], return_tensors='pt')
            if tmp.input_ids.shape[1] - seqlen - 1<= 0:
                continue
            if tmp.input_ids.shape[1] >= seqlen:
                break
        i = random.randint(0, tmp.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        valenc.append(tmp.input_ids[:, i:j])
    valenc = torch.hstack(valenc)
    class TokenizerWrapper:
        def __init__(self, input_ids):
            self.input_ids = input_ids
    valenc = TokenizerWrapper(valenc)

    return trainloader, valenc 

def get_ptb_new(nsamples, seed, seqlen, model):
    from datasets import load_dataset
    traindata = load_dataset('ptb_text_only', 'penn_treebank', split='train')
    testdata = load_dataset('ptb_text_only', 'penn_treebank', split='test')

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model, use_fast=False)
    trainenc = tokenizer(" ".join(traindata['sentence']), return_tensors='pt')
    testenc = tokenizer(" ".join(testdata['sentence']), return_tensors='pt')

    import random
    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))
    return trainloader, testenc

def get_c4_new(nsamples, seed, seqlen, model):
    from datasets import load_dataset

    traindata = load_dataset(
        'allenai/c4', data_files={'train': 'en/c4-train.00000-of-01024.json.gz'}, split='train'
    )
    valdata = load_dataset(
        'allenai/c4', data_files={'validation': 'en/c4-validation.00000-of-00008.json.gz'}, split='validation'
    )


    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model, use_fast=False)

    import random
    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        while True:
            i = random.randint(0, len(traindata) - 1)
            trainenc = tokenizer(traindata[i]['text'], return_tensors='pt')
            if trainenc.input_ids.shape[1] >= seqlen:
                break
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))

    valenc = tokenizer(' '.join(valdata[:1100]['text']), return_tensors='pt')
    valenc = valenc.input_ids[:, :(256 * seqlen)]

    class TokenizerWrapper:
        def __init__(self, input_ids):
            self.input_ids = input_ids
    valenc = TokenizerWrapper(valenc)

    return trainloader, valenc


def get_loaders(
    name, nsamples=128, seed=0, seqlen=2048, model_name='', model = None
):
    if 'wikitext2' in name:
        return get_wikitext2(nsamples, seed, seqlen, model_name)
    if 'ptb' in name:
        if 'new' in name:
            return get_ptb_new(nsamples, seed, seqlen, model_name)
        return get_ptb(nsamples, seed, seqlen, model_name)
    if 'c4' in name:
        if 'new' in name:
            return get_c4_new(nsamples, seed, seqlen, model_name)
        return get_c4(nsamples, seed, seqlen, model_name)
    if "QA" in name:
        return get_QA(nsamples, seed, seqlen, model_name, model)
