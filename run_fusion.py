import time

import torch
import torch.nn as nn

from fuse_utils import *

from tqdm import *

import os 



def get_llama(model):
    import torch
    def skip(*args, **kwargs):
        pass
    # torch.nn.init.kaiming_uniform_ = skip
    # torch.nn.init.uniform_ = skip
    # torch.nn.init.normal_ = skip
    from transformers import LlamaForCausalLM
    model = LlamaForCausalLM.from_pretrained(model, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, device_map = 'auto')
    model.seqlen = 2048
    return model

def get_llava(model):
    def skip(*args, **kwargs):
        pass
    # torch.nn.init.kaiming_uniform_ = skip
    # torch.nn.init.uniform_ = skip
    # torch.nn.init.normal_ = skip

    from llava.model import LlavaLlamaForCausalLM

    model = LlavaLlamaForCausalLM.from_pretrained(model, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, device_map = 'auto')
    model.seqlen = 2048
    return model

def transform_model_tosave(layers):
    for idx in range(len(layers)):
        layer = layers[idx]
        full_2trans = find_fuse_layers(layer)

        for name_2trans, module_2trans in full_2trans.items():
            bias = False if module_2trans.bias is None else True
            newlinear = nn.Linear(module_2trans.in_features, module_2trans.out_features, bias=bias, device=module_2trans.weight.device, dtype=torch.float32)

            weight_2fuse = torch.zeros_like(module_2trans.weight, dtype = torch.float32)
            weight_2fuse += module_2trans.weight.to( dtype = torch.float32)
            if True:
                weight_2fuse += torch.matmul(module_2trans.lora_B.to( dtype = torch.float32), module_2trans.lora_A.to( dtype = torch.float32))

            for coef, add_w in zip(module_2trans.coef_list, module_2trans.add_weight_list):
                if not module_2trans.coef_lora:
                    weight_2fuse += coef.to(device=module_2trans.weight.device, dtype = torch.float32) * add_w.to(device=module_2trans.weight.device, dtype = torch.float32)
                else:
                    weight_2fuse += torch.matmul(coef[0].to(device=module_2trans.weight.device, dtype = torch.float32), coef[1].to(device=module_2trans.weight.device, dtype = torch.float32)) * add_w.to(device=module_2trans.weight.device, dtype = torch.float32)
        
            newlinear.weight.data = weight_2fuse.to(dtype=module_2trans.weight.dtype).data

            name_split = name_2trans.split('.')
            temp_target = getattr(layer, name_split[0])
            setattr(temp_target, name_split[1], newlinear)

    return layers


def find_fuse_layers(module, layers=[FuseLinear], name=''):
    if type(module) in layers:
        return {name: module}
    res = {}
    for name1, child in module.named_children():
        res.update(find_fuse_layers(
            child, layers=layers, name=name + '.' + name1 if name != '' else name1
        ))
    return res

import copy

def run_macro_fusion(args, layers, inps, attention_mask, position_ids, dev):
    fuse_step = args.group_size
    eval_batch_n = 4
    bsz = 8
    inps_eval = torch.zeros(
        (eval_batch_n, inps.shape[1], inps.shape[2], inps.shape[3]), dtype=inps.dtype, device='cuda'
    )
    inps_eval = copy.deepcopy(inps[:eval_batch_n]).to(device = 'cuda')

    if not args.iterative:

        importance_list = full_importance_eval(layers, inps_eval, attention_mask, position_ids)
        removed_list = importance_list[:len(layers)//4]
        print(removed_list)

    import math

    fuse_times = int(math.ceil(len(layers)*args.prune_rate))
    print(f"Fuse times: {fuse_times}")

    fused_index = []
    unchanged_head_idx = -1
    outs_cache = []
    original_outs = None
    dangerous_list = []
    for idx in range(fuse_times):
        layer_max = len(layers)
        importance_list = []
        if args.iterative:

            importance_list, outs_cache = full_importance_eval(layers, inps_eval, attention_mask, position_ids, unchanged_head_idx, outs_cache, original_outs)
            dangerous_list = importance_list[1 : fuse_times - idx]
            fuse_idx = importance_list[0]

        else:
            fuse_ori = removed_list[idx]
            for cnt, layer in enumerate(layers):
                if layer.self_attn.layer_idx == fuse_ori:
                    fuse_idx = cnt
                    break
                else:
                    continue
        fused_index.append(layers[fuse_idx].self_attn.layer_idx)
        print('fuse_layers: ', fused_index)
        
        if fuse_idx == 0:
            target_idx = 1
        else:
            target_idx = fuse_idx -1

        if fuse_idx == 0:
            group = list(range(0, fuse_step))
        else:
            if target_idx - (fuse_step/2-1) < 0:
                group = list(range(0, fuse_step))
            elif fuse_idx + (fuse_step/2-1) > len(layers)-1:
                group = list(range(layer_max-fuse_step, layer_max))
            else:
                group = list(range(target_idx-(int(fuse_step/2)-1), fuse_idx + (int(fuse_step/2)-1)+1))

        fuse_idx_t = fuse_idx - group[0]
        target_idx_t = target_idx -group[0]

        layers_2fuse = nn.ModuleList(
                [layers[layer_idx] for layer_idx in group]
            )

        if group[0] > 0:
            layers_unfuse_left = nn.ModuleList(
                    [layers[layer_idx] for layer_idx in list(range(layer_max)) if layer_idx < group[0]]
                )
        else:
            layers_unfuse_left = nn.ModuleList([])
        unchanged_head_idx = group[0] - 1
        
        if group[-1] < layer_max-1:
            layers_unfuse_right = nn.ModuleList(
                    [layers[layer_idx] for layer_idx in list(range(layer_max)) if layer_idx > group[-1]]
                )
        else:
            layers_unfuse_right = nn.ModuleList([])


        inps_run = compute_inps_run(layers, inps, attention_mask, position_ids, layer_start_idx = group[0], args = args)

        group_importance = [i for i in importance_list if i in group and i not in dangerous_list]
        importance_rank = [group_importance.index(i) / (len(group_importance) - 1) for i in group]
        Fuse_manager = Fuser(layers_2fuse, inps_run, attention_mask, position_ids, fuse_idx_t, target_idx_t, importance_rank, args)

        Fuse_manager.compute_full_states()

        layers_new = Fuse_manager.fuse_one_layer(layers_2fuse, args)

        layers.cpu()
        del layers[fuse_idx]
        layers_new.cuda()

        layers_new = Fuse_manager.update(layers_new, inps_run, inps_eval, args)
        layers.cuda()

        layers = layers_unfuse_left + layers_new + layers_unfuse_right


        del inps_run
        del Fuse_manager

    if args.save:
        layers = transform_model_tosave(layers)
        for i in range(len(layers)):
            layers[i].self_attn.layer_idx = i
    
    print('fuse_layers: ', fused_index)
    return layers

def fusion_sequential(model, dataloader, dev, args):
    print('Starting ...')

    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.layers

    dtype = next(iter(model.parameters())).dtype
    bsz = 8
    
    inps = torch.zeros(
        (args.nsamples//bsz, bsz, model.seqlen, model.config.hidden_size), dtype=dtype, device='cpu'
    )
    cache = {'i': 0, 'attention_mask': None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        def forward(self, inp, **kwargs):

            inps[cache['i']] = inp
            cache['i'] += 1
            cache['attention_mask'] = kwargs['attention_mask']
            cache['position_ids'] = kwargs['position_ids']
            raise ValueError
    layers[0] = Catcher(layers[0])
    
    with torch.no_grad():
        for batch in dataloader:
            try:
                model(batch[0].to(dev))
            except ValueError:
                pass

    layers[0] = layers[0].module

    torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)
    attention_mask = cache['attention_mask']
    position_ids = cache['position_ids']

    print('Ready.')
    model.cpu()
    layers.cuda()

    layers_fused = run_macro_fusion(args, layers, inps, attention_mask, position_ids, dev)
    model.cuda()

    model.model.layers = layers_fused

    model.config.use_cache = use_cache
    model.config.num_hidden_layers = len(model.model.layers)
    
    return model

@torch.no_grad()
def fusion_eval(model, testenc, dev, eval_set, args):
    print('Evaluating ...')

    testenc = testenc.input_ids
    nsamples = testenc.numel() // model.seqlen

    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.layers

    model.model.embed_tokens = model.model.embed_tokens.to(dev)
    model.model.norm = model.model.norm.to(dev)
    model.model.rotary_emb = model.model.rotary_emb.to(dev)
    layers[0] = layers[0].to(dev)

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros(
        (nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device=dev
    )
    cache = {'i': 0, 'attention_mask': None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp
            cache['i'] += 1
            cache['attention_mask'] = kwargs['attention_mask']
            cache['position_ids'] = kwargs['position_ids']
            raise ValueError
    layers[0] = Catcher(layers[0])
    for i in range(nsamples):
        batch = testenc[:, (i * model.seqlen):((i + 1) * model.seqlen)].to(dev)
        try:
            model(batch)
        except ValueError:
            pass
    layers[0] = layers[0].module
    
    layers[0] = layers[0].cpu()
    model.model.embed_tokens = model.model.embed_tokens.cpu()
    model.model.norm = model.model.norm.cpu()
    model.model.rotary_emb = model.model.rotary_emb.cpu()
    torch.cuda.empty_cache()
    
    outs = torch.zeros_like(inps)
    attention_mask = cache['attention_mask']
    position_ids = cache['position_ids']

    for i in tqdm(range(len(layers)), desc= 'Processing...'):
        # print(i)
        layer = layers[i].to(dev)

        for j in range(nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
        layers[i] = layer.cpu()
        del layer
        torch.cuda.empty_cache()
        inps, outs = outs, inps

    if model.model.norm is not None:
        model.model.norm = model.model.norm.to(dev)
    model.lm_head = model.lm_head.to(dev)

    testenc = testenc.to(dev)
    nlls = []
    for i in range(nsamples):
        hidden_states = inps[i].unsqueeze(0)
        if model.model.norm is not None:
            hidden_states = model.model.norm(hidden_states)
        lm_logits = model.lm_head(hidden_states)
        shift_logits = lm_logits[:, :-1, :].contiguous()
        shift_labels = testenc[
            :, (i * model.seqlen):((i + 1) * model.seqlen)
        ][:, 1:]
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        neg_log_likelihood = loss.float() * model.seqlen
        nlls.append(neg_log_likelihood)
    ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * model.seqlen))
    print(ppl.item())
    file_name = "results.txt"
    if 'llama-3' in args.model.lower():
        file_name = "results-llama3.txt"
    if 'llama-2' in args.model.lower():
        file_name = "results-llama2.txt"
    if 'llava' in args.model.lower():
        file_name = "results-llava.txt"    
    if 'tiny' in args.model.lower():
        file_name = "results-tiny-llama.txt"    
    results = f"ppl {eval_set}: " + str(ppl.item()) + ";num_samples: " + str(args.nsamples) + ";model: " + str(args.model) + ';coef_lr: ' + str(args.coef_lr) + ';cali_data: ' + str(args.dataset)
    if args.iterative:
        results = results + ";iterative " + str(args.iterative)
    if args.coef_matrix:
        results = results + ";coef-matrix " + str(args.coef_matrix)
    if args.coef_lora:
        results = results + ";coef-lora " + str(args.coef_lora)
    results = results + ";group_size: " + str(args.group_size)
    results = results + ";min_lora_rank: " + str(args.min_lora_rank)
    results = results + ";max_lora_rank: " + str(args.max_lora_rank)
    results = results + ";prune_rate: " + str(args.prune_rate)
    results = results + '\n'
    if not os.path.exists(file_name):
        with open(file_name, "w") as file:
            file.write(results)
    else:
        with open(file_name, "a") as file:
            file.write(results)

    model.config.use_cache = use_cache


if __name__ == '__main__':
    import argparse
    from datautils import *

    parser = argparse.ArgumentParser()

    parser.add_argument(
        'model', type=str,
        help='Model to load; pass location of hugginface converted checkpoint.'
    )
    parser.add_argument(
        'dataset', type=str, choices=['wikitext2', 'ptb', 'c4'],
        help='Where to extract calibration data from.'
    )
    parser.add_argument(
        '--seed',
        type=int, default=0, help='Seed for sampling the calibration data.'
    )
    parser.add_argument(
        '--nsamples', type=int, default=128,
        help='Number of calibration data samples. (Fine-tuning data for fusegpt)'
    )
    parser.add_argument(
        '--new-eval', action='store_true',
        help='Whether to use the new PTB and C4 eval.'
    )
    parser.add_argument(
        '--coef-lr',
        type=float, default=0.001, help='Initial learning rate for coef.'
    )
    parser.add_argument(
        '--prune-rate',
        type=float, default=0.25, help='Pruning rate for block removal.'
    )
    parser.add_argument(
        '--iterative', action='store_true',
        help='Whether to iterative compute block importance when running fusion.'
    )
    parser.add_argument(
        '--coef-matrix', action='store_true',
        help='Whether to use matrix size coef during fusion.'
    )
    parser.add_argument(
        '--coef-lora', action='store_true',
        help='Whether to use lora decomp for matrix coef during fusion.'
    )
    parser.add_argument(
        '--save', action='store_true',
        help='Whether to save the model (as huggingface format).'
    )
    parser.add_argument(
        '--save-dir', type=str,
        help='The dir to save the fused model.'
    )
    parser.add_argument(
        '--min-lora-rank', type=int, default=128,
        help='Rank of lora for matrix coef decomp.'
    )
    parser.add_argument(
        '--max-lora-rank', type=int, default=128,
        help='Rank of lora for matrix coef decomp.'
    )
    parser.add_argument(
        '--group-size', type=int, default=8,
        help='Size of group. In paper, the group size does not count the block to fuse, but in code we include it for convenience.'
    )

    args = parser.parse_args()

    if 'llava' in args.model.lower():
        model = get_llava(args.model)
    else:
        model = get_llama(args.model)
    model.eval()
    
    dataloader, testloader = get_loaders(
        args.dataset, nsamples=args.nsamples, seed=args.seed, model=args.model, seqlen=model.seqlen
    )

    print("number of data: ", args.nsamples)
    print("model: ", args.model)
    print("group size: ", args.group_size)
    print("cali_data: ", args.dataset)

    tick = time.time()
    fused_model = fusion_sequential(model, dataloader, DEV, args)
    run_time = time.time() - tick
    print("Runtime of Fusion: ", run_time)
    file_name = "results.txt"
    if 'llama-3' in args.model.lower():
        file_name = "results-llama3.txt"
    if 'llama-2' in args.model.lower():
        file_name = "results-llama2.txt"
    if 'llava' in args.model.lower():
        file_name = "results-llava.txt"    
    if 'tiny' in args.model.lower():
        file_name = "results-tiny-llama.txt"
    with open(file_name, "a") as file:
        file.write(f"runtime: {run_time}\n")

    if args.save:
        from transformers import AutoTokenizer 
        tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=False)
        model_name = args.model.split("/")[-1]
        folder_name = f"{model_name}_{args.dataset}_{args.prune_rate}_{args.nsamples}"
        output_dir = os.path.join(args.save_dir, folder_name)
        save_hf_format(fused_model, tokenizer, output_dir = output_dir)

    datasets = ['wikitext2', 'c4-new']
    for dataset in datasets:
        dataloader, testloader = get_loaders(
            dataset, seed=args.seed, model=args.model, seqlen=model.seqlen
        )
        print(dataset)
        eval_set = dataset
        fusion_eval(fused_model, testloader, DEV, eval_set, args)

    print("number of data: ", args.nsamples)
    print("model: ", args.model)
    print("cali_data: ", args.dataset)


