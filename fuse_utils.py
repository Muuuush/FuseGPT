import torch

import torch.nn as nn

import torch.nn.functional as F

from tqdm import *

import gc

import numpy as np

import os

import copy

import math

DEV = torch.device('cuda:0')

class FuseLinear(torch.nn.Module):
    __constants__ = ['in_features', 'out_features']
    in_features: int
    out_features: int
    weight: torch.Tensor

    def __init__(self, in_features: int, out_features: int, bias: bool = True,
                 device=None, dtype=None, args = None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.coef_matrix = args.coef_matrix
        self.coef_lora = args.coef_lora
        self.lora_rank = args.lora_rank
        self.args = args

        self.weight = nn.Parameter(torch.empty((out_features, in_features), **factory_kwargs))
        self.lora_A = nn.Parameter(self.weight.new_zeros((128, in_features)))
        self.lora_B = nn.Parameter(self.weight.new_zeros((out_features,128)))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        self.weight.requires_grad = False
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features, **factory_kwargs))
        else:
            self.register_parameter('bias', None)
        
        if bias:
            self.fuse_coef_b = nn.Parameter(torch.zeros(out_features, **factory_kwargs))
        
        if not self.coef_matrix:
            self.coef_list = nn.ParameterList([nn.Parameter(torch.zeros((out_features, 1), **factory_kwargs))])
        else: 
            if self.coef_lora:
                lora_tuple_list = nn.ParameterList([nn.Parameter(torch.zeros((out_features, self.lora_rank), **factory_kwargs)), nn.Parameter(torch.zeros((self.lora_rank, in_features), **factory_kwargs))])
                nn.init.kaiming_uniform_(lora_tuple_list[1], a=math.sqrt(5))
                self.coef_list = nn.ParameterList([lora_tuple_list])
            else:
                self.coef_list = nn.ParameterList([nn.Parameter(torch.zeros((out_features, in_features), **factory_kwargs))])

        self.add_weight_list = []
        self.new = True

    
    def do_fuse(self, linear_2fuse):
        if isinstance(linear_2fuse, FuseLinear):
            weight_2fuse = torch.zeros_like(linear_2fuse.weight.data)
            weight_2fuse += linear_2fuse.weight.data
            if True:
                weight_2fuse += torch.matmul(linear_2fuse.lora_B.data, linear_2fuse.lora_A.data)

            for coef, add_w in zip(linear_2fuse.coef_list, linear_2fuse.add_weight_list):
                if not self.coef_lora:
                    weight_2fuse += coef.data.to(device=linear_2fuse.weight.device) * add_w.data.to(device=linear_2fuse.weight.device)
                else:
                    weight_2fuse += torch.matmul(coef[0].data.to(device=linear_2fuse.weight.device), coef[1].data.to(device=linear_2fuse.weight.device)) * add_w.to(device=linear_2fuse.weight.device)
        else:
            weight_2fuse = linear_2fuse.weight.data

        if self.new:
            self.add_weight_list.append(weight_2fuse)
            self.new = False
        else:
            if not self.coef_matrix:
                self.coef_list.append(nn.Parameter(torch.zeros((self.out_features, 1), device=self.weight.device, dtype=self.weight.dtype)))
            else:
                # Compute and store the added weights to save the memory for large models.
                if len(self.coef_list) >= 4 and ('13b' in self.args.model.lower()):
                    fused_weight = torch.zeros_like(self.weight)
                    for coef, add_w in zip(self.coef_list, self.add_weight_list):
                        if not self.coef_lora:
                            fused_weight += coef.to(device=self.weight.device) * add_w.to(device=self.weight.device)
                        else:
                            fused_weight += torch.matmul(coef[0], coef[1]) * add_w.to(device=self.weight.device)
                    self.coef_list.cpu()
                    for add_weight in self.add_weight_list:
                        add_weight.cpu()
                    del self.coef_list, self.add_weight_list

                    if self.coef_lora:
                        lora_tuple_list = nn.ParameterList([nn.Parameter(torch.zeros((self.out_features, self.lora_rank), device=self.weight.device, dtype=self.weight.dtype)), nn.Parameter(torch.zeros((self.lora_rank, self.in_features), device=self.weight.device, dtype=self.weight.dtype))])
                        nn.init.kaiming_uniform_(lora_tuple_list[1], a=math.sqrt(5))
                        self.coef_list = nn.ParameterList([lora_tuple_list])
                    else:
                        self.coef_list = nn.ParameterList([nn.Parameter(torch.zeros((self.out_features, self.in_features), device=self.weight.device, dtype=self.weight.dtype))])
                    self.add_weight_list = [fused_weight.data]

                if self.coef_lora:
                    lora_tuple_list = nn.ParameterList([nn.Parameter(torch.zeros((self.out_features, self.lora_rank), device=self.weight.device, dtype=self.weight.dtype)), nn.Parameter(torch.zeros((self.lora_rank, self.in_features), device=self.weight.device, dtype=self.weight.dtype))])
                    nn.init.kaiming_uniform_(lora_tuple_list[1], a=math.sqrt(5))
                    self.coef_list.append(lora_tuple_list)
                else:
                    self.coef_list.append(nn.Parameter(torch.zeros((self.out_features, self.in_features), device=self.weight.device, dtype=self.weight.dtype)))
            self.add_weight_list.append(weight_2fuse)
        
        assert len(self.coef_list) == len(self.add_weight_list)

        if linear_2fuse.bias is not None:
            self.add_bias = linear_2fuse.bias.data
        else:
            self.add_bias = None


    def forward(self, input: torch.Tensor) -> torch.Tensor:
        fused_weight = torch.zeros_like(self.weight)
        fused_weight += self.weight
        if True:
            fused_weight += torch.matmul(self.lora_B, self.lora_A)
        for coef, add_w in zip(self.coef_list, self.add_weight_list):
            if not self.coef_lora:
                fused_weight += coef.to(device=input.device) * add_w.to(device=input.device)
            else:
                fused_weight += torch.matmul(coef[0], coef[1]) * add_w.to(device=input.device)

        if self.bias is not None:
            fused_bias = self.bias + self.fuse_coef_b * self.add_bias
        else:
            fused_bias = self.bias
        return F.linear(input, fused_weight, fused_bias)

def find_layers(module, layers=[nn.Conv2d, nn.Linear, FuseLinear], name=''):
    if type(module) in layers:
        return {name: module}
    res = {}
    for name1, child in module.named_children():
        res.update(find_layers(
            child, layers=layers, name=name + '.' + name1 if name != '' else name1
        ))
    return res

def save_hf_format(model, tokenizer, output_dir, sub_folder=""):
    # used to save huggingface format, so we can use it for hf.from_pretrained
    model_to_save = model.module if hasattr(model, 'module') else model
    CONFIG_NAME = "config.json"
    WEIGHTS_NAME = "pytorch_model.bin"
    GENE_CONFIG_NAME = "generation_config.json"
    output_dir = os.path.join(output_dir, sub_folder)
    os.makedirs(output_dir, exist_ok=True)
    output_model_file = os.path.join(output_dir, WEIGHTS_NAME)
    output_config_file = os.path.join(output_dir, CONFIG_NAME)
    output_gene_config_file = os.path.join(output_dir, GENE_CONFIG_NAME)
    save_dict = model_to_save.state_dict()
    for key in list(save_dict.keys()):
        if "lora" in key:
            del save_dict[key]
    torch.save(save_dict, output_model_file)
    model_to_save.config.to_json_file(output_config_file)
    model_to_save.generation_config.to_json_file(output_gene_config_file)
    if tokenizer.is_fast:
        tokenizer.save_pretrained(output_dir)
    else:
        tokenizer.save_vocabulary(output_dir)

@torch.no_grad()
def compute_inps_run(layers, inps_initial, attention_mask, position_ids, layer_start_idx, args):
    inps_run = inps_initial
    bsz =  inps_run.shape[0]

    for i in range(len(layers)):
        if i == layer_start_idx:
            return inps_run

        #We store the inps in cpu to save memory in case the model is too large. Runtime will increase.
        # if '7b' in args.model.lower() and args.prune_rate != 0.3:
        #     device = 'cuda'
        # else:
        #     device = 'cpu'
        device = 'cuda'
        
        outs_new = torch.zeros((inps_run.shape[0], inps_run.shape[1], inps_run.shape[2], inps_run.shape[3]), dtype=inps_run.dtype, device=device)
        layer = layers[i]
        for j in range(bsz):
            outs_new[j] = layer(inps_run[j].to(device='cuda'), attention_mask=attention_mask, position_ids=position_ids)[0]

        del layer
        torch.cuda.empty_cache()

        inps_run = outs_new
        del outs_new
        torch.cuda.empty_cache()


@torch.no_grad()
def full_importance_eval(
    layers, inps_eval, attention_mask, position_ids,
    unchanged_head_idx = -1, outs_cache = []
    ):
    eval_batch_n = 4
    if unchanged_head_idx == -1:
        inps_run_f = copy.deepcopy(inps_eval).to(device = 'cuda')
    else:
        inps_run_f = copy.deepcopy(outs_cache[unchanged_head_idx]).to(device = "cuda")
    outs_cache_new = []

    for i in range(unchanged_head_idx + 1, len(layers)):

        outs_new = torch.zeros((eval_batch_n, inps_run_f.shape[1], inps_run_f.shape[2], inps_run_f.shape[3]), dtype=inps_run_f.dtype, device='cuda')
        layer = layers[i]
        for j in range(eval_batch_n):
            outs_new[j] = layer(inps_run_f[j], attention_mask=attention_mask, position_ids=position_ids)[0]

        torch.cuda.empty_cache()

        outs_cache_new.append(outs_new)
        inps_run_f = outs_new
    
    outs_cache_new = copy.deepcopy(outs_cache[0:unchanged_head_idx + 1]) + outs_cache_new
    del outs_cache
    outs_full = outs_new
    del inps_run_f

    inps_run = copy.deepcopy(inps_eval).to(device = 'cuda')
    sim= []
    for i in tqdm(range(len(layers)), desc = 'Importance Calculating...'):

        if i == 0:
            inps_run_t = copy.deepcopy(inps_run).to(device = 'cuda')
        else:
            inps_run_t = copy.deepcopy(outs_cache_new[i - 1]).to(device = "cuda")

        layers_new = nn.ModuleList([layer for j, layer in enumerate(layers) if j != i])
        
        if i < len(layers_new):
            for k in range(i, len(layers_new)):
                outs_new = torch.zeros((eval_batch_n, inps_run_t.shape[1], inps_run_t.shape[2], inps_run_t.shape[3]), dtype=inps_run_t.dtype, device='cuda')

                layer = layers_new[k]
                for j in range(eval_batch_n):
                    outs_new[j] = layer(inps_run_t[j], attention_mask=attention_mask, position_ids=position_ids)[0]

                torch.cuda.empty_cache()

                inps_run_t = outs_new
            
            del inps_run_t
        else:
            outs_new = inps_run_t

        outs_part = outs_new
        sim_i = F.cosine_similarity(outs_full, outs_part, dim = 3).mean()
        sim.append(sim_i)
        del outs_part, outs_new
    del inps_run
    
    print(f"Similarity: {[float(i) for i in sim]}")
    sim = torch.tensor(sim)

    out_idx = torch.argsort(sim, descending = True)

    return out_idx, outs_cache_new

def fuse_importance_eval(
    layers, inps, attention_mask, position_ids,
    eval_args, unchanged_head_idx = -1, outs_cache = []
    ):
    eval_batch_n = 4
    if unchanged_head_idx == -1:
        inps_run_f = copy.deepcopy(inps).to(device = 'cuda')
    else:
        inps_run_f = copy.deepcopy(outs_cache[unchanged_head_idx]).to(device = "cuda")
    outs_cache_new = []

    with torch.no_grad():
        for i in range(unchanged_head_idx + 1, len(layers)):

            outs_new = torch.zeros_like(inps).to(device = "cuda")
            layer = layers[i]
            for j in range(inps.shape[0]):
                outs_new[j] = layer(inps_run_f[j], attention_mask=attention_mask, position_ids=position_ids)[0]

            torch.cuda.empty_cache()

            outs_cache_new.append(outs_new)
            inps_run_f = outs_new
    
    outs_cache_new = copy.deepcopy(outs_cache[0:unchanged_head_idx + 1]) + outs_cache_new
    del outs_cache
    outs_full = outs_new
    del inps_run_f

    inps_run = copy.deepcopy(inps).to(device = 'cuda')
    sim= []
    layers.cpu()
    fuse_step = eval_args.eval_group_size
    for i in tqdm(range(len(layers)), desc = 'Importance Calculating...'):

        fuse_idx = i
        layer_max = len(layers)
        if fuse_idx == 0:
            tmp_group = list(range(0, fuse_step))
        else:
            if fuse_idx - 1 - (fuse_step / 2 - 1) < 0:
                tmp_group = list(range(0, fuse_step))
            elif fuse_idx + (fuse_step / 2 - 1) > len(layers) - 1:
                tmp_group = list(range(layer_max - fuse_step, layer_max))
            else:
                tmp_group = list(range(fuse_idx - 1 - (int(fuse_step / 2) - 1), fuse_idx + (int(fuse_step / 2) - 1) + 1))

        fuse_idx_t = i - tmp_group[0]
        layers_2fuse = copy.deepcopy(nn.ModuleList(
                [layers[layer_idx] for layer_idx in tmp_group]
            )).cuda()

        tmp_unchanged_head_idx = tmp_group[0] - 1
        
        if tmp_group[-1] < len(layers) - 1:
            layers_unfuse_right = nn.ModuleList(
                    [layers[layer_idx] for layer_idx in list(range(len(layers))) if layer_idx > tmp_group[-1]]
                )
        else:
            layers_unfuse_right = nn.ModuleList([])


        if tmp_unchanged_head_idx == -1:
            inps_run_t = copy.deepcopy(inps_run).to(device = 'cuda')
        else:
            inps_run_t = copy.deepcopy(outs_cache_new[tmp_unchanged_head_idx - 1]).to(device = "cuda")
        Fuse_manager = Fuser(layers_2fuse, inps_run_t, attention_mask, position_ids, fuse_idx_t, eval_args)

        # Fuse_manager.compute_full_states()
        Fuse_manager.outs_full = outs_cache_new[tmp_group[-1]][:Fuse_manager.n_batchs].to(device = "cuda")
        Fuse_manager.outs_full_train = copy.deepcopy(outs_cache_new[tmp_group[-1]]).to(device = "cuda")

        fused_layers = Fuse_manager.fuse_one_layer(layers_2fuse, eval_args)
        fused_layers.cuda()
        layers_unfuse_right.cuda()
        fused_layers = Fuse_manager.update(fused_layers, inps_run_t, inps_run_t[:eval_batch_n], eval_args, True)
        layers_new = fused_layers + layers_unfuse_right
        torch.cuda.empty_cache()


        if tmp_unchanged_head_idx == -1:
            inps_run_t = copy.deepcopy(inps_run).to(device = 'cuda')
        else:
            inps_run_t = copy.deepcopy(outs_cache_new[tmp_unchanged_head_idx - 1]).to(device = "cuda")
        with torch.no_grad():

            for k in range(0, len(layers_new)):
                outs_new = torch.zeros_like(inps).to(device = "cuda")

                layer = layers_new[k]
                for j in range(inps.shape[0]):
                    outs_new[j] = layer(inps_run_t[j], attention_mask=attention_mask, position_ids=position_ids)[0]

                torch.cuda.empty_cache()

                inps_run_t = outs_new
            
            del inps_run_t

            outs_part = outs_new
            sim_i = F.cosine_similarity(outs_full, outs_part, dim = 3).mean()
        sim.append(sim_i)

        layers_unfuse_right.cpu()
        del fused_layers
        del outs_part, outs_new
        torch.cuda.empty_cache()

    del inps_run
    layers.cuda()
    
    print(f"Similarity: {[float(i) for i in sim]}")
    sim = torch.tensor(sim)

    out_idx = torch.argsort(sim, descending = True)

    return out_idx, outs_cache_new

class Fuser():
    
    def __init__(self, layers_2fuse, inps, attention_mask, position_ids, fuse_idx, args):
        
        self.layers_2fuse = layers_2fuse
        self.inps = inps
        self.attention_mask = attention_mask
        self.position_ids = position_ids
        self.fuse_idx = fuse_idx
        self.eval_batch_n = 4
        self.args = args
        
        self.outs_full = torch.zeros((self.eval_batch_n, self.inps.shape[1], self.inps.shape[2], self.inps.shape[3]), dtype=self.inps.dtype, device='cpu')
        self.outs_full_train = torch.zeros_like(self.inps)
        self.n_batchs = self.inps.shape[0]
    
    @torch.no_grad()
    def compute_full_states(self):

        for j in range(self.n_batchs):
            
            hidden_states = self.inps[j].to(device='cuda')
            for decoder_layer in self.layers_2fuse:
                hidden_states = decoder_layer(hidden_states, attention_mask=self.attention_mask, position_ids=self.position_ids)[0]
            
            if j < self.eval_batch_n:
                self.outs_full[j] = hidden_states.detach()
            self.outs_full_train[j] = hidden_states.detach()
            self.inps[j] = self.inps[j].to(device='cpu')
        
        hidden_states.cpu()
        del hidden_states
        
    def remove_layer(self, layers_origin, index):
        layers_new = nn.ModuleList([layer for j, layer in enumerate(layers_origin) if j != index])
        layers_origin[index] = layers_origin[index].cpu()
        del layers_origin[index]

        
        return layers_new

    @torch.no_grad()
    def fuse_by_coef(self, layers_origin, fuse_idx, target_idx):
        layer_2fuse = layers_origin[fuse_idx]
        target_idx = target_idx
        
        layer_target = layers_origin[target_idx]
        
        full_2fuse = find_layers(layer_2fuse)
        full_target = find_layers(layer_target)


        for (name_2fuse, module_2fuse), (name_target, module_target) in zip(full_2fuse.items(), full_target.items()):
            bias = False if module_target.bias is None else True
            if not isinstance(module_target, FuseLinear):
                Flinear = FuseLinear(module_target.in_features, module_target.out_features, bias=bias, device=module_target.weight.device, dtype=module_target.weight.dtype, args = self.args)
                state_dict = module_target.state_dict()
                Flinear.load_state_dict(state_dict, strict=False)
            else:
                Flinear = module_target
            Flinear.do_fuse(module_2fuse)

            name_split = name_target.split('.')
            temp_target = getattr(layer_target, name_split[0])
            setattr(temp_target, name_split[1], Flinear)

        
        layers_origin[target_idx] = layer_target
        layers_out = layers_origin

        return layers_out
    
    @torch.no_grad()
    def fuse_one_layer(self, layers_origin, args):
        for i in range(len(layers_origin)):
            if i == self.fuse_idx:
                continue
            target_idx = i
            layers_origin = self.fuse_by_coef(layers_origin, self.fuse_idx, target_idx)
            
        layers_out = self.remove_layer(layers_origin, self.fuse_idx)
        return layers_out


    def update(self, layers_new, inps_run, inps_eval, args, during_eval = False):
        inps_run_f = copy.deepcopy(inps_eval).to(device = 'cuda')

        criterion = nn.MSELoss()
        
        layers_new.cuda()
        layers_new.eval()

        with torch.no_grad():
            for i in range(len(layers_new)):

                outs_new = torch.zeros((self.eval_batch_n, inps_run_f.shape[1], inps_run_f.shape[2], inps_run_f.shape[3]), dtype=inps_run_f.dtype, device='cuda')
                layer = layers_new[i]
                for j in range(self.eval_batch_n):
                    outs_new[j] = layer(inps_run_f[j], attention_mask=self.attention_mask, position_ids=self.position_ids)[0]

                torch.cuda.empty_cache()

                inps_run_f = outs_new
            
            if not during_eval:
                print("Error_pre:", criterion(outs_new.to(device='cuda'), self.outs_full.to(device='cuda')))
        
        inps_run_f = inps_run_f.cpu()

        del inps_run_f

        outs_new = outs_new.cpu()

        del outs_new

        gc.collect(generation=2)
        torch.cuda.empty_cache()
        
        lr = 9.65e-6
        wd = 0 #1e-5
        max_epochs = 20 # if not during_eval else 10
        betas = (0.9, 0.95)
        eps = 1e-8

        coef_params = []
        base_params = []

        for name, param in layers_new.named_parameters():
            if 'coef' in name:
                coef_params.append(param)
            else:
                base_params.append(param)
        
        optimizer = torch.optim.Adam([
            {'params': base_params, 'lr': lr, 'betas': betas, 'eps': eps, 'wd': wd},
            {'params': coef_params, 'lr': args.coef_lr, 'betas': betas, 'eps': eps, 'wd': wd}
        ])
        
        
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs * self.n_batchs)
        
        
        layers_new.cuda()
        layers_new.train()
        
        counter = tqdm(range(max_epochs), desc = 'Fused Layers Updating...') if not during_eval else range(max_epochs)
        for epoch in counter:

            total_loss = 0.0
            for j in range(self.n_batchs):
                
                optimizer.zero_grad()
                hidden_states = inps_run[j].to(device='cuda')
                for decoder_layer in layers_new:
                    hidden_states = decoder_layer(hidden_states, attention_mask=self.attention_mask, position_ids=self.position_ids)[0]
            
                loss = F.kl_div(F.log_softmax(hidden_states, dim=0), F.softmax(self.outs_full_train[j].to(device='cuda'), dim=0), reduction='batchmean')

                loss.backward()
                optimizer.step()

                loss.to('cpu')
                hidden_states.cpu()
                loss = None


                del loss
                del hidden_states
                gc.collect(generation=2)
                torch.cuda.empty_cache()
            
                scheduler.step()

        
        del optimizer, scheduler
        
        layers_new.cuda()
        layers_new.eval()

        inps_run_f = copy.deepcopy(inps_eval).to(device = 'cuda')
        
        with torch.no_grad():
            for i in range(len(layers_new)):

                outs_new = torch.zeros((self.eval_batch_n, inps_run_f.shape[1], inps_run_f.shape[2], inps_run_f.shape[3]), dtype=inps_run_f.dtype, device='cuda')
                layer = layers_new[i]
                for j in range(self.eval_batch_n):
                    outs_new[j] = layer(inps_run_f[j], attention_mask=self.attention_mask, position_ids=self.position_ids)[0]

                torch.cuda.empty_cache()

                inps_run_f = outs_new
            
            if not during_eval:
                print("Error_after:", criterion(outs_new.to(device='cuda'), self.outs_full.to(device='cuda')))
        
        inps_run_f = inps_run_f.cpu()

        del inps_run_f

        outs_new = outs_new.cpu()

        del outs_new
        
        gc.collect(generation=2)
        torch.cuda.empty_cache()

        return layers_new
        