"""Original DPO loss/model-isolation helpers with portable base location."""
import math
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM
from peft import PeftModel
from .common import BASE

def dpo_loss(policy,reference,beta=.1):
    assert math.isfinite(beta) and beta>0
    assert policy.numel()%2==0 and policy.shape==reference.shape
    p=policy.reshape(-1,2);r=reference.detach().reshape(-1,2)
    z=beta*((p[:,0]-p[:,1])-(r[:,0]-r[:,1]))
    return F.softplus(-z).mean()

def disable_dropout(model):
    for m in model.modules():
        if isinstance(m,torch.nn.Dropout): m.p=0.
    model.config.use_cache=False

def freeze_reference(model):
    model.requires_grad_(False);model.eval();disable_dropout(model)
    return model

def load_effective(adapter,device,trainable=False):
    base=AutoModelForCausalLM.from_pretrained(str(BASE),local_files_only=True,torch_dtype=torch.float32,attn_implementation='eager')
    model=PeftModel.from_pretrained(base,str(adapter),is_trainable=trainable).to(device)
    disable_dropout(model)
    if trainable:
        assert all('lora_' in n for n,p in model.named_parameters() if p.requires_grad)
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
    else: freeze_reference(model)
    return model

def assert_isolated(policy,reference):
    assert not any(p.requires_grad for p in reference.parameters()) and not reference.training
    assert not {p.data_ptr() for p in policy.parameters()} & {p.data_ptr() for p in reference.parameters()}
