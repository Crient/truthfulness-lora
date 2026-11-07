"""CPU tensor/state tests; no pretrained weights, downloads or GPU operations."""
import copy
import os
from pathlib import Path
import tempfile
import unittest
import torch
from truthfulness_lora.dpo import train,core

class CoreTests(unittest.TestCase):
    def test_orientation_and_reference_detach(self):
        p=torch.tensor([1.,0.],requires_grad=True);r=torch.tensor([1.,0.],requires_grad=True)
        x=core.dpo_loss(p,r);self.assertAlmostEqual(float(x.detach()),__import__('math').log(2),places=6)
        x.backward();self.assertLess(p.grad[0],0);self.assertGreater(p.grad[1],0);self.assertIsNone(r.grad)

    @unittest.skipUnless(os.environ.get('TL_TEST_TOKENIZER'),'Supply a local pinned tokenizer for optional real-template test')
    def test_real_template_and_supervised_mask(self):
        from transformers import AutoTokenizer
        from truthfulness_lora import supervised
        tok=AutoTokenizer.from_pretrained(os.environ['TL_TEST_TOKENIZER'],local_files_only=True,use_fast=True)
        supervised.tokenizer=tok;supervised.MAX_LENGTH=512
        item={'question':'Synthetic verification query?', 'right_answer':'Synthetic verification response.'}
        f=train.serialize(tok,item['question'],item['right_answer'])
        supervised_item=supervised.tokenize_example(item,0)
        self.assertEqual(supervised_item['input_ids'],f['input_ids'])
        start=next(i for i,v in enumerate(f['mask']) if v)
        stop=f['input_ids'].index(151645,start)
        self.assertTrue(f['mask'][stop]);self.assertFalse(any(f['mask'][stop+1:]))
        self.assertTrue(all((label!=-100)==(i>=next(j for j,v in enumerate(f['mask']) if v)) for i,label in enumerate(supervised_item['labels'])))
        with self.assertRaises(AssertionError):train.serialize(tok,item['question'],'word '*3000)

    def test_real_tiny_peft_atomic_recovery(self):
        from transformers import Qwen3Config,Qwen3ForCausalLM
        from peft import LoraConfig,get_peft_model,get_peft_model_state_dict
        torch.manual_seed(42);torch.set_num_threads(1)
        config=Qwen3Config(vocab_size=64,hidden_size=16,intermediate_size=32,num_hidden_layers=1,num_attention_heads=2,num_key_value_heads=1,head_dim=8)
        model=get_peft_model(Qwen3ForCausalLM(config),LoraConfig(r=4,lora_alpha=8,target_modules=['q_proj','v_proj'],task_type='CAUSAL_LM',lora_dropout=0))
        ref=copy.deepcopy(model).requires_grad_(False).eval();core.assert_isolated(model,ref)
        features={'x':[{'input_ids':[1,3,4,5,2],'mask':[False,False,True,True,True]},
                       {'input_ids':[1,3,6,7,2],'mask':[False,False,True,True,True]}]}
        with torch.no_grad():scores={'x':train.score(ref,features['x'],0,'cpu').tolist()}
        opt,scheduler=train.optimizer_for(model)
        train.update(model,opt,scheduler,['x'],features,scores,0,core,'cpu')
        with tempfile.TemporaryDirectory() as d:
            path=train.checkpoint(Path(d),model,opt,scheduler,dict(epoch=0,cursor=1,updates=1,order=['x','x'],best_checkpoint=None))
            train.update(model,opt,scheduler,['x'],features,scores,0,core,'cpu')
            expected={k:v.clone() for k,v in get_peft_model_state_dict(model).items()};rng=torch.rand(3)
            state=train.restore(path,model,opt,scheduler);self.assertEqual(state['cursor'],1)
            train.update(model,opt,scheduler,['x'],features,scores,0,core,'cpu')
            self.assertTrue(all(torch.equal(v,get_peft_model_state_dict(model)[k]) for k,v in expected.items()))
            self.assertTrue(torch.equal(rng,torch.rand(3)))
            self.assertFalse(any(p.name.startswith('.tmp') for p in path.parent.iterdir()))
        self.assertTrue(all(p.grad is None for p in ref.parameters()))

if __name__=='__main__':unittest.main()
