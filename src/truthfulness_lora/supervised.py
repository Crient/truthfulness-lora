"""Original SFT notebook helpers, extracted with portable configuration.

Masking/tokenization/trainer functions are retained; path/model/setup are adapted.
Historical code has no declared project-wide distribution license; see notices.
"""
import gc,json,os
from pathlib import Path
import torch
from datasets import load_from_disk
from peft import LoraConfig,get_peft_model
from transformers import AutoModelForCausalLM,AutoTokenizer,DataCollatorForSeq2Seq,Trainer,TrainingArguments
from transformers.trainer_utils import get_last_checkpoint
from .runtime import require_gpu

def require_allocated_cuda():
    require_gpu(os.environ.get('TL_ALLOW_LOCAL_COMPUTE')=='1')

def inside_project(path):
    path = Path(path)
    if not path.is_absolute():
        path = PROJECT / path
    path = path.resolve()
    if not path.is_relative_to(PROJECT):
        raise ValueError(f"Path must stay inside {PROJECT}: {path}")
    return path
def require_empty_destination(path):
    path = inside_project(path)
    if path.exists() and (not path.is_dir() or any(path.iterdir())):
        raise FileExistsError(
            f"Refusing to overwrite existing results: {path}. "
            "Preserve the existing run before starting another."
        )

def tokenize_example(example, index):
    question = example["question"]
    answer = example["right_answer"]
    if not isinstance(question, str) or not question.strip():
        raise ValueError(f"Example {index}: missing question text.")
    if not isinstance(answer, str) or not answer.strip():
        raise ValueError(f"Example {index}: missing right_answer text.")
    user_text = "Answer truthfully and concisely.\n\n" + f"Question: {question}"
    prompt = [{"role": "user", "content": user_text}]
    conversation = prompt + [{"role": "assistant", "content": answer}]

    # Keep Qwen's existing template defaults, including thinking behavior.
    prompt_text = tokenizer.apply_chat_template(
        prompt, tokenize=False, add_generation_prompt=True
    )
    full_text = tokenizer.apply_chat_template(
        conversation, tokenize=False, add_generation_prompt=False
    )
    if not full_text.startswith(prompt_text):
        raise ValueError(f"Example {index}: prompt is not a prefix of the full chat.")
    # The template already supplies special tokens: do not add them again.
    prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
    encoded = tokenizer(
        full_text, add_special_tokens=False, return_offsets_mapping=True
    )
    all_ids = encoded["input_ids"]
    if all_ids[:len(prompt_ids)] != prompt_ids:
        raise ValueError(f"Example {index}: token boundary mismatch; refusing wrong labels.")

    input_ids = all_ids[:MAX_LENGTH]
    prompt_length = len(prompt_ids)
    labels = [-100] * min(prompt_length, len(input_ids))
    labels += input_ids[prompt_length:]

    # Check actual answer text, not just assistant scaffolding or an EOS token.
    answer_start = full_text.rfind(answer.strip())
    answer_end = answer_start + len(answer.strip())
    if answer_start < len(prompt_text):
        raise ValueError(f"Example {index}: cannot locate the assistant answer.")
    has_answer_token = any(
        label != -100 and end > answer_start and start < answer_end
        for label, (start, end) in zip(
            labels, encoded["offset_mapping"][:MAX_LENGTH]
        )
    )
    if not has_answer_token:
        raise ValueError(
            f"Example {index}: truncation at {MAX_LENGTH} tokens removes all answer "
            "supervision. Review this example; no rows are silently dropped."
        )
    return {
        "input_ids": input_ids,
        "attention_mask": [1] * len(input_ids),
        "labels": labels,
    }

def tokenize_split(name, limit=None):
    dataset = splits[name]
    if limit is not None:
        if len(dataset) < limit:
            raise ValueError(f"Smoke test requires {limit} training examples.")
        dataset = dataset.select(range(limit))
    # Rebuild labels after the masking fix instead of reusing stale map caches.
    return dataset.map(
        tokenize_example,
        with_indices=True,
        remove_columns=dataset.column_names,
        load_from_cache_file=False,
        desc=f"Tokenizing {name}",
    )

def latest_checkpoint():
    run_dir = inside_project(RUN_DIR)
    if not run_dir.is_dir():
        return None
    checkpoint = get_last_checkpoint(str(run_dir))
    return inside_project(checkpoint) if checkpoint is not None else None

def validate_checkpoint(checkpoint):
    checkpoint = inside_project(checkpoint)
    if checkpoint.parent != inside_project(RUN_DIR):
        raise ValueError("Resume checkpoint must be a direct child of RUN_DIR.")
    required = (
        "adapter_config.json", "trainer_state.json", "optimizer.pt",
        "scheduler.pt", "rng_state.pth",
    )
    missing = [name for name in required if not inside_project(checkpoint / name).is_file()]
    weights = ("adapter_model.safetensors", "adapter_model.bin")
    if not any(inside_project(checkpoint / name).is_file() for name in weights):
        missing.append("adapter_model.safetensors (or adapter_model.bin)")
    if missing:
        raise FileNotFoundError(f"Incomplete Trainer checkpoint {checkpoint}: {missing}")
    config = json.loads((checkpoint / "adapter_config.json").read_text())
    if config.get("base_model_name_or_path") != MODEL_ID:
        raise ValueError("Checkpoint base model does not match MODEL_ID.")
    state = json.loads((checkpoint / "trainer_state.json").read_text())
    best = state.get("best_model_checkpoint")
    if best:
        best = inside_project(best)
        if not best.is_dir() or not any(
            inside_project(best / name).is_file() for name in weights
        ):
            raise FileNotFoundError(f"Best-model checkpoint referenced by state is missing: {best}")
    return checkpoint

def run_lora_training(*, smoke=False, checkpoint=None):
    # All model/Trainer objects belong to this call; no previous run is needed.
    require_allocated_cuda()
    if smoke and checkpoint is not None:
        raise ValueError("Smoke testing cannot resume a full-training checkpoint.")
    run_dir = inside_project(SMOKE_RUN_DIR if smoke else RUN_DIR)
    adapter_dir = inside_project(SMOKE_ADAPTER_DIR if smoke else ADAPTER_DIR)
    if checkpoint is None:
        require_empty_destination(run_dir)
    else:
        checkpoint = validate_checkpoint(checkpoint)
    require_empty_destination(adapter_dir)
    model = trainer = None
    try:
        train_data = tokenize_split("train", SMOKE_EXAMPLES if smoke else None)
        val_data = None if smoke else tokenize_split("validation")
        torch.cuda.reset_peak_memory_stats()
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            dtype=torch.float16,
            low_cpu_mem_usage=True,
            local_files_only=True,
        ).to("cuda")
        model.config.use_cache = False
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
        model = get_peft_model(model, LoraConfig(**LORA_PARAMETERS))
        model.print_trainable_parameters()
        collator = DataCollatorForSeq2Seq(
            tokenizer=tokenizer,
            padding=True,
            label_pad_token_id=-100,
            return_tensors="pt",
        )
        parameters = SMOKE_PARAMETERS if smoke else FULL_PARAMETERS
        args = TrainingArguments(output_dir=str(run_dir), **parameters)
        trainer = Trainer(
            model=model,
            args=args,
            train_dataset=train_data,
            eval_dataset=val_data,
            data_collator=collator,
            processing_class=tokenizer,
        )
        if checkpoint is None:
            trainer.train()
        else:
            trainer.train(resume_from_checkpoint=str(checkpoint))
        trainer.save_model(str(adapter_dir))
        tokenizer.save_pretrained(str(adapter_dir))
        print("Adapter saved:", adapter_dir)
        print("Peak GPU memory (GB):", round(torch.cuda.max_memory_allocated() / 1024**3, 2))
    finally:
        trainer = None
        model = None
        gc.collect()
        torch.cuda.empty_cache()

def configure(model_path,split_path,output):
    global PROJECT,MODEL_ID,MAX_LENGTH,SMOKE_EXAMPLES,LORA_PARAMETERS,FULL_PARAMETERS,SMOKE_PARAMETERS
    global RUN_DIR,ADAPTER_DIR,SMOKE_RUN_DIR,SMOKE_ADAPTER_DIR,splits,tokenizer
    PROJECT=Path(output).resolve();MODEL_ID=str(Path(model_path).resolve());MAX_LENGTH=512;SMOKE_EXAMPLES=32
    RUN_DIR=PROJECT/'trainer';ADAPTER_DIR=PROJECT/'adapters/sft'
    SMOKE_RUN_DIR=PROJECT/'smoke';SMOKE_ADAPTER_DIR=PROJECT/'adapters/smoke'
    LORA_PARAMETERS=dict(r=16,lora_alpha=32,lora_dropout=.05,target_modules='all-linear',bias='none',task_type='CAUSAL_LM')
    FULL_PARAMETERS=dict(num_train_epochs=1,per_device_train_batch_size=1,per_device_eval_batch_size=2,
        gradient_accumulation_steps=4,learning_rate=2e-4,fp16=True,gradient_checkpointing=True,
        eval_strategy='steps',eval_steps=500,save_strategy='steps',save_steps=500,save_total_limit=2,
        load_best_model_at_end=True,metric_for_best_model='eval_loss',greater_is_better=False,
        logging_steps=25,report_to='none',seed=42,optim='adamw_torch_fused',lr_scheduler_type='linear',
        weight_decay=0.,adam_beta1=.9,adam_beta2=.999,adam_epsilon=1e-8,max_grad_norm=1.,warmup_steps=0)
    SMOKE_PARAMETERS=dict(max_steps=2,per_device_train_batch_size=1,gradient_accumulation_steps=4,
        learning_rate=2e-4,fp16=True,logging_steps=1,save_strategy='no',report_to='none')
    splits=load_from_disk(str(split_path))
    if len(splits['train'])!=9500 or len(splits['validation'])!=500:
        raise ValueError('Expected frozen 9500/500 source split')
    tokenizer=AutoTokenizer.from_pretrained(MODEL_ID,local_files_only=True,use_fast=True)
    if not tokenizer.is_fast or not tokenizer.chat_template:raise ValueError('Native fast chat tokenizer required')
    if tokenizer.pad_token is None:tokenizer.pad_token=tokenizer.eos_token
    tokenizer.padding_side='right'
    return globals()

def context():
    if 'PROJECT' not in globals():raise RuntimeError('Configure inputs/output through CLI first')
    return globals()
