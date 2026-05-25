#!/usr/bin/env python3
"""
Fine-tune BERT/RoBERTa on GLUE tasks with ClearML logging.
Usage:
  python finetune_glue.py --model bert-base-uncased --task sst2 --epochs 3
"""

import os
import argparse
import numpy as np

from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
    EvalPrediction,
)
import evaluate

from adagram_optimizers.AdamGram import AdamGram, SymAdamGram, EQAdamGram, SVDAdamGram
from adagram_optimizers.AdaGram_eq import AdaGramEQ
from adagram_optimizers.AdagramSVD import AdaGramFR
from adagram_optimizers.SymAdaGram import SymAdaGram
# ── ClearML ────────────────────────────────────────────────────────────────────
from clearml import Task

# ── GLUE task → column names and metric ───────────────────────────────────────
TASK_CONFIG = {
    "cola":  {"keys": ("sentence", None),           "metric": "matthews_correlation"},
    "sst2":  {"keys": ("sentence", None),           "metric": "accuracy"},
    "mrpc":  {"keys": ("sentence1", "sentence2"),   "metric": "f1"},
    "qqp":   {"keys": ("question1", "question2"),   "metric": "f1"},
    "mnli":  {"keys": ("premise", "hypothesis"),    "metric": "accuracy"},
    "qnli":  {"keys": ("question", "sentence"),     "metric": "accuracy"},
    "rte":   {"keys": ("sentence1", "sentence2"),   "metric": "accuracy"},
    "wnli":  {"keys": ("sentence1", "sentence2"),   "metric": "accuracy"},
    "stsb":  {"keys": ("sentence1", "sentence2"),   "metric": "pearson"},
}


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune BERT/RoBERTa on a GLUE task")
    parser.add_argument("--model",      type=str, default="bert-base-uncased",
                        help="HuggingFace model name or local path")
    parser.add_argument("--task",       type=str, default="sst2",
                        choices=list(TASK_CONFIG.keys()), help="GLUE task name")
    parser.add_argument("--epochs",     type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr",         type=float, default=2e-5)
    parser.add_argument("--max_len",    type=int, default=128)
    parser.add_argument("--optim",      type=str, default="adamw_torch",
                        help="Optimizer: adamw_torch | adamw_torch_fused | adafactor | sgd ...")
    parser.add_argument("--output_dir", type=str, default="./output")
    parser.add_argument("--project",    type=str, default="GLUE Fine-tuning",
                        help="ClearML project name")
    return parser.parse_args()

def configure_optimizers(model, weight_decay, learning_rate, betas, device_type, opt_name='AdamW', rank=2):
        # start with all of the candidate parameters

    param_dict = {pn: p for pn, p in model.named_parameters()}
    # filter out those that do not require grad
    param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
    # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
    # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
    decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
    nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
    optim_groups = [
        {'params': decay_params, 'weight_decay': weight_decay},
        {'params': nodecay_params, 'weight_decay': 0.0}
    ]
    num_decay_params = sum(p.numel() for p in decay_params)
    num_nodecay_params = sum(p.numel() for p in nodecay_params)
    print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
    print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
    print("optimizer", opt_name)
    if opt_name == 'AdaGram':
        print("RANK", rank)
        optimizer = AdamGram(optim_groups, lr=learning_rate, max_rank=rank)
    if opt_name == 'SymAdaGram':
        print("RANK", rank)
        optimizer = SymAdaGram(optim_groups, lr=learning_rate, max_rank=rank)
    if opt_name == 'SymAdamGram':
        print("RANK", rank)
        optimizer = SymAdamGram(optim_groups, lr=learning_rate, max_rank=rank)
    if opt_name == 'AdaGramSVD':
        print("RANK", rank)
        optimizer = AdaGramFR(optim_groups, lr=learning_rate, max_rank=rank)
    if opt_name == 'AdaGramEQ':
        optimizer = AdaGramEQ(optim_groups, lr=learning_rate, max_rank=rank, enable_logging=False)
    if opt_name == 'EQAdamGram':
        optimizer = EQAdamGram(optim_groups, lr=learning_rate, max_rank=rank, enable_logging=False)
    if opt_name == 'SVDAdamGram':
        optimizer = SVDAdamGram(optim_groups, lr=learning_rate, max_rank=rank, enable_logging=False)
    if opt_name == 'AdamW':
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
    if opt_name == 'SGD':
        optimizer = torch.optim.SGD(optim_groups, lr=learning_rate)
    if opt_name == 'AdaGrad':
        optimizer = torch.optim.Adagrad(optim_groups, lr=learning_rate)
    if opt_name == 'MuonWithAdamW':
        param_groups = [
            dict(params=decay_params, use_muon=True, lr=learning_rate, weight_decay=weight_decay),
            dict(params=nodecay_params, use_muon=False, lr=learning_rate, betas=betas, weight_decay=weight_decay),
        ]
        optimizer = MuonWithAuxAdam(param_groups)
    # optimizer = AdagramAdam(optim_groups, lr=learning_rate, max_rank=rank)
    # print(f"using fused AdamW: {use_fused}")
    return optimizer


def get_num_labels(task: str, dataset) -> int:
    if task == "stsb":
        return 1  # regression
    return dataset["train"].features["label"].num_classes


def build_tokenize_fn(tokenizer, key1: str, key2, max_len: int):
    def tokenize(batch):
        if key2 is None:
            return tokenizer(batch[key1], truncation=True, max_length=max_len)
        return tokenizer(batch[key1], batch[key2], truncation=True, max_length=max_len)
    return tokenize


def build_compute_metrics(task: str):
    metric_name = TASK_CONFIG[task]["metric"]
    metric      = evaluate.load("glue", task)

    def compute_metrics(p: EvalPrediction):
        preds = p.predictions
        if task == "stsb":
            preds = preds.squeeze()
        else:
            preds = np.argmax(preds, axis=1)
        return metric.compute(predictions=preds, references=p.label_ids)

    return compute_metrics


def main():
    args = parse_args()
    cfg  = TASK_CONFIG[args.task]

    # ── Init ClearML task BEFORE anything else ─────────────────────────────────
    task = Task.init(
        project_name=args.project,
        task_name=f"{args.model.split('/')[-1]}-{args.task}",
        auto_connect_frameworks=True,   # auto-logs Trainer metrics
    )
    # Log all CLI args as hyperparameters
    task.connect(vars(args), name="hyperparameters")

    # ── Load dataset & tokenizer ───────────────────────────────────────────────
    print(f"[*] Loading dataset: GLUE/{args.task}")
    raw = load_dataset("nyu-mll/glue", args.task)

    tokenizer    = AutoTokenizer.from_pretrained(args.model)
    tokenize_fn  = build_tokenize_fn(tokenizer, *cfg["keys"], args.max_len)
    tokenized    = raw.map(tokenize_fn, batched=True, remove_columns=raw["train"].column_names
                           if args.task not in ("mnli",) else None)

    # For MNLI use matched validation split
    eval_split = "validation_matched" if args.task == "mnli" else "validation"

    # ── Model ──────────────────────────────────────────────────────────────────
    num_labels = get_num_labels(args.task, raw)
    print(f"[*] Loading model: {args.model}  (num_labels={num_labels})")
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model, num_labels=num_labels
    )

    # ── Training arguments ─────────────────────────────────────────────────────
    output_dir = os.path.join(args.output_dir, args.task)
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.lr,
        weight_decay=0.01,
        optim=args.optim,                       # ← optimizer choice
        lr_scheduler_type="linear",
        warmup_ratio=0.06,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model=cfg["metric"],
        greater_is_better=True,
        logging_steps=50,
        report_to="clearml",                    # ← ClearML logging
        fp16=True,                              # mixed precision (disable if no GPU)
        dataloader_num_workers=4,
    )

    # ── Optional: custom optimizer with per-parameter weight decay ─────────────
    # Uncomment to use a fully custom optimizer instead of --optim flag:
    #
    # from torch.optim import AdamW
    # no_decay = ["bias", "LayerNorm.weight"]
    # param_groups = [
    #     {"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
    #      "weight_decay": 0.01},
    #     {"params": [p for n, p in model.named_parameters() if     any(nd in n for nd in no_decay)],
    #      "weight_decay": 0.0},
    # ]
    # optimizer = AdamW(param_groups, lr=args.lr)
    # Pass to Trainer: optimizers=(optimizer, None)

    optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type, opt_name=optimizer_name, rank=int(rank))
    if init_from == 'resume':
        optimizer.load_state_dict(checkpoint['optimizer'])
    checkpoint = None # free up memory


    # ── Trainer ────────────────────────────────────────────────────────────────
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized[eval_split],
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=build_compute_metrics(args.task),
        optimizers=(optimizer, None),  # uncomment if using custom optimizer above
    )

    # ── Train ──────────────────────────────────────────────────────────────────
    print("[*] Starting training...")
    train_result = trainer.train()

    # ── Save model & log final metrics ────────────────────────────────────────
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    metrics = trainer.evaluate()
    trainer.log_metrics("eval", metrics)
    trainer.save_metrics("eval", metrics)

    # Explicitly log final metrics to ClearML
    logger = task.get_logger()
    for key, val in metrics.items():
        logger.report_single_value(name=key, value=val)

    print(f"[✓] Training complete. Model saved to: {output_dir}")
    print(f"[✓] Final metrics: {metrics}")

    task.close()


if __name__ == "__main__":
    main()
