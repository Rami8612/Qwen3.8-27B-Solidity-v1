#!/usr/bin/env python3
"""LoRA bf16 de Qwen3.8-27B con Unsloth sobre data/train.jsonl.

Prueba corta (medir memoria y velocidad):
    python train_unsloth.py --max-steps 30
Entrenamiento completo + exportación a GGUF Q4_K_M:
    python train_unsloth.py --export

Notas:
- Unsloth recomienda LoRA bf16 (no QLoRA) para la arquitectura Qwen3.5/3.8: ~56 GB de VRAM -> H100 80 GB.
- Dos modos en el mismo entrenamiento:
    * ejemplo SIN razonamiento -> enable_thinking=False -> "<think>\\n\\n</think>\\n\\n<informe>"
    * ejemplo CON razonamiento (campo reasoning_content, de 10_gen_reasoning.py) -> enable_thinking=True
      (el template añade "Reasoning effort is set to xhigh..." al system) -> "<think>\\n<razonamiento>\\n</think>\\n\\n<informe>"
- La loss se calcula sobre todo lo que escribe el asistente (razonamiento incluido), no sobre el prompt.
"""
import argparse, json, os, time
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")  # menos fragmentación de VRAM
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="unsloth/Qwen3.8-27B")
ap.add_argument("--data", default="data", help="carpeta con train.jsonl y val.jsonl")
ap.add_argument("--train-file", default=None, help="por defecto <data>/train.jsonl; p.ej. data/train_mixed.jsonl")
ap.add_argument("--out", default="out")
ap.add_argument("--max-seq", type=int, default=25600)  # razonamientos largos de DeepSeek; si OOM: --gc unsloth y luego 20480
ap.add_argument("--epochs", type=float, default=2)
ap.add_argument("--max-steps", type=int, default=-1, help=">0 para una prueba corta")
ap.add_argument("--lr", type=float, default=1e-4)
ap.add_argument("--r", type=int, default=16)
ap.add_argument("--grad-accum", type=int, default=8)
ap.add_argument("--max-eval-seq", type=int, default=8192, help="ejemplos de val más largos se omiten (OOM en eval)")
ap.add_argument("--eval-steps", type=int, default=100)
ap.add_argument("--longest-first", action="store_true",
                help="prueba de memoria: entrenar solo con los ejemplos más largos (usar con --max-steps)")
ap.add_argument("--gc", default="gpu", choices=["unsloth", "gpu"],
                help="gpu = checkpointing en GPU (medido: 16,5 s/paso, 62 GB en H100 NVL); "
                     "unsloth = con offload a RAM (21 s/paso; solo si falta VRAM)")
ap.add_argument("--export", action="store_true", help="fusionar LoRA y exportar GGUF q4_k_m al terminar")
a = ap.parse_args()

from unsloth import FastLanguageModel  # importar antes que transformers/trl
from unsloth.chat_templates import train_on_responses_only
from datasets import load_dataset
from trl import SFTConfig, SFTTrainer
import torch

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=a.model, max_seq_length=a.max_seq,
    load_in_4bit=False, load_in_16bit=True, full_finetuning=False,
)
tok = getattr(tokenizer, "tokenizer", tokenizer)  # por si devuelve un processor

model = FastLanguageModel.get_peft_model(
    model, r=a.r, lora_alpha=a.r, lora_dropout=0, bias="none",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    use_gradient_checkpointing="unsloth" if a.gc == "unsloth" else True,
    random_state=3407, max_seq_length=a.max_seq,
)

ds = load_dataset("json", data_files={"train": a.train_file or f"{a.data}/train.jsonl",
                                      "val": f"{a.data}/val.jsonl"})


def render(ex):
    msgs = [{k: v for k, v in m.items() if v is not None} for m in ex["messages"]]  # datasets rellena con None
    thinking = bool(msgs[-1].get("reasoning_content"))
    text = tok.apply_chat_template(msgs, tokenize=False, enable_thinking=thinking)
    return {"text": text, "thinking": thinking,
            "n_tokens": len(tok(text, add_special_tokens=False).input_ids)}


ds = ds.map(render, remove_columns=["messages"], num_proc=4)
before = {k: len(v) for k, v in ds.items()}
ds["train"] = ds["train"].filter(lambda e: e["n_tokens"] <= a.max_seq)
# la evaluación calcula logits completos (vocabulario de 248k): con secuencias largas no cabe en VRAM
ds["val"] = ds["val"].filter(lambda e: e["n_tokens"] <= a.max_eval_seq)
print(f"ejemplos (antes -> después de filtrar; train > {a.max_seq}, val > {a.max_eval_seq} tokens):",
      {k: f"{before[k]} -> {len(v)}" for k, v in ds.items()})
if a.longest_first:
    n = max(1, a.max_steps) * a.grad_accum
    ds["train"] = ds["train"].sort("n_tokens", reverse=True).select(range(min(n, len(ds["train"]))))
    print(f"--longest-first: {len(ds['train'])} ejemplos más largos, hasta {ds['train'][0]['n_tokens']} tokens")
print("tokens de train por época:", sum(ds["train"]["n_tokens"]),
      "| ejemplos con razonamiento:", sum(ds["train"]["thinking"]))
for ex in ds["train"].select(range(min(200, len(ds["train"])))):
    if ex["thinking"]:
        assert "Reasoning effort" in ex["text"] and "<think>\n\n</think>" not in ex["text"], "render con thinking incorrecto"
    else:
        assert "<think>\n\n</think>\n\n" in ex["text"] and "Reasoning effort" not in ex["text"], "render sin thinking incorrecto"

trainer = SFTTrainer(
    model=model, tokenizer=tokenizer,
    train_dataset=ds["train"], eval_dataset=ds["val"],
    args=SFTConfig(
        dataset_text_field="text", max_seq_length=a.max_seq, packing=False,
        per_device_train_batch_size=1, gradient_accumulation_steps=a.grad_accum,
        per_device_eval_batch_size=1, prediction_loss_only=True,
        num_train_epochs=a.epochs, max_steps=a.max_steps,
        learning_rate=a.lr, lr_scheduler_type="cosine", warmup_steps=20,
        weight_decay=0.01, optim="adamw_8bit", bf16=True,
        logging_steps=5, eval_strategy="steps", eval_steps=a.eval_steps,
        save_strategy="steps", save_steps=a.eval_steps, save_total_limit=3,
        output_dir=f"{a.out}/checkpoints", report_to="none", seed=3407,
    ),
)
# loss solo sobre lo que escribe el asistente (bloque think incluido: vacío o con razonamiento)
trainer = train_on_responses_only(
    trainer,
    instruction_part="<|im_start|>user\n",
    response_part="<|im_start|>assistant\n",
)

t0 = time.time()
stats = trainer.train()
elapsed = time.time() - t0
mem = torch.cuda.max_memory_reserved() / 2**30
summary = {"seconds": round(elapsed), "peak_vram_gb": round(mem, 1),
           "train_loss": stats.training_loss, "steps": stats.global_step}
print(json.dumps(summary, indent=2))
Path(a.out).mkdir(exist_ok=True)
(Path(a.out) / "train_summary.json").write_text(json.dumps(summary, indent=2))

model.save_pretrained(f"{a.out}/lora")
tokenizer.save_pretrained(f"{a.out}/lora")
if a.export:
    model.save_pretrained_merged(f"{a.out}/merged", tokenizer, save_method="merged_16bit")
    model.save_pretrained_gguf(f"{a.out}/gguf", tokenizer, quantization_method="q4_k_m")
    print("GGUF en", f"{a.out}/gguf")
