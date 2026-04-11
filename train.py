"""
BDH Training Script - Memory Optimized for L4
"""

import argparse
import json
import os
import subprocess
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
from datasets import load_from_disk
from bdh_model import BDH
from tqdm import tqdm

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision('medium')

# ============================================================
# CONFIG - Reduced for L4 (23GB VRAM)
# ============================================================
CONFIG = {
    "n_neurons": 16384,
    "d_internal": 192,
    "num_layers": 4,
    "num_heads": 4,

    # 🔥 TRAINING (OPTIMIZED)
    "batch_size": 16,        # increase (V100 can handle)
    "seq_len": 256,
    "lr": 3e-4,              # LOWER = more stable recall
    "epochs": 8,             # enough for convergence
    "grad_accum": 4,         # effective batch = 64
    "save_every": 200,       # more frequent checkpoints
}


class TextDataset(Dataset):
    def __init__(self, records, seq_len=256):
        self.data = []
        self.seq_len = seq_len
        print("Processing paired memory dataset...")

        grouped = {}
        for record in records:
            if not isinstance(record, dict):
                continue
            entity = record.get("entity")
            row_type = record.get("type")
            if not entity or row_type not in {"store", "recall"}:
                continue
            grouped.setdefault(str(entity), {"store": [], "recall": []})[row_type].append(record)

        for entity_rows in tqdm(grouped.values()):
            stores = entity_rows["store"]
            recalls = entity_rows["recall"]
            recalls_by_id = {
                self._base_example_id(row.get("example_id")): row
                for row in recalls
            }

            for index, store in enumerate(stores):
                recall = recalls_by_id.get(self._base_example_id(store.get("example_id")))
                if recall is None and recalls:
                    recall = recalls[index % len(recalls)]
                if recall is None:
                    continue

                fact_tokens = self._encode(store.get("text") or f"{store.get('input', '')}{store.get('target', '')}")
                question_tokens = self._encode(recall.get("input", ""))
                answer_tokens = self._encode(recall.get("target", ""))

                if not fact_tokens or not question_tokens or not answer_tokens:
                    continue
                if len(fact_tokens) > seq_len or len(question_tokens) + len(answer_tokens) > seq_len + 1:
                    continue

                self.data.append((
                    torch.tensor(fact_tokens, dtype=torch.long),
                    torch.tensor(question_tokens, dtype=torch.long),
                    torch.tensor(answer_tokens, dtype=torch.long),
                ))

        print(f"Created {len(self.data)} paired memory examples")

    @staticmethod
    def _base_example_id(example_id):
        if example_id is None:
            return None
        return str(example_id).removesuffix("-store").removesuffix("-recall")

    @staticmethod
    def _encode(text):
        return list(str(text).encode("utf-8", errors="ignore"))

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]


def memory_collate(batch):
    facts, questions, answers = zip(*batch)
    fact_lens = torch.tensor([len(tokens) for tokens in facts], dtype=torch.long)
    question_lens = torch.tensor([len(tokens) for tokens in questions], dtype=torch.long)
    answer_lens = torch.tensor([len(tokens) for tokens in answers], dtype=torch.long)
    return (
        pad_sequence(facts, batch_first=True, padding_value=0),
        pad_sequence(questions, batch_first=True, padding_value=0),
        pad_sequence(answers, batch_first=True, padding_value=0),
        fact_lens,
        question_lens,
        answer_lens,
    )


def length_mask(lengths, max_len, device):
    return torch.arange(max_len, device=device).unsqueeze(0) < lengths.to(device).unsqueeze(1)


def parse_args():
    parser = argparse.ArgumentParser(description="Train or fine-tune the BDH model.")
    parser.add_argument("--dataset-path", default="./tinystories_data", help="HF dataset directory or JSONL file with a text field.")
    parser.add_argument("--resume-path", default="checkpoints/bdh_final.pt")
    parser.add_argument("--output-path", default="checkpoints/bdh_final.pt")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--seq-len", type=int)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--grad-accum", type=int)
    parser.add_argument("--save-every", type=int)
    parser.add_argument("--gcs-checkpoint-dir", default=os.environ.get("GCS_CHECKPOINT_DIR"))
    return parser.parse_args()


def load_records(dataset_path):
    path = Path(dataset_path)
    if path.suffix == ".jsonl":
        records = []
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                if isinstance(row, dict) and (row.get("text") or (row.get("input") is not None and row.get("target") is not None)):
                    records.append(row)
        return records

    dataset = load_from_disk(str(path))
    return [ex.get("text", "") for ex in dataset]


def resolve_config(args):
    config = dict(CONFIG)
    overrides = {
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "seq_len": args.seq_len,
        "lr": args.lr,
        "grad_accum": args.grad_accum,
        "save_every": args.save_every,
    }
    for key, value in overrides.items():
        if value is not None:
            config[key] = value
    return config


def upload_checkpoint_to_gcs(checkpoint_path, gcs_checkpoint_dir):
    if not gcs_checkpoint_dir:
        return

    # Keep upload outside torch save logic so local checkpoints remain usable.
    print(f"Uploading {checkpoint_path} to {gcs_checkpoint_dir}...")
    subprocess.run(
        ["gcloud", "storage", "cp", str(checkpoint_path), gcs_checkpoint_dir],
        check=True,
    )


def main():
    args = parse_args()
    device = "cuda"
    torch.cuda.empty_cache()
    config = resolve_config(args)
    
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Load data
    print("\nLoading dataset...")
    records = load_records(args.dataset_path)
    train_data = TextDataset(records, config["seq_len"])
    train_loader = DataLoader(
        train_data,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        collate_fn=memory_collate,
    )

    checkpoint = None
    resume_path = args.resume_path
    if os.path.exists(resume_path):
        print(f"\nResuming from {resume_path}...")
        checkpoint = torch.load(resume_path, map_location=device)
        print(f"Resumed at step {checkpoint.get('step', 0)}")
    
    # Create model
    print("\nInitializing model...")
    model_config = dict(config)
    if checkpoint is not None:
        for key in ["n_neurons", "d_internal", "num_layers", "num_heads"]:
            model_config[key] = checkpoint["config"].get(key, model_config[key])
    model = BDH(
        vocab_size=256,
        n_neurons=model_config["n_neurons"],
        d_internal=model_config["d_internal"],
        num_layers=model_config["num_layers"],
        num_heads=model_config["num_heads"],
        dropout=0.0,
    ).to(device)
    if checkpoint is not None:
        model.load_state_dict(checkpoint["model"])
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=0.1, fused=True)
    
    os.makedirs("checkpoints", exist_ok=True)
    global_step = checkpoint.get("step", 0) if checkpoint is not None else 0
    
    print("\n" + "="*60)
    print("TRAINING STARTED")
    print(f"Dataset path: {args.dataset_path}")
    print(f"Effective batch size: {config['batch_size'] * config['grad_accum']}")
    print("="*60)
    
    for epoch in range(config["epochs"]):
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
        
        for batch_idx, (fact, question, answer, fact_len, question_len, answer_len) in enumerate(pbar):
            fact = fact.to(device)
            question = question.to(device)
            answer = answer.to(device)
            fact_len = fact_len.to(device)
            question_len = question_len.to(device)
            answer_len = answer_len.to(device)

            fact_mask = length_mask(fact_len, fact.size(1), device)
            question_mask = length_mask(question_len, question.size(1), device)
            answer_mask = length_mask(answer_len, answer.size(1), device)

            # Build Hebbian state from every fact in parallel; masks stop padding from writing memory.
            _, _, fact_states = model(
                fact,
                states=None,
                use_state=False,
                position_offset=0,
                token_mask=fact_mask,
            )
            if epoch == 0 and batch_idx == 0:
                print(f"fact_states[0].requires_grad: {fact_states[0].requires_grad}")

            # Process all questions against their matching fact-derived states.
            _, _, question_states = model(
                question,
                states=fact_states,
                use_state=True,
                position_offset=fact_len,
                token_mask=question_mask,
            )

            # Teacher-force all answers as one batch and mask padded targets out of the loss.
            last_question_index = (question_len - 1).clamp_min(0)
            last_question_token = question.gather(1, last_question_index.view(-1, 1))
            answer_inputs = torch.cat([last_question_token, answer[:, :-1]], dim=1)
            answer_targets = answer.masked_fill(~answer_mask, -1)
            _, loss, _ = model(
                answer_inputs,
                targets=answer_targets,
                states=question_states,
                use_state=True,
                position_offset=fact_len + question_len,
                token_mask=answer_mask,
            )
            raw_loss = loss.item()
            loss = loss / config["grad_accum"]
            loss.backward()
            
            if (batch_idx + 1) % config["grad_accum"] == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1
            
            pbar.set_postfix({"loss": f"{raw_loss:.3f}", "step": global_step})
            
            if global_step > 0 and global_step % config["save_every"] == 0:
                checkpoint_path = Path(f"checkpoints/bdh_step_{global_step}.pt")
                torch.save({"model": model.state_dict(), "config": model_config, "step": global_step}, checkpoint_path)
                upload_checkpoint_to_gcs(checkpoint_path, args.gcs_checkpoint_dir)
                
                # Quick generation test
                model.eval()
                with torch.no_grad():
                    # Test with EXACT training format
                    fact = torch.tensor([list(b"User: Tesla is a company.\nAssistant: Noted.")], device=device)
                    question = torch.tensor([list(b"User: What is Tesla?\nAssistant:")], device=device)

                    # Encode fact into state
                    _, states, pos = model.encode_to_state(fact)

                    # Test 1: WITH state (should output "Tesla is a company")
                    out_with, _, _, _ = model.generate(
                        question,
                        max_new_tokens=30,
                        temperature=0.3,
                        top_k=10,
                        initial_states=states,
                        initial_position=pos,
                    )
                    result_with = bytes(out_with[0].tolist()).decode('utf-8', errors='replace')

                    # Test 2: WITHOUT state (should output garbage or different answer)
                    out_without, _, _, _ = model.generate(
                        question,
                        max_new_tokens=30,
                        temperature=0.3,
                        top_k=10,
                        initial_states=None,
                        initial_position=0,
                    )
                    result_without = bytes(out_without[0].tolist()).decode('utf-8', errors='replace')

                    print(f"\n[Fact: User: Tesla is a company.]")
                    print(f"[WITH state:    {result_with}]")
                    print(f"[WITHOUT state: {result_without}]")
                model.train()
    
    torch.save({"model": model.state_dict(), "config": model_config, "step": global_step}, args.output_path)
    upload_checkpoint_to_gcs(args.output_path, args.gcs_checkpoint_dir)
    print(f"\nDone! Saved to {args.output_path}")


if __name__ == "__main__":
    main()
