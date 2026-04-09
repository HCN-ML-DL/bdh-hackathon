"""
BDH Training Script - FAST Mode for L4 (23GB VRAM)
"""

import torch
from torch.utils.data import DataLoader, Dataset
from datasets import load_from_disk
from bdh_model import BDH
from tqdm import tqdm
import os
import glob

# SPEED OPTIMIZATIONS
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision('medium')

# ============================================================
# CONFIG - Optimized for L4 speed
# ============================================================
CONFIG = {
    "n_neurons": 16384,
    "d_internal": 192,
    "num_layers": 4,
    "num_heads": 4,
    "batch_size": 32,
    "seq_len": 256,
    "lr": 3e-4,          # Lower LR for continued training
    "epochs": 3,
    "grad_accum": 2,
    "save_every": 500,
}

class TextDataset(Dataset):
    def __init__(self, hf_dataset, seq_len=256):
        self.data = []
        print("Processing dataset...")
        for ex in tqdm(hf_dataset):
            text = ex.get("text", "")
            if len(text) > seq_len:
                tokens = list(text.encode("utf-8", errors="ignore"))[:seq_len + 1]
                if len(tokens) == seq_len + 1:
                    self.data.append(tokens)
        print(f"Created {len(self.data)} training sequences")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        tokens = self.data[idx]
        x = torch.tensor(tokens[:-1], dtype=torch.long)
        y = torch.tensor(tokens[1:], dtype=torch.long)
        return x, y


def find_latest_checkpoint():
    """Find checkpoint with highest step number."""
    checkpoints = glob.glob("checkpoints/bdh_step_*.pt") + glob.glob("checkpoints/bdh_final.pt")
    
    if not checkpoints:
        return None, 0
    
    def get_step(path):
        if "final" in path:
            try:
                ckpt = torch.load(path, map_location="cpu")
                return ckpt.get("step", 0)
            except:
                return 0
        try:
            return int(path.split("_")[-1].replace(".pt", ""))
        except:
            return 0
    
    latest = max(checkpoints, key=get_step)
    return latest, get_step(latest)


def main():
    device = "cuda"
    torch.cuda.empty_cache()
    
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print("TF32: ON | cuDNN benchmark: ON | Deterministic: OFF")
    
    # Load data
    print("\nLoading dataset...")
    dataset = load_from_disk("./tinystories_data")
    train_data = TextDataset(dataset, CONFIG["seq_len"])
    train_loader = DataLoader(
        train_data, 
        batch_size=CONFIG["batch_size"], 
        shuffle=True, 
        num_workers=4,
        pin_memory=True,
        persistent_workers=True
    )
    
    # Create model
    print("\nInitializing model...")
    model = BDH(
        vocab_size=256,
        n_neurons=CONFIG["n_neurons"],
        d_internal=CONFIG["d_internal"],
        num_layers=CONFIG["num_layers"],
        num_heads=CONFIG["num_heads"],
        dropout=0.0,
    ).to(device)
    
    os.makedirs("checkpoints", exist_ok=True)
    global_step = 0
    
    # Resume from LATEST checkpoint
    resume_path, resume_step = find_latest_checkpoint()
    if resume_path:
        print(f"\nResuming from {resume_path} (step {resume_step})...")
        ckpt = torch.load(resume_path, map_location=device)
        model.load_state_dict(ckpt["model"])
        global_step = resume_step
    
    # Use lower LR if resuming
    lr = CONFIG["lr"] if global_step == 0 else CONFIG["lr"] / 3
    print(f"Using LR: {lr}")
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.1, fused=True)
    scaler = torch.amp.GradScaler('cuda')
    
    print("\n" + "="*60)
    print("TRAINING STARTED")
    print(f"Effective batch size: {CONFIG['batch_size'] * CONFIG['grad_accum']}")
    print(f"Starting from step: {global_step}")
    print("="*60)
    
    for epoch in range(CONFIG["epochs"]):
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
        
        for batch_idx, (x, y) in enumerate(pbar):
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            
            with torch.amp.autocast('cuda'):
                logits, loss, _ = model(x, y)
                loss = loss / CONFIG["grad_accum"]
            
            # NaN check
            if torch.isnan(loss):
                print("\n⚠️ NaN detected! Skipping batch...")
                optimizer.zero_grad(set_to_none=True)
                continue
            
            scaler.scale(loss).backward()
            
            if (batch_idx + 1) % CONFIG["grad_accum"] == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)  # Tighter clipping
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
            
            pbar.set_postfix({"loss": f"{loss.item() * CONFIG['grad_accum']:.3f}", "step": global_step})
            
            if global_step > 0 and global_step % CONFIG["save_every"] == 0:
                torch.save({"model": model.state_dict(), "config": CONFIG, "step": global_step}, f"checkpoints/bdh_step_{global_step}.pt")
                
                # Quick generation test
                model.eval()
                with torch.no_grad():
                    prompt = torch.tensor([list(b"Once upon a time")], device=device)
                    out, _ = model.generate(prompt, max_new_tokens=50, temperature=0.8)
                    print(f"\nSample: {bytes(out[0].tolist()).decode('utf-8', errors='replace')}")
                model.train()
    
    torch.save({"model": model.state_dict(), "config": CONFIG, "step": global_step}, "checkpoints/bdh_final.pt")
    print("\nDone! Saved to checkpoints/bdh_final.pt")


if __name__ == "__main__":
    main()
