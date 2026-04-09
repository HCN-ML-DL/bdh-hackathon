"""
BDH Training Script - Memory Optimized for L4
"""

import torch
from torch.utils.data import DataLoader, Dataset
from datasets import load_from_disk
from bdh_model import BDH
from tqdm import tqdm
import os

# ============================================================
# CONFIG - Reduced for L4 (23GB VRAM)
# ============================================================
CONFIG = {
    "n_neurons": 16384,      # Reduced from 32k to 16k
    "d_internal": 192,       # Reduced from 256 to 192
    "num_layers": 4,         # Reduced from 6 to 4
    "num_heads": 4,          # Reduced from 8 to 4
    "batch_size": 4,         # Reduced from 16 to 4
    "seq_len": 256,          # Reduced from 512 to 256
    "lr": 1e-3,
    "epochs": 3,
    "grad_accum": 8,         # Increased to compensate for smaller batch
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


def main():
    device = "cuda"
    torch.cuda.empty_cache()
    
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Load data
    print("\nLoading dataset...")
    dataset = load_from_disk("./tinystories_data")
    train_data = TextDataset(dataset, CONFIG["seq_len"])
    train_loader = DataLoader(train_data, batch_size=CONFIG["batch_size"], shuffle=True, num_workers=2, pin_memory=True)
    
    # Create model
    print("\nInitializing model...")
    model = BDH(
        vocab_size=256,
        n_neurons=CONFIG["n_neurons"],
        d_internal=CONFIG["d_internal"],
        num_layers=CONFIG["num_layers"],
        num_heads=CONFIG["num_heads"],
        dropout=0.0,  # No dropout for faster training
    ).to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG["lr"], weight_decay=0.1)
    
    os.makedirs("checkpoints", exist_ok=True)
    global_step = 0
    
    print("\n" + "="*60)
    print("TRAINING STARTED")
    print(f"Effective batch size: {CONFIG['batch_size'] * CONFIG['grad_accum']}")
    print("="*60)
    
    for epoch in range(CONFIG["epochs"]):
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
        
        for batch_idx, (x, y) in enumerate(pbar):
            x, y = x.to(device), y.to(device)
            
            # forward() now returns (logits, loss, states)
            logits, loss, _ = model(x, y)
            loss = loss / CONFIG["grad_accum"]
            loss.backward()
            
            if (batch_idx + 1) % CONFIG["grad_accum"] == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()
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
