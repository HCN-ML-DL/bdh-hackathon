import torch
from bdh_model import BDH

# Load
ckpt = torch.load("checkpoints/bdh_memory_v1.pt", map_location="cuda")
model = BDH(
    vocab_size=256,
    n_neurons=ckpt["config"]["n_neurons"],
    d_internal=ckpt["config"]["d_internal"],
    num_layers=ckpt["config"]["num_layers"],
    num_heads=ckpt["config"]["num_heads"],
).cuda().eval()
model.load_state_dict(ckpt["model"])

# Test the CORRECT way
with torch.no_grad():
    fact = torch.tensor([list(b"Bruno is a dog")], device="cuda")
    question = torch.tensor([list(b"What is Bruno?")], device="cuda")
    model.diagnose_state(fact, question)

    # Encode fact
    _, states, pos = model.encode_to_state(fact)

    # Generate with state
    out, _, _, _ = model.generate(
        question,
        max_new_tokens=30,
        temperature=0.3,
        top_k=10,
        initial_states=states,
        initial_position=pos,
    )
    print(bytes(out[0].tolist()).decode('utf-8', errors='replace'))
