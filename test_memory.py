"""
Interactive KV memory check for a trained BDH checkpoint (byte-level KEY=value / KEY? → value).
"""

import argparse

import torch
from bdh_model import BDH


def main():
    parser = argparse.ArgumentParser(description="Test BDH Hebbian recall on KV-style prompts.")
    parser.add_argument("--checkpoint", default="checkpoints/bdh_kv.pt", help="Path to saved .pt")
    parser.add_argument("--fact", default="X7=blue", help="Store string (e.g. X7=blue)")
    parser.add_argument("--question", default="X7?", help="Query string (e.g. X7?)")
    parser.add_argument("--max-new-tokens", type=int, default=24)
    parser.add_argument("--temperature", type=float, default=0.3)
    parser.add_argument("--top-k", type=int, default=10)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(args.checkpoint, map_location=device)
    cfg = ckpt["config"]
    model = BDH(
        vocab_size=256,
        n_neurons=cfg["n_neurons"],
        d_internal=cfg["d_internal"],
        num_layers=cfg["num_layers"],
        num_heads=cfg["num_heads"],
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    fact = torch.tensor([list(args.fact.encode("utf-8"))], device=device)
    question = torch.tensor([list(args.question.encode("utf-8"))], device=device)

    with torch.no_grad():
        model.diagnose_state(fact, question, device=device)

        _, states, pos = model.encode_to_state(fact)

        out_with, _, _, _ = model.generate(
            question,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            initial_states=states,
            initial_position=pos,
        )
        text_with = bytes(out_with[0].tolist()).decode("utf-8", errors="replace")

        out_without, _, _, _ = model.generate(
            question,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            initial_states=None,
            initial_position=0,
        )
        text_without = bytes(out_without[0].tolist()).decode("utf-8", errors="replace")

    print("---")
    print(f"Fact:     {args.fact!r}")
    print(f"Question: {args.question!r}")
    print(f"WITH state:    {text_with!r}")
    print(f"WITHOUT state: {text_without!r}")


if __name__ == "__main__":
    main()
