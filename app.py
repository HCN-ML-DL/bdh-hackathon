import os

import streamlit as st
import torch
from bdh_model import BDH
from transformers import GPT2LMHeadModel, GPT2Tokenizer

st.set_page_config(page_title="BDH Inference-Time Learning", layout="wide")

# Match train.py / test_memory.py defaults (KV JSONL: KEY=value, query KEY?, answer value).
_DEFAULT_CKPT = "checkpoints/bdh_kv.pt"
_FALLBACK_CKPT = "checkpoints/bdh_final.pt"


def resolve_checkpoint_path() -> str:
    env = (os.environ.get("BDH_CHECKPOINT") or "").strip()
    if env and os.path.isfile(env):
        return env
    if os.path.isfile(_DEFAULT_CKPT):
        return _DEFAULT_CKPT
    if os.path.isfile(_FALLBACK_CKPT):
        return _FALLBACK_CKPT
    return env or _DEFAULT_CKPT


@st.cache_resource
def load_bdh(checkpoint_path: str):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint["config"]
    model = BDH(
        vocab_size=256,
        n_neurons=config["n_neurons"],
        d_internal=config["d_internal"],
        num_layers=config["num_layers"],
        num_heads=config["num_heads"],
    )
    model.load_state_dict(checkpoint["model"])
    model.eval().to(device)
    return model, config, device, checkpoint_path


@st.cache_resource
def load_gpt2():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2").to(device)
    model.eval()
    tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer, device


ckpt_path = resolve_checkpoint_path()
if not os.path.isfile(ckpt_path):
    st.error(
        f"No checkpoint found. Train with `train.py` (writes `{_DEFAULT_CKPT}`) or set "
        f"`BDH_CHECKPOINT` to a `.pt` file. Tried: {ckpt_path!r}."
    )
    st.stop()

with st.spinner("Loading models..."):
    bdh_model, bdh_config, torch_device, loaded_ckpt = load_bdh(ckpt_path)
    gpt2_model, gpt2_tokenizer, gpt2_device = load_gpt2()

bdh_param_m = sum(p.numel() for p in bdh_model.parameters()) / 1e6

if "bdh_states" not in st.session_state:
    st.session_state.bdh_states = None
if "bdh_position" not in st.session_state:
    st.session_state.bdh_position = 0
if "last_state_deltas" not in st.session_state:
    st.session_state.last_state_deltas = None
if "last_bdh_answer" not in st.session_state:
    st.session_state.last_bdh_answer = None
if "last_bdh_cold_answer" not in st.session_state:
    st.session_state.last_bdh_cold_answer = None
if "last_gpt2_answer" not in st.session_state:
    st.session_state.last_gpt2_answer = None
if "last_action" not in st.session_state:
    st.session_state.last_action = None

if st.session_state.bdh_states is not None and len(st.session_state.bdh_states) != bdh_model.num_layers:
    st.session_state.bdh_states = None
    st.session_state.bdh_position = 0

# Header
st.title("BDH learns at inference time")
st.markdown(
    """
**Frozen weights, updating state.** After you **Teach**, the model folds the fact into a fixed-size
Hebbian state (same idea as the `store` phase in `train.py`). **Ask** runs generation **on top of that
state**—no retrieval index, no gradient step—so the network “remembers” only through state dynamics.
"""
)
st.caption(f"Checkpoint: `{loaded_ckpt}` · device: `{torch_device}`")
st.markdown("---")

# Main demo
st.header("Key–value memory demo")
st.caption(
    "Training data look like `KEY=value` then `KEY?` → `value` (see `memory_data/final/kv_train.jsonl`). "
    "Use the same style here for best results."
)

fact_input = st.text_input("Fact (store)", value="X7=blue", help="Example: X7=blue — encodes one association.")
question_input = st.text_input("Question (recall)", value="X7?", help="Example: X7? — should complete with the value.")

col1, col2, col3 = st.columns([1, 1, 1])
with col1:
    max_tokens = st.slider("Max new tokens (characters)", 8, 800, 64)
with col2:
    temperature = st.slider("Temperature", 0.1, 1.5, 0.3)
with col3:
    top_k = st.slider("Top-k sampling", 1, 50, 10)

memory_col, reset_col = st.columns([3, 1])
with reset_col:
    if st.button("Reset memory"):
        st.session_state.bdh_states = None
        st.session_state.bdh_position = 0
        st.session_state.last_state_deltas = None
        st.session_state.last_bdh_answer = None
        st.session_state.last_bdh_cold_answer = None
        st.session_state.last_gpt2_answer = None
        st.session_state.last_action = None
with memory_col:
    memory_status = "active" if st.session_state.bdh_states is not None else "empty"
    st.caption(f"Hebbian state: {memory_status} · position: {st.session_state.bdh_position}")

teach_col, ask_col = st.columns(2)
with teach_col:
    teach_clicked = st.button("Teach (encode fact)", type="primary")
with ask_col:
    ask_clicked = st.button("Ask (generate with state)")

if teach_clicked and fact_input.strip():
    with st.spinner("Folding fact into Hebbian state..."):
        tokens = torch.tensor([list(fact_input.encode("utf-8"))], device=torch_device)
        with torch.no_grad():
            state_deltas, updated_states, updated_position = bdh_model.encode_to_state(
                tokens,
                initial_states=st.session_state.bdh_states,
                initial_position=st.session_state.bdh_position,
            )
        st.session_state.bdh_states = [
            state.detach().clone() if state is not None else None
            for state in (updated_states or [])
        ] or None
        st.session_state.bdh_position = updated_position
        st.session_state.last_state_deltas = state_deltas
        st.session_state.last_bdh_answer = None
        st.session_state.last_bdh_cold_answer = None
        st.session_state.last_gpt2_answer = None
        st.session_state.last_action = "teach"
    st.success("Fact encoded. Internal state updated; weights unchanged.")

if ask_clicked and question_input.strip():
    col_a, col_b, col_c = st.columns(3)

    with col_a:
        st.subheader("BDH · with memory")
        with st.spinner("Generating with taught state..."):
            tokens = torch.tensor([list(question_input.encode("utf-8"))], device=torch_device)
            with torch.no_grad():
                output, state_deltas, updated_states, updated_position = bdh_model.generate(
                    tokens,
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                    top_k=top_k,
                    use_hebbian=True,
                    initial_states=st.session_state.bdh_states,
                    initial_position=st.session_state.bdh_position,
                )
            st.session_state.bdh_states = [
                state.detach().clone() if state is not None else None
                for state in (updated_states or [])
            ] or None
            st.session_state.bdh_position = updated_position
            st.session_state.last_state_deltas = state_deltas

            bdh_text = bytes(output[0].cpu().tolist()).decode("utf-8", errors="replace")
            bdh_answer = bdh_text[len(question_input) :].strip() or bdh_text.strip()
            st.session_state.last_bdh_answer = bdh_answer
            st.session_state.last_action = "ask"

        st.write(bdh_answer if bdh_answer else "_(empty)_")

    with col_b:
        st.subheader("BDH · no memory")
        st.caption("Same question, cold state (matches `test_memory.py` baseline).")
        with st.spinner("Generating without prior state..."):
            tokens = torch.tensor([list(question_input.encode("utf-8"))], device=torch_device)
            with torch.no_grad():
                out_cold, _, _, _ = bdh_model.generate(
                    tokens,
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                    top_k=top_k,
                    use_hebbian=True,
                    initial_states=None,
                    initial_position=0,
                )
            cold_text = bytes(out_cold[0].cpu().tolist()).decode("utf-8", errors="replace")
            cold_answer = cold_text[len(question_input) :].strip() or cold_text.strip()
            st.session_state.last_bdh_cold_answer = cold_answer
        st.write(cold_answer if cold_answer else "_(empty)_")

    with col_c:
        st.subheader("GPT-2 · baseline")
        with st.spinner("Generating..."):
            inputs = gpt2_tokenizer(question_input, return_tensors="pt").to(gpt2_device)
            with torch.no_grad():
                output = gpt2_model.generate(
                    inputs.input_ids,
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                    do_sample=True,
                    pad_token_id=gpt2_tokenizer.eos_token_id,
                )
            gpt2_text = gpt2_tokenizer.decode(output[0], skip_special_tokens=True)
            gpt2_text = gpt2_text[: len(question_input) + max_tokens]
            gpt2_answer = gpt2_text[len(question_input) :].strip() or gpt2_text.strip()
            st.session_state.last_gpt2_answer = gpt2_answer

        st.write(gpt2_answer)

        st.markdown("---")
        st.markdown("**Weight updates**")
        st.markdown("None — frozen transformer weights.")

if st.session_state.last_state_deltas:
    st.markdown("---")
    st.markdown("**Hebbian state change (last Teach or Ask)**")
    total_delta = sum(st.session_state.last_state_deltas)
    for i, delta in enumerate(st.session_state.last_state_deltas):
        pct = (delta / total_delta * 100) if total_delta > 0 else 0
        st.markdown(f"Layer {i + 1}: `{delta:.1f}` ({pct:.0f}%)")
    st.metric("Total L2 delta across layers", f"{total_delta:.1f}")
    if st.session_state.last_action == "teach":
        st.caption("Fact pass: state accumulated from the store string (parallel encoding when memory was empty).")
    else:
        st.caption("Ask pass: state carried in and updated token-wise during generation.")

# Explanation
st.markdown("---")
st.header("How this maps to training")

st.markdown(
    """
| Phase | `train.py` | This app |
|-------|------------|----------|
| Store | Forward on **fact** tokens → build layer states | **Teach** → `encode_to_state` |
| Recall | Forward on **question** with those states, then teacher-forced **answer** | **Ask** → `generate` from question with `initial_states` |
| Learning | Supervised CE on byte targets; Hebbian state still updates during forward | No backward pass at demo time; only inference-time state |
"""
)

col1, col2 = st.columns(2)

with col1:
    st.markdown(
        """
### BDH (Hebbian recurrence)

When `use_state=True`, each layer maintains a fixed-size state matrix. Each token can **read** from
and **write** to that state (Hebbian-style), so the same weights implement both “memory” and prediction.

```text
read:  output ∝ f(state, token)
write: state ← state + Δ(state, token)   # inference-time only; no ∂L/∂W here at demo
```
"""
    )

with col2:
    st.markdown(
        """
### GPT-2

Causal attention with a growing KV cache. Context is stored in activations, not in a separate
learned Hebbian state. Weights do not change at inference.

```text
attention = softmax(QKᵀ) V   # frozen W_Q, W_K, W_V
```
"""
    )

# Model details
st.markdown("---")
with st.expander("Model details"):
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(
            f"""
**BDH** (`{loaded_ckpt}`)

- Neurons: {bdh_config["n_neurons"]:,}
- Internal dim: {bdh_config["d_internal"]}
- Layers: {bdh_config["num_layers"]}
- Heads: {bdh_config["num_heads"]}
- Parameters: ~{bdh_param_m:.1f}M
"""
        )
    with col2:
        gpt2_m = sum(p.numel() for p in gpt2_model.parameters()) / 1e6
        st.markdown(
            f"""
**GPT-2 (small)**

- Vocabulary: 50,257
- Embedding: 768
- Layers: 12
- Parameters: ~{gpt2_m:.0f}M
"""
        )

st.markdown("---")
st.caption("Pathway / IIT Ropar Post-Transformer Hackathon · BDH inference-time learning demo")
