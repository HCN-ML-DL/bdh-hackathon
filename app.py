import streamlit as st
import torch
from bdh_model import BDH
from transformers import GPT2LMHeadModel, GPT2Tokenizer

st.set_page_config(page_title="BDH Inference-Time Learning", layout="wide")

# Load models
@st.cache_resource
def load_bdh():
    checkpoint = torch.load("checkpoints/bdh_final.pt", map_location="cuda")
    config = checkpoint["config"]
    model = BDH(
        vocab_size=256,
        n_neurons=config["n_neurons"],
        d_internal=config["d_internal"],
        num_layers=config["num_layers"],
        num_heads=config["num_heads"],
    )
    model.load_state_dict(checkpoint["model"])
    model.eval().cuda()
    return model, config

@st.cache_resource
def load_gpt2():
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2").cuda()
    model.eval()
    tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer

with st.spinner("Loading models..."):
    bdh_model, bdh_config = load_bdh()
    gpt2_model, gpt2_tokenizer = load_gpt2()

if "bdh_states" not in st.session_state:
    st.session_state.bdh_states = None
if "bdh_position" not in st.session_state:
    st.session_state.bdh_position = 0
if "last_state_deltas" not in st.session_state:
    st.session_state.last_state_deltas = None
if "last_bdh_answer" not in st.session_state:
    st.session_state.last_bdh_answer = None
if "last_gpt2_answer" not in st.session_state:
    st.session_state.last_gpt2_answer = None
if "last_action" not in st.session_state:
    st.session_state.last_action = None

if st.session_state.bdh_states is not None and len(st.session_state.bdh_states) != bdh_model.num_layers:
    st.session_state.bdh_states = None
    st.session_state.bdh_position = 0

# Header
st.title("🧠 BDH Learns at Inference Time")
st.markdown("### Hebbian state updates during generation")
st.markdown("**Memory persists internally (no RAG)**")
st.markdown("---")

# Main demo
st.header("Memory Demo")

fact_input = st.text_input(
    "Fact input",
    "Bruno is a dog"
)
question_input = st.text_input(
    "Question input",
    "What is Bruno?"
)

col1, col2 = st.columns([1, 1])
with col1:
    max_tokens = st.slider("Generation length (characters)", 50, 800, 100)
with col2:
    temperature = st.slider("Temperature", 0.1, 1.5, 0.2)

memory_col, reset_col = st.columns([3, 1])
with reset_col:
    if st.button("Reset Memory"):
        st.session_state.bdh_states = None
        st.session_state.bdh_position = 0
        st.session_state.last_state_deltas = None
        st.session_state.last_bdh_answer = None
        st.session_state.last_gpt2_answer = None
        st.session_state.last_action = None
with memory_col:
    memory_status = "active" if st.session_state.bdh_states is not None else "empty"
    st.caption(f"BDH memory: {memory_status} | position: {st.session_state.bdh_position}")

teach_col, ask_col = st.columns(2)
with teach_col:
    teach_clicked = st.button("🧠 Teach", type="primary")
with ask_col:
    ask_clicked = st.button("❓ Ask")

if teach_clicked and fact_input.strip():
    with st.spinner("Updating BDH memory..."):
        tokens = torch.tensor([list(fact_input.encode())], device="cuda")
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
        st.session_state.last_gpt2_answer = None
        st.session_state.last_action = "teach"
    st.success("Fact stored in BDH memory.")

if ask_clicked and question_input.strip():
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🧠 BDH")
        with st.spinner("Answering with internal memory..."):
            tokens = torch.tensor([list(question_input.encode())], device="cuda")
            with torch.no_grad():
                output, state_deltas, updated_states, updated_position = bdh_model.generate(
                    tokens,
                    max_new_tokens=max_tokens,
                    temperature=temperature,
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
            bdh_answer = bdh_text[len(question_input):].strip() or bdh_text.strip()
            st.session_state.last_bdh_answer = bdh_answer
            st.session_state.last_action = "ask"

        st.write(bdh_answer)

    with col2:
        st.subheader("🤖 GPT-2")
        with st.spinner("Generating..."):
            inputs = gpt2_tokenizer(question_input, return_tensors="pt").to("cuda")
            with torch.no_grad():
                output = gpt2_model.generate(
                    inputs.input_ids,
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                    do_sample=True,
                    pad_token_id=gpt2_tokenizer.eos_token_id
                )
            gpt2_text = gpt2_tokenizer.decode(output[0], skip_special_tokens=True)
            gpt2_text = gpt2_text[:len(question_input) + max_tokens]
            gpt2_answer = gpt2_text[len(question_input):].strip() or gpt2_text.strip()
            st.session_state.last_gpt2_answer = gpt2_answer

        st.write(gpt2_answer)

        st.markdown("---")
        st.markdown("**📊 State Updates:**")
        st.markdown("❌ None — weights frozen")
        st.caption("Cannot learn without fine-tuning")

if st.session_state.last_state_deltas:
    st.markdown("---")
    st.markdown("**📊 Hebbian State Updates:**")
    total_delta = sum(st.session_state.last_state_deltas)
    for i, delta in enumerate(st.session_state.last_state_deltas):
        pct = (delta / total_delta * 100) if total_delta > 0 else 0
        st.markdown(f"Layer {i+1}: `{delta:.1f}` ({pct:.0f}%)")
    st.metric("Total State Change", f"{total_delta:.1f}")
    if st.session_state.last_action == "teach":
        st.caption("☝️ Fact encoded into BDH's internal state.")
    else:
        st.caption("☝️ Memory reused directly from BDH's internal state.")

# Explanation
st.markdown("---")
st.header("🔬 How It Works")

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    ### BDH Architecture
    ```python
    # During generation:
    output = query @ state
    state += key.T @ value  # HEBBIAN
    ```
    
    ✅ Linear attention (no softmax)  
    ✅ Fixed-size state matrix  
    ✅ State **updates** each token  
    ✅ "Fire together → wire together"
    """)

with col2:
    st.markdown("""
    ### Transformer Architecture
    ```python
    # During generation:
    output = softmax(Q @ K.T) @ V
    # weights FROZEN
    ```
    
    ❌ Softmax attention (quadratic)  
    ❌ KV-cache **grows** with context  
    ❌ Weights never change  
    ❌ Needs fine-tuning to learn
    """)

# Model details
st.markdown("---")
with st.expander("📊 Model Details"):
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"""
        **BDH:**
        - Neurons: {bdh_config['n_neurons']:,}
        - Internal dim: {bdh_config['d_internal']}
        - Layers: {bdh_config['num_layers']}
        - Parameters: ~37.8M
        """)
    with col2:
        st.markdown("""
        **GPT-2 Small:**
        - Vocabulary: 50,257
        - Embedding: 768
        - Layers: 12
        - Parameters: ~124M
        """)

st.markdown("---")
st.caption("Pathway/IIT Ropar Post-Transformer Hackathon | BDH Inference-Time Learning")
