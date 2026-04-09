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

# Header
st.title("🧠 BDH Learns at Inference Time")
st.markdown("### Hebbian state updates during generation")
st.markdown("---")

# Main demo
st.header("Story Continuation")

prompt = st.text_area(
    "Enter a story prompt:",
    "Once upon a time, there was a little dragon named Spark. Spark loved to",
    height=100
)

col1, col2 = st.columns([1, 1])
with col1:
    max_tokens = st.slider("Generation length", 50, 200, 100)
with col2:
    temperature = st.slider("Temperature", 0.5, 1.5, 0.8)

if st.button("🚀 Generate", type="primary"):
    
    col1, col2 = st.columns(2)
    
    # BDH Generation
    with col1:
        st.subheader("🧠 BDH")
        with st.spinner("Generating with Hebbian updates..."):
            tokens = torch.tensor([list(prompt.encode())], device="cuda")
            with torch.no_grad():
                output, state_deltas = bdh_model.generate(
                    tokens, 
                    max_new_tokens=max_tokens, 
                    temperature=temperature,
                    use_hebbian=True
                )
            bdh_text = bytes(output[0].cpu().tolist()).decode("utf-8", errors="replace")
        
        st.write(bdh_text)
        
        # Show state changes
        if state_deltas:
            st.markdown("---")
            st.markdown("**📊 Hebbian State Updates:**")
            total_delta = sum(state_deltas)
            for i, delta in enumerate(state_deltas):
                pct = (delta / total_delta * 100) if total_delta > 0 else 0
                st.markdown(f"Layer {i+1}: `{delta:.1f}` ({pct:.0f}%)")
            st.metric("Total State Change", f"{total_delta:.1f}")
            st.caption("☝️ Learning at inference time!")
    
    # GPT-2 Generation
    with col2:
        st.subheader("🤖 GPT-2")
        with st.spinner("Generating..."):
            inputs = gpt2_tokenizer(prompt, return_tensors="pt").to("cuda")
            with torch.no_grad():
                output = gpt2_model.generate(
                    inputs.input_ids,
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                    do_sample=True,
                    pad_token_id=gpt2_tokenizer.eos_token_id
                )
            gpt2_text = gpt2_tokenizer.decode(output[0], skip_special_tokens=True)
        
        st.write(gpt2_text)
        
        st.markdown("---")
        st.markdown("**📊 State Updates:**")
        st.markdown("❌ None — weights frozen")
        st.caption("Cannot learn without fine-tuning")

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
