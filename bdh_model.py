"""
BDH Model for Hackathon Demo
Based on: https://github.com/pathwaycom/bdh
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_seq_len=8192, base=10000):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)
        self.max_seq_len = max_seq_len
        self._build_cache(max_seq_len)
    
    def _build_cache(self, seq_len):
        t = torch.arange(seq_len, dtype=torch.float32)
        freqs = torch.outer(t, self.inv_freq)
        self.register_buffer('cos_cached', freqs.cos(), persistent=False)
        self.register_buffer('sin_cached', freqs.sin(), persistent=False)
    
    def forward(self, x, position_offset=0):
        seq_len = x.shape[-2]
        positions = torch.arange(position_offset, position_offset + seq_len, device=x.device)
        freqs = torch.outer(positions.float(), self.inv_freq.to(x.device))
        cos = freqs.cos().unsqueeze(0).unsqueeze(0)
        sin = freqs.sin().unsqueeze(0).unsqueeze(0)
        x1, x2 = x[..., 0::2], x[..., 1::2]
        return torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1).flatten(-2)


class BDHLayer(nn.Module):
    def __init__(self, n_neurons, d_internal, num_heads=4, dropout=0.1):
        super().__init__()
        self.n_neurons = n_neurons
        self.d_internal = d_internal
        self.num_heads = num_heads
        self.head_dim = n_neurons // num_heads
        
        # Core parameters: encoder and two decoders
        self.encoder = nn.Parameter(torch.randn(n_neurons, d_internal) * 0.02)
        self.decoder_x = nn.Parameter(torch.randn(num_heads, d_internal, self.head_dim) * 0.02)
        self.decoder_y = nn.Parameter(torch.randn(num_heads, d_internal, self.head_dim) * 0.02)
        
        self.ln = nn.LayerNorm(d_internal, elementwise_affine=False, bias=False)
        self.rope = RotaryEmbedding(self.head_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, v_star, state=None, position_offset=0, use_state=False):
        B, T, D = v_star.shape
        device = v_star.device
        v_norm = self.ln(v_star)
        
        # Expand to neuron space with ReLU (creates sparsity!)
        v_expanded = v_norm.unsqueeze(1)
        x = F.relu(torch.matmul(v_expanded, self.decoder_x))  # [B, H, T, head_dim]
        
        # Apply RoPE
        x_rot = self.rope(x, position_offset)
        
        # Initialize state if needed
        if state is None:
            state = torch.zeros(B, self.num_heads, self.head_dim, D, device=device)
        
        if use_state:
            # RECURRENT MODE: Hebbian inference-time learning!
            # Query accumulated state
            a_star = torch.matmul(x_rot, state)  # [B, H, T, D]
            
            # HEBBIAN UPDATE: state += x.T @ v
            v_for_update = v_norm.unsqueeze(1).expand(-1, self.num_heads, -1, -1)
            state_update = torch.matmul(x_rot.transpose(-2, -1), v_for_update)
            new_state = state + state_update
        else:
            # PARALLEL MODE: Full attention (for training)
            scores = torch.matmul(x_rot, x_rot.transpose(-2, -1))
            causal_mask = torch.tril(torch.ones(T, T, device=device), diagonal=-1)
            scores = scores * causal_mask.unsqueeze(0).unsqueeze(0)
            
            v_attn = v_norm.unsqueeze(1).expand(-1, self.num_heads, -1, -1)
            a_star = torch.matmul(scores, v_attn)
            
            # Compute state for transitioning to recurrent mode
            new_state = torch.matmul(x_rot.transpose(-2, -1), v_attn)
        
        # Second ReLU + gating (more sparsity!)
        a_norm = self.ln(a_star)
        y_pre = torch.matmul(a_norm, self.decoder_y)
        y = F.relu(y_pre) * x  # GATING: this creates ~5% sparsity
        
        y = y.transpose(1, 2).reshape(B, T, self.n_neurons)
        y = self.dropout(y)
        
        # Compress back to d_internal
        v_delta = torch.matmul(y, self.encoder)
        return v_star + self.ln(v_delta), new_state


class BDH(nn.Module):
    def __init__(self, vocab_size=256, n_neurons=16384, d_internal=256, num_layers=6, num_heads=4, dropout=0.1):
        super().__init__()
        self.vocab_size = vocab_size
        self.n_neurons = n_neurons
        self.d_internal = d_internal
        self.num_layers = num_layers
        self.num_heads = num_heads
        
        self.embed = nn.Embedding(vocab_size, d_internal)
        self.layers = nn.ModuleList([
            BDHLayer(n_neurons, d_internal, num_heads, dropout)
            for _ in range(num_layers)
        ])
        self.final_ln = nn.LayerNorm(d_internal, elementwise_affine=False, bias=False)
        self.lm_head = nn.Linear(d_internal, vocab_size, bias=False)
        self.lm_head.weight = self.embed.weight  # Weight tying
        
        self.apply(self._init_weights)
        n_params = sum(p.numel() for p in self.parameters())
        print(f"BDH: {n_params/1e6:.1f}M params | n={n_neurons}, d={d_internal}, layers={num_layers}")
        
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.normal_(m.weight, std=0.02)
        elif isinstance(m, nn.Embedding):
            torch.nn.init.normal_(m.weight, std=0.02)
    
    def forward(self, idx, targets=None, states=None, use_state=False, position_offset=0):
        x = self.embed(idx)
        
        if states is None:
            states = [None] * self.num_layers
        
        new_states = []
        for i, layer in enumerate(self.layers):
            x, new_state = layer(x, states[i], position_offset, use_state)
            new_states.append(new_state)
        
        x = self.final_ln(x)
        logits = self.lm_head(x)
        
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, self.vocab_size), targets.view(-1), ignore_index=-1)
        return logits, loss, new_states
    
    def generate(self, idx, max_new_tokens=100, temperature=0.8, top_k=50, use_hebbian=True):
        """Generate with optional Hebbian state updates."""
        B, T = idx.shape
        
        if use_hebbian:
            # Process prompt to build initial state
            _, _, states = self.forward(idx, states=None, use_state=False, position_offset=0)
            states_after_prompt = [s.clone() for s in states]
            position = T
            
            # Generate token by token with Hebbian updates
            for _ in range(max_new_tokens):
                last_token = idx[:, -1:]
                logits, _, states = self.forward(
                    last_token, 
                    states=states, 
                    use_state=True,
                    position_offset=position
                )
                position += 1
                
                logits = logits[:, -1, :] / temperature
                if top_k:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = -float('Inf')
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                idx = torch.cat([idx, next_token], dim=1)
            
            # Compute state deltas
            state_deltas = []
            for s_before, s_after in zip(states_after_prompt, states):
                delta = (s_after - s_before).norm().item()
                state_deltas.append(delta)
            
            return idx, state_deltas
        else:
            # Standard generation (no state)
            for _ in range(max_new_tokens):
                logits, _, _ = self.forward(idx[:, -2048:])
                logits = logits[:, -1, :] / temperature
                if top_k:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = -float('Inf')
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                idx = torch.cat([idx, next_token], dim=1)
            return idx, None
