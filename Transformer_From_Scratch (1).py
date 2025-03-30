import torch
from torch import nn

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# ------------------------------
# Self-Attention Mechanism
# ------------------------------
class SelfAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super().__init__()
        assert embed_size % heads == 0, "Embedding size should be divisible by the number of heads."

        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads  

        self.values = nn.Linear(embed_size, embed_size, bias=False)
        self.keys = nn.Linear(embed_size, embed_size, bias=False)
        self.queries = nn.Linear(embed_size, embed_size, bias=False)
        self.fc_out = nn.Linear(embed_size, embed_size, bias=False)

    def forward(self, values, keys, query, mask=None):
        N = query.shape[0]  
        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]

        values = self.values(values)
        keys = self.keys(keys)
        queries = self.queries(query)

        values = values.view(N, value_len, self.heads, self.head_dim)
        queries = queries.view(N, query_len, self.heads, self.head_dim)
        keys = keys.view(N, key_len, self.heads, self.head_dim)

        # Compute attention scores
        attn_scores = torch.einsum("nqhd,nkhd->nhqk", queries, keys)

        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, float("-1e20"))

        attention = torch.softmax(attn_scores / (self.head_dim ** 0.5), dim=-1)

        # Weighted sum of values
        out = torch.einsum("nhql,nlhd->nqhd", attention, values)
        out = out.reshape(N, query_len, self.embed_size)
        out = self.fc_out(out)
        return out

# ------------------------------
# Transformer Block (Encoder & Decoder Base)
# ------------------------------
class TransformerBlock(nn.Module):
    def __init__(self, embed_size, heads, dropout, forward_expansion):
        super().__init__()
        self.attention = SelfAttention(embed_size, heads)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)

        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, forward_expansion * embed_size),
            nn.ReLU(),
            nn.Linear(forward_expansion * embed_size, embed_size)
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, value, key, query, mask):
        attention = self.attention(value, key, query, mask)
        X = self.dropout(self.norm1(attention + query))
        forward = self.feed_forward(X)
        out = self.dropout(self.norm2(forward + X))
        return out

# ------------------------------
# Encoder (Stacks Transformer Blocks)
# ------------------------------
class Encoder(nn.Module):
    def __init__(self, vocab_size, embed_size, num_layers, heads, dropout, forward_expansion, max_length):
        super().__init__()
        self.embed_size = embed_size
        self.word_embedding = nn.Embedding(vocab_size, embed_size)
        self.position_embedding = nn.Embedding(max_length, embed_size)

        self.layers = nn.ModuleList([
            TransformerBlock(embed_size, heads, dropout, forward_expansion) for _ in range(num_layers)
        ])
        
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        N, seq_length = x.shape
        positions = torch.arange(0, seq_length).expand(N, seq_length).to(device)
        out = self.dropout(self.word_embedding(x) + self.position_embedding(positions))

        for layer in self.layers:
            out = layer(out, out, out, mask)

        return out

# ------------------------------
# Decoder Block (Masked Self-Attention + Encoder-Decoder Attention)
# ------------------------------
class DecoderBlock(nn.Module):
    def __init__(self, embed_size, heads, dropout, forward_expansion):
        super().__init__()
        self.attention = SelfAttention(embed_size, heads)
        self.norm = nn.LayerNorm(embed_size)
        self.transformer_block = TransformerBlock(embed_size, heads, dropout, forward_expansion)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, value, key, src_mask, tgt_mask):
        attention = self.attention(x, x, x, tgt_mask)
        query = self.dropout(self.norm(attention + x))
        out = self.transformer_block(value, key, query, src_mask)
        return out

# ------------------------------
# Decoder (Stacks Decoder Blocks)
# ------------------------------
class Decoder(nn.Module):
    def __init__(self, vocab_size, embed_size, num_layers, heads, dropout, forward_expansion, max_length):
        super().__init__()
        self.word_embedding = nn.Embedding(vocab_size, embed_size)
        self.position_embedding = nn.Embedding(max_length, embed_size)

        self.layers = nn.ModuleList([
            DecoderBlock(embed_size, heads, dropout, forward_expansion) for _ in range(num_layers)
        ])

        self.fc_out = nn.Linear(embed_size, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_out, src_mask, tgt_mask):
        N, seq_length = x.shape
        positions = torch.arange(0, seq_length).expand(N, seq_length).to(device)
        x = self.dropout(self.word_embedding(x) + self.position_embedding(positions))

        for layer in self.layers:
            x = layer(x, enc_out, enc_out, src_mask, tgt_mask)

        return self.fc_out(x)

# ------------------------------
# Full Transformer Model
# ------------------------------
class Transformer(nn.Module):
    def __init__(self, src_vocab_size, trg_vocab_size, embed_size, num_layers, heads, dropout, forward_expansion, max_length):
        super().__init__()
        self.encoder = Encoder(src_vocab_size, embed_size, num_layers, heads, dropout, forward_expansion, max_length)
        self.decoder = Decoder(trg_vocab_size, embed_size, num_layers, heads, dropout, forward_expansion, max_length)

    def forward(self, src, trg, src_mask, tgt_mask):
        enc_out = self.encoder(src, src_mask)
        out = self.decoder(trg, enc_out, src_mask, tgt_mask)
        return out

# ------------------------------
# Test the Transformer Model
# ------------------------------
src_vocab_size = 1000
trg_vocab_size = 1000
embed_size = 512
num_layers = 6
heads = 8
dropout = 0.1
forward_expansion = 4
max_length = 100

# Create dummy input
src = torch.randint(0, src_vocab_size, (2, 10)).to(device)  # (batch_size=2, sequence_length=10)
trg = torch.randint(0, trg_vocab_size, (2, 10)).to(device)

# Masks (dummy ones for now, normally used for padding/masking future words)
src_mask = torch.ones(2, 1, 1, 10).to(device)  # No padding mask applied
tgt_mask = torch.ones(2, 1, 10, 10).to(device)  # No look-ahead mask applied

# Initialize model
model = Transformer(src_vocab_size, trg_vocab_size, embed_size, num_layers, heads, dropout, forward_expansion, max_length).to(device)

# Run a forward pass
out = model(src, trg, src_mask, tgt_mask)
print(f"Transformer Output Shape: {out.shape}")  # Expected: (batch_size, seq_length, trg_vocab_size)