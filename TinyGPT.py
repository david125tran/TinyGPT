# ------------------------------- Imports --------------------------------
import argparse
import sys
import requests
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Callable, Tuple

# Prevent main from executing multiple times
if "__TINY_GPT_MAIN_HAS_RUN__" in globals():
    raise SystemExit("[Guard] Script already executed once — exiting duplicate run.")
__TINY_GPT_MAIN_HAS_RUN__ = True


# ------------------------------- Functions ----------------------------------
def set_seed(seed: int = 1337) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def banner(text: str) -> None:
    """
    Create a banner for easier visualiziation of what's going on 

    Ex.
    Input:  "Device"
    Output:
            *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
            *                          Device                           *
            *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
    """
    banner_len = len(text)
    mid = 29 - banner_len // 2

    print("\n\n\n")
    print("*" + "-*" * 30)
    if (banner_len % 2 != 0):
        print("*"  + " " * mid + text + " " * mid + "*")
    else:
        print("*"  + " " * mid + text + " " + " " * mid + "*")
    print("*" + "-*" * 30 + "\n")


# ------------------------------- Data -----------------------------------
def load_tiny_shakespeare() -> str:
    """
    Downloads tiny Shakespeare into memory. Super small, so fine for a quick demo.
    """
    url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    return r.text


def build_char_tokenizer(text: str):
    """
    Returns (chars, encode, decode).
    chars: sorted list of unique characters in corpus
    encode(s): str -> list[int]
    decode(ids): list[int] -> str
    """
    chars = sorted(set(text))
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}
    def encode(s: str): 
        return [stoi[c] for c in s]
    def decode(ids): 
        return "".join(itos[i] for i in ids)
    return chars, encode, decode


def make_splits(encoded: torch.Tensor, train_frac: float = 0.9) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Splits a 1D tensor of token ids into train/val.
    """
    n = int(train_frac * len(encoded))
    return encoded[:n], encoded[n:]


def make_batcher(
        train_data: torch.Tensor,
        val_data: torch.Tensor,
        block_size: int,
        batch_size: int,
        device: torch.device,
    ) -> Callable[[str], Tuple[torch.Tensor, torch.Tensor]]:
    """
    Returns a function get_batch(split) -> (x, y) on the chosen device.
    """

    def get_batch(split: str):
        data = train_data if split == "train" else val_data
        ix = torch.randint(len(data) - block_size, (batch_size,))
        x = torch.stack([data[i:i + block_size] for i in ix]).to(device)
        y = torch.stack([data[i + 1:i + block_size + 1] for i in ix]).to(device)
        return x, y
    
    return get_batch


# --------------------------- Bigram Language Model -----------------------
class BigramLanguageModel(nn.Module):
    """
    Baseline: next token depends only on the current token.
    That's literally an embedding table of (vocab_size -> logits for next token).
    Very limited, but it proves the training loop and token plumbing work.
    """
    def __init__(self, vocab_size: int):
        super().__init__()
        self.table = nn.Embedding(vocab_size, vocab_size)


    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None):
        # idx: (B, T) ints -> logits: (B, T, vocab)
        logits = self.table(idx)
        loss = None
        if targets is not None:
            B, T, C = logits.shape
            loss = F.cross_entropy(logits.view(B * T, C), targets.view(B * T))
        return logits, loss


    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int) -> torch.Tensor:
        # Greedy-ish sampling by multinomial (random, not argmax).
        for _ in range(max_new_tokens):
            logits, _ = self(idx)
            probs = F.softmax(logits[:, -1, :], dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_id], dim=1)

        return idx


def run_bigram_demo(vocab_size: int, get_batch, decode, device, steps: int = 100) -> None:
    banner("Baseline: Bigram LM")
    model = BigramLanguageModel(vocab_size).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)

    # quick sanity run
    for _ in range(steps):
        xb, yb = get_batch("train")
        _, loss = model(xb, yb)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    print(f"[Bigram] last training loss: {loss.item():.4f}")

    # sample a bit so we see it's "alive" (it won't look like Shakespeare)
    start = torch.zeros((1, 1), dtype=torch.long, device=device)
    sample = model.generate(start, 200)[0].tolist()
    print("[Bigram] sample:")
    print(decode(sample))


# ------------------------- Transformer Language Model --------------------
class Head(nn.Module):
    """
    One self-attention head, masked (causal).
    """
    def __init__(self, n_embd: int, head_size: int, block_size: int, dropout: float):
        super().__init__()
        self.key   = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)
        self.scale = head_size ** -0.5


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        v = self.value(x)

        wei = (q @ k.transpose(-2, -1)) * self.scale              # (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)

        out = wei @ v                                             # (B, T, head_size)
        return out


class MultiHeadAttention(nn.Module):
    def __init__(self, n_embd: int, n_head: int, block_size: int, dropout: float):
        super().__init__()
        head_size = n_embd // n_head
        self.heads = nn.ModuleList([
            Head(n_embd, head_size, block_size, dropout) for _ in range(n_head)
        ])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.proj(out)
        return self.dropout(out)


class FeedForward(nn.Module):
    """
    Simple MLP: expand -> ReLU -> project back. Dropout to be safe.
    """
    def __init__(self, n_embd: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class Block(nn.Module):
    """
    Transformer decoder block: (LN -> MHA -> residual) then (LN -> MLP -> residual)
    """
    def __init__(self, n_embd: int, n_head: int, block_size: int, dropout: float):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.sa  = MultiHeadAttention(n_embd, n_head, block_size, dropout)
        self.ln2 = nn.LayerNorm(n_embd)
        self.ffn = FeedForward(n_embd, dropout)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.sa(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x

class TransformerLanguageModel(nn.Module):
    """
    Tiny causal decoder-only Transformer, character-level.
    """
    def __init__(
        self,
        vocab_size: int,
        n_embd: int,
        n_head: int,
        n_layer: int,
        block_size: int,
        dropout: float,
        device: torch.device,
    ):
        super().__init__()
        self.block_size = block_size
        self.device = device

        self.tok_emb = nn.Embedding(vocab_size, n_embd)
        self.pos_emb = nn.Embedding(block_size, n_embd)
        self.blocks  = nn.Sequential(*[
            Block(n_embd, n_head, block_size, dropout) for _ in range(n_layer)
        ])
        self.ln_f = nn.LayerNorm(n_embd)
        self.head = nn.Linear(n_embd, vocab_size)

        self.apply(self._init_weights)


    def _init_weights(self, m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
        if isinstance(m, nn.Linear) and m.bias is not None:
            nn.init.zeros_(m.bias)


    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None):
        B, T = idx.shape
        # crop (safety) if someone passes longer than block_size
        idx = idx[:, -self.block_size:]
        T = idx.size(1)

        tok = self.tok_emb(idx)                                  # (B, T, C)
        pos = self.pos_emb(torch.arange(T, device=idx.device))   # (T, C)
        x = tok + pos
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.head(x)                                    # (B, T, vocab)

        loss = None
        if targets is not None:
            logits_flat = logits.view(B * T, -1)
            targets_flat = targets[:, -T:].contiguous().view(B * T)
            loss = F.cross_entropy(logits_flat, targets_flat)
        return logits, loss


    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int) -> torch.Tensor:
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size:]
            logits, _ = self(idx_cond)
            probs = F.softmax(logits[:, -1, :], dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_id], dim=1)
        return idx


@torch.no_grad()
def estimate_loss(model, get_batch, eval_iters: int = 100) -> dict:
    """
    Averages cross-entropy on train/val. Model is set to eval/train around this call.
    """
    model.eval()
    out = {}
    for split in ("train", "val"):
        losses = []
        for _ in range(eval_iters):
            xb, yb = get_batch(split)
            _, loss = model(xb, yb)
            losses.append(loss.item())
        out[split] = sum(losses) / len(losses)
    model.train()
    return out


def run_transformer_demo(
    vocab_size: int,
    get_batch,
    decode,
    device,
    *,
    n_embd=64,
    n_head=4,
    n_layer=4,
    block_size=128,
    dropout=0.0,
    iters=3000,
    eval_interval=300,
    lr=1e-2,
) -> None:
    banner("Transformer LM")
    model = TransformerLanguageModel(
        vocab_size=vocab_size,
        n_embd=n_embd,
        n_head=n_head,
        n_layer=n_layer,
        block_size=block_size,
        dropout=dropout,
        device=device,
    ).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=lr)

    for it in range(iters):
        if it % eval_interval == 0:
            losses = estimate_loss(model, get_batch, eval_iters=100)
            print(f"[Transformer] step {it:4d} | train {losses['train']:.4f} | val {losses['val']:.4f}")

        xb, yb = get_batch("train")
        _, loss = model(xb, yb)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

    # Sample a bit at the end so we can see the vibe
    start = torch.zeros((1, 1), dtype=torch.long, device=device)
    sample = model.generate(start, 500)[0].tolist()
    print("[Transformer] sample:")
    print(decode(sample))


# ---------------------------- Toy Attention Demo ------------------------
def toy_attention_demo() -> None:
    """
    Small console-friendly walk-through that goes from simple prefix means
    to learned self-attention weights. Prints shapes and a few tensors.
    """
    banner("Toy: From averages to self-attention")
    torch.manual_seed(42)
    B, T, C = 4, 8, 2
    x = torch.randn(B, T, C)
    print("x.shape:", x.shape)

    # ----- Version 1: Average of all previous tokens (Super simplistic) -----
    # For each time step t, compute the mean of all tokens up to t (prefix mean).
    xbow = torch.zeros_like(x)
    for b in range(B):
        for t in range(T):
            xbow[b, t] = x[b, :t+1].mean(0)
    print("prefix-mean built with loops -> xbow.shape:", xbow.shape)

    # ----- Version 2: Matrix multiplication for weighted aggregation -----
    # Instead of loops, use a lower-triangular weight matrix to achieve the same prefix averaging.
    # This is exactly what self-attention does, but with learned weights instead of fixed averaging.
    wei = torch.tril(torch.ones(T, T))
    wei = wei / wei.sum(1, keepdim=True)
    xbow2 = wei @ x
    print("matrix-avg  -> xbow2.shape:", xbow2.shape)
    print("allclose(xbow, xbow2):", torch.allclose(xbow, xbow2))

    # ----- Version 3: Use Softmax (smooth weighing) -----
    # Replace the uniform averaging with softmax-based weights.
    # Softmax gives a smooth weighting, but here with uniform inputs it ends up the same as mean.
    tril = torch.tril(torch.ones(T, T))
    wei3 = torch.zeros(T, T).masked_fill(tril == 0, float("-inf"))
    wei3 = F.softmax(wei3, dim=-1)
    xbow3 = wei3 @ x
    print("softmax-avg -> xbow3.shape:", xbow3.shape)
    print("allclose(xbow, xbow3):", torch.allclose(xbow, xbow3))

    # ----- Version 4: Self-Attention through Scaled dot-product (learned weights) -----
    banner("Scaled Dot-Product Self-Attention (single head)")
    # Now we make the weights *learned* and *data-dependent* using queries and keys.  We implement
    # a scaled dot-product self-attention mechanism.  This is the core idea behind transformer 
    # model's like GPT.  It lets each sequence 'look back' at previous tokens and decide how
    # much attention to pay to each one when synthesizing the next token (instead of treating
    # all previous tokens equally as in versions 1-3).
    torch.manual_seed(1337)
    C = 32
    head_size = 16
    B, T, C = 4, 8, C
    x = torch.randn(B, T, C)
    key   = nn.Linear(C, head_size, bias=False)
    query = nn.Linear(C, head_size, bias=False)
    value = nn.Linear(C, head_size, bias=False)
    k, q, v = key(x), query(x), value(x)
    wei = (q @ k.transpose(-2, -1)) / (head_size ** 0.5)
    wei = wei.masked_fill(torch.tril(torch.ones(T, T)) == 0, float("-inf"))
    wei = F.softmax(wei, dim=-1)
    out = wei @ v
    print("wei[0]:\n", wei[0])
    print("out[0].shape:", out[0].shape)


# ------------------------------- Main -----------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Tiny character LMs: bigram and a small Transformer.")
    p.add_argument("--block_size", type=int, default=128, help="context size for Transformer")
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--n_embd", type=int, default=128)
    p.add_argument("--n_head", type=int, default=4)
    p.add_argument("--n_layer", type=int, default=4)
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--iters", type=int, default=2000)
    p.add_argument("--eval_interval", type=int, default=200)
    p.add_argument("--bigram_steps", type=int, default=200, help="quick warmup for bigram baseline")
    p.add_argument("--no_bigram", action="store_true", help="skip bigram demo")
    p.add_argument("--no_toy", action="store_true", help="skip toy attention demo")
    return p.parse_args()


def main():
    args = parse_args()
    set_seed(1337)
    device = get_device()

    banner("Device")
    print("Using:", device.type.upper(), end="")
    if device.type == "cuda":
        print(f" ({torch.cuda.get_device_name(0)})")
    else:
        print("")

    banner("Download + Tokenize")
    text = load_tiny_shakespeare()
    print("Loaded tiny Shakespeare: ~{:.1f} KB".format(len(text) / 1024))
    chars, encode, decode = build_char_tokenizer(text)
    vocab_size = len(chars)
    print(f"Vocab size: {vocab_size} | Unique chars: {''.join(chars)}")

    data = torch.tensor(encode(text), dtype=torch.long)
    train_data, val_data = make_splits(data, train_frac=0.9)
    get_batch = make_batcher(train_data, val_data, block_size=args.block_size,
                             batch_size=args.batch_size, device=device)

    if not args.no_bigram:
        run_bigram_demo(vocab_size, get_batch, decode, device, steps=args.bigram_steps)

    run_transformer_demo(
        vocab_size=vocab_size,
        get_batch=get_batch,
        decode=decode,
        device=device,
        n_embd=args.n_embd,
        n_head=args.n_head,
        n_layer=args.n_layer,
        block_size=args.block_size,
        dropout=args.dropout,
        iters=args.iters,
        eval_interval=args.eval_interval,
        lr=args.lr,
    )

    if not args.no_toy:
        toy_attention_demo()




if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted. Exiting cleanly.", file=sys.stderr)


# ------------------------------- Output -----------------------------------
"""
*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
*                          Device                           *
*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

Using: CPU




*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
*                    Download + Tokenize                    *
*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

Loaded tiny Shakespeare: ~1089.3 KB
Vocab size: 65 | Unique chars: 
 !$&',-.3:;?ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz




*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
*                    Baseline: Bigram LM                    *
*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

[Bigram] last training loss: 4.4345
[Bigram] sample:

CJdgknLI!UEHwcObxfNCha!qKt-ocPTFjyXrF
W:..ZKAL.AHA.P!HAXNw,,$zCJ-!or'yxabLWGfkKowCXNe:g;gXEG'uVAPJ$
&AkfNfq-GXlay!B?!
SPSJsBosu$WIgEQzkq$YCZTOiqEMp q?$zriGJl3'IoiKInuJuw
Ci
&CUN
.yff;DRj:Td,&uDK$Wj;Y




*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
*                      Transformer LM                       *
*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

[Transformer] step    0 | train 4.2610 | val 4.2611
[Transformer] step  200 | train 2.2780 | val 2.3054
[Transformer] step  400 | train 1.9156 | val 2.0187
[Transformer] step  600 | train 1.7017 | val 1.8720
[Transformer] step  800 | train 1.5928 | val 1.7699
[Transformer] step 1000 | train 1.5183 | val 1.7154
[Transformer] step 1200 | train 1.4614 | val 1.6596
[Transformer] step 1400 | train 1.4198 | val 1.6340
[Transformer] step 1600 | train 1.3896 | val 1.6182
[Transformer] step 1800 | train 1.3645 | val 1.6097
[Transformer] sample:

Was you as false owell, I would upon your childing,
Relain!

ANCAHCSALUS:
Smile, 'tis not: marsh, our soft, people,
If you gross, that subjecting is to his salworder for his,
Came his perror sweet saince.
For I'll joy provost do since in,
My insolemn I to pront my executione:
Long thyself none handly are joy men sin the enmiony.
Death, good light, and yet you here pray to journ,
An was never to be boldly, which I sleep you give bride.
If see, that is a foote on meat, lo, now, of more,
Wherefore 




*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
*           Toy: From averages to self-attention            *
*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

x.shape: torch.Size([4, 8, 2])
prefix-mean built with loops -> xbow.shape: torch.Size([4, 8, 2])
matrix-avg  -> xbow2.shape: torch.Size([4, 8, 2])
allclose(xbow, xbow2): True
softmax-avg -> xbow3.shape: torch.Size([4, 8, 2])
allclose(xbow, xbow3): True




*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
*      Scaled Dot-Product Self-Attention (single head)      *
*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

wei[0]:
 tensor([[1.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
        [0.3966, 0.6034, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
        [0.3069, 0.2892, 0.4039, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
        [0.3233, 0.2175, 0.2443, 0.2149, 0.0000, 0.0000, 0.0000, 0.0000],
        [0.1479, 0.2034, 0.1663, 0.1455, 0.3369, 0.0000, 0.0000, 0.0000],
        [0.1259, 0.2490, 0.1324, 0.1062, 0.3141, 0.0724, 0.0000, 0.0000],
        [0.1598, 0.1990, 0.1140, 0.1125, 0.1418, 0.1669, 0.1061, 0.0000],
        [0.0845, 0.1197, 0.1078, 0.1537, 0.1086, 0.1146, 0.1558, 0.1553]],
       grad_fn=<SelectBackward0>)
out[0].shape: torch.Size([8, 16])



"""