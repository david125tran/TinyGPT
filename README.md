# TinyGPT: Building a GPT From Scratch

This project is a **from-scratch implementation** of a small GPT-style language model — a miniature version of the architecture described in *“Attention Is All You Need”* (Vaswani (Google Brain) et al., 2017) and popularized by OpenAI’s GPT models. I built this to get a look underneath the hood of transformers, not to write production level code.  
  
I followed along with **Andrej Karpathy’s (OpenAI, Tesla) “Let’s build GPT from scratch”** YouTube video & then refactored all of the code to make it more clear for simplicity and learning.

---
## 🧩 Quick Intro
A **Transformer** is a type of deep learning architecture introduced in the 2017 paper “Attention Is All You Need” by Vaswani et al. It revolutionized natural language processing (NLP) by replacing recurrence (RNNs/LSTMs) with a purely attention-based mechanism.

At its core, a Transformer processes sequences in parallel, not step-by-step, and learns relationships between tokens using self-attention — a way for each token to attend to all others in the sequence.

- **Parallel sequence processing** - Unlike RNNs (which read one token at a time), Transformers look at all tokens simultaneously. This allows for efficient training on GPUs and the ability to model long-range dependencies.
- **Self-attention** - Each token produces three learned vectors — Query, Key, and Value — and computes how much attention to give to every other token.  The result is a weighted average of all token representations, letting the model dynamically focus on relevant context (like subject–verb agreement or long-distance references).
- **Positional embeddings** - Because Transformers have no inherent notion of order, they inject positional information into the token embeddings so the model knows which token came first.
- **Layer normalization and residuals** - Each sublayer (attention, feed-forward) is wrapped with residual connections and LayerNorm, stabilizing training and allowing deeper models.
- **Feed-forward layers** - After self-attention, a small MLP refines each token’s features individually — capturing local nonlinear transformations.
- **Stacked blocks** - Transformers are built by stacking multiple identical layers (“blocks”) — each one learning higher-level abstractions.
In GPT-style models (decoder-only Transformers), each block only looks backward in the sequence using a causal mask.

## 🧠 Why Transformers Matter
- They scale extremely well — performance improves predictably with model size and data.
- They unify many modalities: text, audio, vision, and even protein folding all use Transformer variants today.
- They form the backbone of GPT, BERT, T5, Claude, Gemini, and nearly all modern large language models.

## 💬 Simple summary
- A Transformer is a neural network architecture that learns how each token in a sequence should pay attention to every other token — all at once — enabling it to capture complex dependencies and generate coherent, context-aware outputs.

  
---
## 🚀 Overview

This script walks through the entire process of constructing a small **character-level Transformer** using **PyTorch**, from downloading data to generating text.

It’s organized to make each step clear and reproducible — showing the evolution from a **bigram model** (predict next token from one token) to a **Transformer** (predict using many tokens + self-attention).

- **Bigram language models** were the old-school way of modeling text — they could only predict the next token based on the immediately preceding one, completely ignoring longer context. This meant they could capture local patterns (like letter pairs or short word transitions) but failed to understand global structure such as grammar, meaning, or dependencies across sentences.
- **Transformers**, by contrast, look at all previous tokens at once through a mechanism called **self-attention**. Instead of just memorizing short token pairs, they learn to weigh which prior tokens are most relevant when generating the next one. This allows them to model long-range relationships — capturing not only syntax, but also semantics and context — and is the key innovation behind modern LLMs like GPT.

---

## 🧩 Key Components

### 1. Data Loading
- Downloads **Tiny Shakespeare** dataset.
- Tokenizes dataset at the **character** level using `def build_char_tokenizer(text: str)`.
- Builds `encode()` / `decode()` mappers between characters and integers.
- Splits into train/validation sets.

### 2. Bigram Language Model (Baseline)
- Implements a minimalist bigram model:
  - Predicts the next token based **only** on the current one.
  - Uses a single `nn.Embedding(vocab_size, vocab_size)` layer.
- Trained with cross-entropy loss to sanity-check the data flow.  
- Generates a small text sample after training (random gibberish, by design):  

        CJdgknLI!UEHwcObxfNCha!qKt-ocPTFjyXrF
        W:..ZKAL.AHA.P!HAXNw,,$zCJ-!or'yxabLWGfkKowCXNe:g;gXEG'uVAPJ$
        &AkfNfq-GXlay!B?!
        SPSJsBosu$WIgEQzkq$YCZTOiqEMp q?$zriGJl3'IoiKInuJuw
        Ci
        &CUN
        .yff;DRj:Td,&uDK$Wj;Y

### 3. Transformer Language Model
- Implements a **tiny GPT-like Transformer**:
  - Multi-head self-attention (causal masking)
  - Feed-forward MLP with residual connections
  - Layer normalization
  - Learned positional embeddings
- Trains for a few thousand iterations with `AdamW` optimizer.
- Prints training/validation loss periodically.
- Generates text after training that begins to look Shakespearian-like:

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



### 4. Toy Attention Demo
This section progressively steps through more advanced **weighted aggregation** techniques that lead up to the idea of **self-attention**.  These first three versions are fixed (non-learned) aggregations.  The final version uses learned, content-based weights - the essence of self-attention.  

**Self-attention** - A mechanism that allows a model to dynamically weigh the importance of each token in a sequence when encoding or predicting another token.  In simpler terms, every token (like a word or character) looks at all other tokens in the same sequence and decides how much attention to pay to each one — based on learned relationships.

Different versions:
1. **Version 1**: Average of all previous tokens (Super simplistic)
2. **Version 2**: Matrix multiplication for weighted aggregation
3. **Version 3**: Use Softmax (smooth weighing)
4. **Version 4**: Self-Attention through Scaled Dot-Product (learned weights)

Each version prints tensor shapes and results, showing how the math evolves.

---

## 🛠️ Running It

### Install dependencies
```bash
pip install torch requests
```

---

## 🧠 Design Goals

This repo emphasizes **clarity and readability** over performance.  
Each class and function is documented with plain-English explanations — no hidden magic.

You can read top-to-bottom like a story:
1. Load data →
2. Build vocab →
3. Train a bigram model →
4. Replace with a Transformer →
5. Understand attention mathematically.

---

## 📚 References

- Inspired by **Andrej Karpathy’s** educational work and *“nanoGPT”* series.
- Dataset: [Tiny Shakespeare](https://github.com/karpathy/char-rnn/blob/master/data/tinyshakespeare/input.txt)
- [*“Attention Is All You Need”* (Vaswani et al., 2017)](https://proceedings.neurips.cc/paper_files/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf)
---
