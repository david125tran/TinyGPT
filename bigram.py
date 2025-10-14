# ---------------------------------------- Imports ---------------------------------------- 
import requests
import torch
import torch.nn as nn
from torch.nn import functional as F


# ---------------------------------------- Hyperparameters ---------------------------------------- 
# Independent sequences processed in parallel 
batch_size = 32
# Maximum context length for predictions
block_size = 8
# Number of training iterations
max_iters = 3000
# Number of iterations between evaluations
eval_interval = 300
# Learning rate for the optimizer
learning_rate = 1e-2
# Device configuration: use GPU if available, otherwise CPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'
# Number of iterations for loss evaluation
eval_iters = 200


# ---------------------------------------- GPU or CPU ---------------------------------------- 
if torch.cuda.is_available():
    print(torch.cuda.get_device_name(0))
else:
    print("Running on CPU")


# ---------------------------------------- Read Text ---------------------------------------- 
torch.manual_seed(1337)
shakespeare_dataset = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
# Send a GET request to the URL
response = requests.get(shakespeare_dataset)

# Store the text content in a variable
text = response.text


# ---------------------------------------- Tokenization ---------------------------------------- 
# All of the unique characters that occur in this text
chars = sorted(list(set(text)))
vocab_size = len(chars)
print("all the unique characters:", ''.join(chars))
print(f"vocab size: {vocab_size}")

# Create a mapping from characters to integers
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s] # encoder: take a string, output a list of integers
decode = lambda l: ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string

# Train and test splits
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9*len(data)) # first 90% will be train, rest val
train_data = data[:n]
val_data = data[n:]

# Data loading
def get_batch(split):
    # Generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

# Estimate loss on train and val sets
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# Super simple bigram model
class BigramLanguageModel(nn.Module):

    def __init__(self, vocab_size):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets=None):

        # idx and targets are both (B,T) tensor of integers
        logits = self.token_embedding_table(idx) # (B,T,C)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # get the predictions
            logits, loss = self(idx)
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx

model = BigramLanguageModel(vocab_size)
m = model.to(device)

# Create a PyTorch optimizer which is used to update the model weights based on the computed gradients.
# When you fine-tune a model, you will typically create an optimizer after you have created the model.
# There are many different optimization algorithms available in PyTorch, each with its own strengths 
# and weaknesses. The AdamW optimizer is a variant of the Adam optimizer that includes weight decay,
# which can help to prevent overfitting.  

# The main differences between the different optimizers are how they update the model weights at each
# step to influence the learning process. At each step of training, the optimizer will change the
# weights of different parameters in different ways based on the specific algorithm being used.  You
# don't cherry pick which parameters it changes, the algorithm decides underneath the hood. 
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for iter in range(max_iters):

    # Every once in a while evaluate the loss on train and val sets
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    # sample a batch of data
    xb, yb = get_batch('train')

    # Forward pass - Make adjustments to the model.  Evaluate the impact of those adjustments (loss).
    logits, loss = model(xb, yb)
    # Backward pass - Compute the gradient of the loss with respect to each parameter of the model to 
    # see how to change the parameters to reduce the loss.
    optimizer.zero_grad(set_to_none=True)
    # Backpropagation - Redo the adjustments to the model based on the computed gradients.
    loss.backward()
    # Update the parameters with the new gradients.
    optimizer.step()

# generate from the model
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))

# step 0: train loss 4.7305, val loss 4.7241
# step 300: train loss 2.8110, val loss 2.8249
# step 600: train loss 2.5434, val loss 2.5682
# step 900: train loss 2.4932, val loss 2.5088
# step 1200: train loss 2.4863, val loss 2.5035
# step 1500: train loss 2.4665, val loss 2.4921
# step 1800: train loss 2.4683, val loss 2.4936
# step 2100: train loss 2.4696, val loss 2.4846
# step 2400: train loss 2.4638, val loss 2.4879
# step 2700: train loss 2.4738, val loss 2.4911

# MARI he avayokis erceller thour d, myono thishe me tord se by he me, Forder anen: at trselorinjulour t yoru thrd wo ththathy IUShe bavidelanoby man ond be jus as g e atot Meste hrle s, ppat t JLENCOLIUS:
# Oppid tes d s o ged moer y pevehear soue maramapay fo t: bueyo malalyo!
# Duir.
# Fl ke it I t l o'ddre d ondu s?
# cr, havetrathackes w.
# PUpee meshancun, hrendspouthoulouren whel's'sesoread pe, s whure our heredinsethes; sedsend r lo pamit,
# QUMIVIVIOfe m ne RDINid we tr ort; t:
# MINENXI l dintandore r


