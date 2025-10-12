# ---------------------------------------- Imports ---------------------------------------- 
import requests
import torch
import torch.nn as nn
from torch.nn import functional as F

# ---------------------------------------- Read Text ---------------------------------------- 
shakespeare_dataset = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"

# Send a GET request to the URL
response = requests.get(shakespeare_dataset)

# Store the text content in a variable
text = response.text

# Print the first 500 characters
# print(text[:500])


# ---------------------------------------- Tokenization ---------------------------------------- 
# Get all the unique characters that occur in the text
chars = sorted(list(set(text)))
vocab_size = len(chars)

# print("All the unique characters:", ''.join(chars))
# !$&',-.3:;?ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz

# print("Vocab size:", vocab_size)
# Vocab size: 65

# Create a mapping from characters to integers

# String to integer hash map
str_to_int = {}
for i, ch in enumerate(chars):
    str_to_int[ch] = i

# print(str_to_int)
# {'\n': 0, ' ': 1, '!': 2, '$': 3, '&': 4, "'": 5, ',': 6, '-': 7, '.': 8, '3': 9, ':': 10, ';': 11, '?': 12, 'A': 13, 'B': 14, 'C': 15, 'D': 16, 'E': 17, 'F': 18, 'G': 19, 'H': 20, 'I': 21, 'J': 22, 'K': 23, 'L': 24, 'M': 25, 'N': 26, 'O': 27, 'P': 28, 'Q': 29, 'R': 30, 'S': 31, 'T': 32, 'U': 33, 'V': 34, 'W': 35, 'X': 36, 'Y': 37, 'Z': 38, 'a': 39, 'b': 40, 'c': 41, 'd': 42, 'e': 43, 'f': 44, 'g': 45, 'h': 46, 'i': 47, 'j': 48, 'k': 49, 'l': 50, 'm': 51, 'n': 52, 'o': 53, 'p': 54, 'q': 55, 'r': 56, 's': 57, 't': 58, 'u': 59, 'v': 60, 'w': 61, 'x': 62, 'y': 63, 'z': 64}

# Integer to string hash map
int_to_str = {}
for i, ch in enumerate(chars):
    int_to_str[i] = ch

# print(int_to_str)
# {0: '\n', 1: ' ', 2: '!', 3: '$', 4: '&', 5: "'", 6: ',', 7: '-', 8: '.', 9: '3', 10: ':', 11: ';', 12: '?', 13: 'A', 14: 'B', 15: 'C', 16: 'D', 17: 'E', 18: 'F', 19: 'G', 20: 'H', 21: 'I', 22: 'J', 23: 'K', 24: 'L', 25: 'M', 26: 'N', 27: 'O', 28: 'P', 29: 'Q', 30: 'R', 31: 'S', 32: 'T', 33: 'U', 34: 'V', 35: 'W', 36: 'X', 37: 'Y', 38: 'Z', 39: 'a', 40: 'b', 41: 'c', 42: 'd', 43: 'e', 44: 'f', 45: 'g', 46: 'h', 47: 'i', 48: 'j', 49: 'k', 50: 'l', 51: 'm', 52: 'n', 53: 'o', 54: 'p', 55: 'q', 56: 'r', 57: 's', 58: 't', 59: 'u', 60: 'v', 61: 'w', 62: 'x', 63: 'y', 64: 'z'}

# Encode the entire text dataset
def encode(string):
    """
    Encode a string to a list of integers
    """
    return [str_to_int[c] for c in string]

def decode(list_of_ints):
    """
    Decode a list of integers to a string
    """
    return ''.join([int_to_str[i] for i in list_of_ints])

data = encode("I love eating pizza!")
# print("Encoded text:", data[:20])
# Encoded text: [21, 1, 50, 53, 60, 43, 1, 43, 39, 58, 47, 52, 45, 1, 54, 47, 64, 64, 39, 2]

# print(f"Decoded data: {decode(data)}")
# Decoded data: I love eating pizza!

# Wrap the list of integers into a PyTorch tensor, which is the data structure PyTorch models
# use.  It's like a NumPy array, but optimized for GPU acceleration and Autograd (automatic 
# differentiation).  This allows the LLM to run faster.  Think of a tensor as a generalized 
# matrix that can hold numbers in a grid of any number of dimensions.  

# dtype=torch.long is the required type when passing token IDs to things like embedding layers
data = torch.tensor(encode(text), dtype=torch.long)

# print(data.shape, data.dtype)
# torch.Size([1115394]) torch.int64

# print(data[:15])
# tensor([18, 47, 56, 57, 58,  1, 15, 47, 58, 47, 64, 43, 52, 10,  0])

# ---------------------------------------- Create Train and Validation Data Sets ---------------------------------------- 
# Split the data into train and validation sets so that we can see if our model is overfitting.
# We don't want the model to just memorize the training data, we want it to generalize to new data.

# 90% will be train, 10% will be val
n = int(0.9 * len(data))

# Used to train our LLM
train_data = data[:n]

# Our validation data will be used to see our overfitting
val_data = data[n:]

# How many characters to consider as context for the prediction task.  The model will be trained on chunks
# of Shakespeare text of this length.
block_size = 8

train_data = train_data[:block_size+1]
# print(train_data)
# tensor([18, 47, 56, 57, 58,  1, 15, 47, 58])

# First 8 tokens (context)
x = train_data[:block_size]

# Next 8 tokens (target)
y = train_data[1:block_size+1]

for target in range(block_size):
    context = x[:target+1]
    target = y[target]
    # print(f"When the input is {context} the target: {target}")

# When the input is tensor([18]) the target: 47
# When the input is tensor([18, 47]) the target: 56
# When the input is tensor([18, 47, 56]) the target: 57
# When the input is tensor([18, 47, 56, 57]) the target: 58
# When the input is tensor([18, 47, 56, 57, 58]) the target: 1
# When the input is tensor([18, 47, 56, 57, 58,  1]) the target: 15
# When the input is tensor([18, 47, 56, 57, 58,  1, 15]) the target: 47
# When the input is tensor([18, 47, 56, 57, 58,  1, 15, 47]) the target: 58

# ---------------------------------------- Training ----------------------------------------
# Iteratively get small batches of data to train the model on.  Each batch will contain many
# independent sequences of block_size length.  The model will learn to predict the next token
# in each sequence.  You want to train in batches instead of all of the data at once for   
# manageable memory usage and to get a more accurate estimate of the gradient.

# This also allows the model to learn from a more diverse set of examples in each batch,
# rather that memorizing a single sequence.

torch.manual_seed(1337)
batch_size = 4 # how many independent sequences will we process in parallel?
block_size = 8 # what is the maximum context length for predictions?

def get_batch(split):
    """
    Generate a small batch of data of inputs x and targets y training on many sequences
    in parallel.  This function trains on random chunks of text from the data.  
    """
    if split == 'train':
        data = train_data
    else:
        data = val_data

    # Randomly choose starting indices for each sequence in the batch
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x, y

xb, yb = get_batch('train')
# print('inputs:')
# print(xb.shape)
# print(xb)
# print('targets:')
# print(yb.shape)
# print(yb)

# Visualizing what the model is doing during training.  
#   *For each position t in the sequence,
#   *It looks at the context so far: [x0, x1, ..., xt]
#   *And it learns to predict the next token: yt
# This is exactly how next-token prediction works in LLMs like GPT.

for b in range(batch_size):         # Loop over each example in the batch
    for t in range(block_size):     # Loop over each time step (token position)
        context = xb[b, :t+1]       # Input up to current position
        target = yb[b,t]            # The next token to predict
        # print(f"when input is {context.tolist()} the target: {target}")

# when input is [18] the target: 47
# when input is [18, 47] the target: 56
# when input is [18, 47, 56] the target: 57
# when input is [18, 47, 56, 57] the target: 58
# when input is [18, 47, 56, 57, 58] the target: 1
# when input is [18, 47, 56, 57, 58, 1] the target: 15
# when input is [18, 47, 56, 57, 58, 1, 15] the target: 47
# when input is [18, 47, 56, 57, 58, 1, 15, 47] the target: 58
# when input is [18] the target: 47
# when input is [18, 47] the target: 56
# when input is [18, 47, 56] the target: 57
# when input is [18, 47, 56, 57] the target: 58
# when input is [18, 47, 56, 57, 58] the target: 1
# when input is [18, 47, 56, 57, 58, 1] the target: 15
# when input is [18, 47, 56, 57, 58, 1, 15] the target: 47
# when input is [18, 47, 56, 57, 58, 1, 15, 47] the target: 58
# when input is [18] the target: 47
# when input is [18, 47] the target: 56
# when input is [18, 47, 56] the target: 57
# when input is [18, 47, 56, 57] the target: 58
# when input is [18, 47, 56, 57, 58] the target: 1
# when input is [18, 47, 56, 57, 58, 1] the target: 15
# when input is [18, 47, 56, 57, 58, 1, 15] the target: 47
# when input is [18, 47, 56, 57, 58, 1, 15, 47] the target: 58
# when input is [18] the target: 47
# when input is [18, 47] the target: 56
# when input is [18, 47, 56] the target: 57
# when input is [18, 47, 56, 57] the target: 58
# when input is [18, 47, 56, 57, 58] the target: 1
# when input is [18, 47, 56, 57, 58, 1] the target: 15
# when input is [18, 47, 56, 57, 58, 1, 15] the target: 47
# when input is [18, 47, 56, 57, 58, 1, 15, 47] the target: 58


# The input to the transformer
# print(xb)
# tensor([[18, 47, 56, 57, 58,  1, 15, 47],
#         [18, 47, 56, 57, 58,  1, 15, 47],
#         [18, 47, 56, 57, 58,  1, 15, 47],
#         [18, 47, 56, 57, 58,  1, 15, 47]])

# When training a model using a library like PyTorch or TensorFlow, the library is doing a lot of stuff
# underneath the hood.  It's handling the batching, the randomization of of the data that the model sees,
# the updates to the weights, etc.  Under the hood, it is also training it on a little sequences and 
# teaching it to predict the next 'target' sequence.  For example, it doesn't show the model an entire
# paragraph and then show the target response.  Instead, it shows the model a little chunk of text
# (the context) and then shows it the next token (the target).  It does this for many little chunks
# of text in parallel (the batch).  It does this that way when the the model runs inference, it can
# predict the next token given a little context.  This is how LLMs like GPT work



# ---------------------------------------- Bigram Language Model ----------------------------------------
# Ensure reproducibility of results - same random numbers each time you run
torch.manual_seed(1337)

class BigramLanguageModel(nn.Module):
    """
    This class implements a simple bigram language model.  A bigram model predicts the next token
    based solely on the current token.  It does this using a lookup table (an embedding layer) that
    maps each token to a set of logits for the next token.  The model is trained using cross-entropy loss.

    Bigram models are the old school method of language modeling, and are not very good.  Modern LLMs use
    much more sophisticated architectures (like transformers) that can take into account much more context.

    Unlike modern Transformer-based models (which consider long contexts), this model ignores all context 
    beyond one character and has no attention, positional encoding, or recurrence.

    *No attention mechanism
    *No memory of earlier context beyond the current token

    Example:
        "H" → "e"  
        "He" → "l"  
        "Hel" → "l"  (but bigram model only looks at "e" → "l")
    """

    def __init__(self, vocab_size):
        super().__init__()
        # Create a lookup table that maps each token to a set of logits for the next token.
        # Each token (character index) maps directly to a vocab-sized output vector (logits
        # for the next token).
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

m = BigramLanguageModel(vocab_size)
logits, loss = m(xb, yb)
print(logits.shape)
print(loss)

print(decode(m.generate(idx = torch.zeros((1, 1), dtype=torch.long), max_new_tokens=100)[0].tolist()))
# torch.Size([32, 65])
# tensor(4.4032, grad_fn=<NllLossBackward0>)

# SKIcLT;AcELMoTbvZv C?nq-QE33:CJqkOKH-q;:la!oiywkHjgChzbQ?u!3bLIgwevmyFJGUGp
# wnYWmnxKWWev-tDqXErVKLgJ

# create a PyTorch optimizer
optimizer = torch.optim.AdamW(m.parameters(), lr=1e-3)


# Train the super-simple bigram language model and print its loss.
batch_size = 32
for steps in range(100): # increase number of steps for good results...

    # sample a batch of data
    xb, yb = get_batch('train')

    # evaluate the loss
    logits, loss = m(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

print(loss.item())

batch_size = 32
for steps in range(10000): # increase number of steps for good results...

    # sample a batch of data
    xb, yb = get_batch('train')

    # evaluate the loss
    logits, loss = m(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

print(loss.item())

