## import the required dependencies

import math
import sentencepiece as spm
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.nn.functional as F

from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

## mount google drive

from google.colab import drive
drive.mount('/content/drive')

## use hugging face token to download english to french corpus data

from google.colab import userdata
from huggingface_hub import login

HF_TOKEN = userdata.get("HF_TOKEN")

login(token=HF_TOKEN)

## check device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Device:", device)

## initialize hyperparameters

d_model = 512
num_heads = 8
d_ff = 2048
num_layers = 6

dropout_rate = 0.2
eps = 1e-6

max_len = 128
max_target_length = 128

batch_size = 256
learning_rate = 1e-4

epochs = 10

## initialize token ids
PAD_ID = 0
BOS_ID = 1
EOS_ID = 2
UNK_ID = 3

## download datasets

dataset = load_dataset("Helsinki-NLP/opus-100","en-fr")
print(dataset)

## print dataset

print(dataset)

## load training , validation and test sets

train_data = dataset["train"]

val_data = dataset["validation"]

test_data = dataset["test"]

## print length of loaded training, validation and test examples

print("Training examples:", len(train_data))
print("Validation examples:", len(val_data))
print("Test examples:", len(test_data))

## print training data

print(train_data)

## write training data in a file

with open(
    "shared_train.txt",
    "w",
    encoding="utf-8"
) as f:

    for example in train_data:
        f.write(example["translation"]["en"] + "\n")
        f.write(example["translation"]["fr"] + "\n")

## train sentencepiece tokenizer

spm.SentencePieceTrainer.train(
    input="shared_train.txt",
    model_prefix="shared_tokenizer",
    vocab_size=25000,
    model_type = "bpe",
    pad_id=PAD_ID,
    bos_id=BOS_ID,
    eos_id=EOS_ID,
    unk_id=UNK_ID
)

## invoke the sentencepiece processor

shared_tokenizer = spm.SentencePieceProcessor(
    model_file="shared_tokenizer.model"
)

## test sample sentences

test_sentence = "I love cats"

english_tokens = shared_tokenizer.encode(test_sentence, out_type=str)
print(english_tokens)

english_ids = shared_tokenizer.encode(test_sentence, out_type=int, add_bos=True, add_eos=True)
print(english_ids)

## print results

print(shared_tokenizer.decode(english_ids))

## get vocabulary size

shared_vocab_size = shared_tokenizer.get_piece_size()
print("Shared Vocabulary Size:", shared_vocab_size)

## get source and target sequence lengths for all training examples

source_lengths = []
target_lengths = []

for example in train_data:
    source_ids = shared_tokenizer.encode(
        example["translation"]["en"],
        out_type=int,
        add_bos=True,
        add_eos=True
    )

    target_ids = shared_tokenizer.encode(
        example["translation"]["fr"],
        out_type=int,
        add_bos=True,
        add_eos=True
    )

    source_lengths.append(len(source_ids))
    target_lengths.append(len(target_ids))

## calculate the percentage of training pairs excluded if we cap it to a particular sequence length among the values taken

for cap in [64, 96, 128, 160, 192, 256, 384, 512]:

    source_excluded = sum(
        length > cap for length in source_lengths
    )

    target_excluded = sum(
        length > cap for length in target_lengths
    )

    excluded = sum(
        s > cap or t > cap
        for s, t in zip(source_lengths, target_lengths)
    )

    percentage = excluded / len(train_data) * 100

    print(
        f"Cap {cap}: "
        f"{excluded} pairs excluded "
        f"({percentage:.2f}%)"
    )

## filter out training data or remove training pairs above a particular sequence length as specified in hyperparameters . Here 128 is taken for both english and french sentences

train_data = train_data.filter(
    lambda example:
        len(shared_tokenizer.encode(
            example["translation"]["en"],
            out_type=int,
            add_bos=True,
            add_eos=True
        )) <= max_len
        and
        len(shared_tokenizer.encode(
            example["translation"]["fr"],
            out_type=int,
            add_bos=True,
            add_eos=True
        )) <= max_target_length
)

## validation data pairs filtering

val_data = val_data.filter(
    lambda example:
        len(shared_tokenizer.encode(
            example["translation"]["en"],
            out_type=int,
            add_bos=True,
            add_eos=True
        )) <= max_len
        and
        len(shared_tokenizer.encode(
            example["translation"]["fr"],
            out_type=int,
            add_bos=True,
            add_eos=True
        )) <= max_target_length
)

## tokenize batch function

def tokenize_batch(batch):
    source_ids = [shared_tokenizer.encode(ex["en"], out_type=int, add_bos=True, add_eos=True) for ex in batch["translation"]]
    target_ids = [shared_tokenizer.encode(ex["fr"], out_type=int, add_bos=True, add_eos=True) for ex in batch["translation"]]
    return {"source_ids": source_ids, "target_ids": target_ids}

## tokenization of entire training and validation sets

train_tokenized = train_data.map(tokenize_batch, batched=True, num_proc=4)
val_tokenized = val_data.map(tokenize_batch, batched=True, num_proc=4)

## dataset class
# returns one training pair at a time

class PreTokenizedDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        source = torch.tensor(self.data[idx]["source_ids"], dtype=torch.long)
        target = torch.tensor(self.data[idx]["target_ids"], dtype=torch.long)
        return source, target

## pass tokenized data into dataset class

train_dataset = PreTokenizedDataset(train_tokenized)
val_dataset = PreTokenizedDataset(val_tokenized)

## collate function
# used for dynamic padding

def collate_fn(batch):

  sources, targets = zip(*batch)

  sources = pad_sequence(sources, batch_first=True, padding_value=PAD_ID)
  targets = pad_sequence(targets, batch_first=True, padding_value=PAD_ID)

  return sources, targets

## dataloaders
# num_workers and pin memory used to reduce cpu bottlenecks and improve speed of batch transfer to gpu

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True, collate_fn=collate_fn)

val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True, collate_fn=collate_fn)

## embeddings class

class InputEmbeddings(nn.Module):

  def __init__(self, vocab_size, d_model, pad_token_id):
    super().__init__()

    self.d_model = d_model

    self.embedding = nn.Embedding(vocab_size, d_model, padding_idx = pad_token_id)

  def forward(self, x):

    return self.embedding(x) * math.sqrt(self.d_model)

## positional encodings
# here we have used sin and cos functions

class PositionalEncoding(nn.Module):

  def __init__(self, max_len, d_model, dropout):
    super().__init__()

    position = torch.arange(0, max_len).unsqueeze(-1)

    i = torch.arange(0, d_model // 2)

    division = torch.exp(-math.log(10000) * (2 * i) / d_model)

    pe = torch.zeros(max_len, d_model)

    pe[:, 0::2] = torch.sin(position * division)
    pe[:, 1::2] = torch.cos(position * division)

    pe = pe.unsqueeze(0)

    self.register_buffer("pe", pe)

    self.dropout = nn.Dropout(dropout)

  def forward(self, x):

    seq_len = x.shape[1]

    pe = self.pe[:, :seq_len, :]

    x = x + pe

    x = self.dropout(x)

    return x

## multiheadattention class
# used flash attention for speed and compatibility with floating point 16 bit precision

class MultiHeadAttention(nn.Module):

  def __init__(self, d_model, num_heads, dropout):
    super().__init__()

    assert(d_model % num_heads == 0)

    self.d_model = d_model
    self.num_heads = num_heads

    self.d_k = d_model // num_heads

    self.w_q = nn.Linear(d_model, d_model)
    self.w_k = nn.Linear(d_model, d_model)
    self.w_v = nn.Linear(d_model, d_model)
    self.w_o = nn.Linear(d_model, d_model)

    self.dropout = nn.Dropout(dropout)

  def forward(self, query_input, key_value_input, attention_mask=None):

    batch_size = query_input.shape[0]

    query_seq_len = query_input.shape[1]

    kv_seq_len = key_value_input.shape[1]

    q = self.w_q(query_input)

    k = self.w_k(key_value_input)

    v = self.w_v(key_value_input)

    q = q.reshape(batch_size, query_seq_len, self.num_heads, self.d_k)
    k = k.reshape(batch_size, kv_seq_len, self.num_heads, self.d_k)
    v = v.reshape(batch_size, kv_seq_len, self.num_heads, self.d_k)

    q = q.transpose(1, 2)
    k = k.transpose(1, 2)
    v = v.transpose(1, 2)

    if attention_mask is not None:
      attn_mask = ~attention_mask

    else:
      attn_mask = None

    attention_output = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, dropout_p=self.dropout.p if self.training else 0.0)

    attention_output = attention_output.transpose(1, 2).contiguous()

    attention_output = attention_output.view(batch_size, query_seq_len, self.d_model)

    output = self.w_o(attention_output)

    return output

## feedforward class

class FeedForward(nn.Module):
  def __init__(self, d_model, d_ff, dropout):
    super().__init__()

    self.linear1 = nn.Linear(d_model, d_ff)

    self.relu = nn.ReLU()

    self.linear2 = nn.Linear(d_ff, d_model)

    self.dropout = nn.Dropout(dropout)

  def forward(self, x):
    x = self.linear1(x)
    x = self.relu(x)
    x = self.linear2(x)
    x = self.dropout(x)
    return x

## encoderblock class

class EncoderBlock(nn.Module):
  def __init__(self, d_model, num_heads, d_ff, eps, dropout):
    super().__init__()

    self.self_attention = MultiHeadAttention(d_model, num_heads, dropout)

    self.ffn = FeedForward(d_model, d_ff, dropout)

    self.ln1 = nn.LayerNorm(d_model, eps)

    self.ln2 = nn.LayerNorm(d_model, eps)

  def forward(self, x, encoder_attention_mask):

    self_attn_output = self.self_attention(x, x, encoder_attention_mask)

    x = x + self_attn_output

    x = self.ln1(x)

    ffn_output = self.ffn(x)

    x = x + ffn_output

    x = self.ln2(x)

    return x

## encoder class

class Encoder(nn.Module):
  def __init__(self, vocab_size, max_len, d_model, num_heads, d_ff, eps, num_layers, dropout, pad_token_id):
    super().__init__()

    self.encoder_input_embeddings = InputEmbeddings(vocab_size, d_model, pad_token_id)

    self.encoder_pos = PositionalEncoding(max_len, d_model, dropout)

    self.modulelist = nn.ModuleList()

    for _ in range(num_layers):

      encoder_block = EncoderBlock(d_model, num_heads, d_ff, eps, dropout)

      self.modulelist.append(encoder_block)

  def forward(self, x, encoder_attention_mask):
    x = self.encoder_input_embeddings(x)

    x = self.encoder_pos(x)

    for layer in self.modulelist:
      x = layer(x, encoder_attention_mask)

    return x

## decoderblock class

class DecoderBlock(nn.Module):
  def __init__(self, d_model, num_heads, d_ff, eps, dropout):
    super().__init__()

    self.self_attention = MultiHeadAttention(d_model, num_heads, dropout)

    self.crossattention = MultiHeadAttention(d_model, num_heads, dropout)

    self.ffn = FeedForward(d_model, d_ff, dropout)

    self.ln1 = nn.LayerNorm(d_model, eps)

    self.ln2 = nn.LayerNorm(d_model, eps)

    self.ln3 = nn.LayerNorm(d_model, eps)

  def forward(self, x, encoder_output, decoder_attention_mask, encoder_attention_mask):

    self_attn_output = self.self_attention(x, x, decoder_attention_mask)

    x = x + self_attn_output

    x = self.ln1(x)

    cross_attn_output = self.crossattention(x, encoder_output, encoder_attention_mask)

    x = x + cross_attn_output

    x = self.ln2(x)

    ffn_output = self.ffn(x)

    x = x + ffn_output

    x = self.ln3(x)

    return x

## decoder class

class Decoder(nn.Module):
  def __init__(self, target_vocab_size, max_target_length, d_model, num_heads, d_ff, eps, num_layers, dropout, pad_token_id):
    super().__init__()

    self.decoder_input_embeddings = InputEmbeddings(target_vocab_size, d_model, pad_token_id)

    self.decoder_pos = PositionalEncoding(max_target_length, d_model, dropout)

    self.modulelist = nn.ModuleList()

    for _ in range(num_layers):

      decoder_block = DecoderBlock(d_model, num_heads, d_ff, eps, dropout)

      self.modulelist.append(decoder_block)

  def forward(self, x , encoder_output, decoder_attention_mask, encoder_attention_mask):
    x = self.decoder_input_embeddings(x)

    x = self.decoder_pos(x)

    for layer in self.modulelist:
        x = layer(x, encoder_output, decoder_attention_mask, encoder_attention_mask)

    return x

## padding class

def create_padding_mask(x, pad_token_id):
  mask = (x == pad_token_id)

  mask = mask.unsqueeze(1)

  mask = mask.unsqueeze(2)

  return mask

## custom causal mask
# here mask returns true for future positions
# mask inversion is done in multiheadattention class since we are using F.scaled_dot_product_attention

def create_causal_mask(seq_len, device):
  mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()

  mask = mask.unsqueeze(0)

  mask = mask.unsqueeze(1)

  mask = mask.to(device)

  return mask

## decoder attention mask

def create_decoder_attention_mask(decoder_input, pad_token_id):
  decoder_padding_mask = create_padding_mask(decoder_input, pad_token_id)

  seq_len = decoder_input.shape[1]

  decoder_causal_mask = create_causal_mask(seq_len, decoder_input.device)

  decoder_attention_mask = decoder_padding_mask | decoder_causal_mask

  return decoder_attention_mask

## transformer class

class Transformer(nn.Module):
  def __init__(self, vocab_size, target_vocab_size, max_len, max_target_length, d_model, num_heads, d_ff, eps, num_layers, pad_token_id, dropout):
    super().__init__()

    self.pad_token_id = pad_token_id

    self.encoder = Encoder(vocab_size=vocab_size, max_len=max_len, d_model=d_model, num_heads=num_heads, d_ff=d_ff, eps=eps, num_layers=num_layers, dropout=dropout, pad_token_id=pad_token_id)

    self.decoder = Decoder(target_vocab_size=target_vocab_size, max_target_length=max_target_length, d_model=d_model, num_heads=num_heads, d_ff=d_ff, eps=eps, num_layers=num_layers, dropout=dropout, pad_token_id=pad_token_id)

    self.output_projection = nn.Linear(d_model, target_vocab_size)

  def forward(self, source_input, target_input):

    encoder_mask = create_padding_mask(source_input, self.pad_token_id)

    decoder_mask = create_decoder_attention_mask(target_input, self.pad_token_id)

    encoder_output = self.encoder(source_input, encoder_mask)

    decoder_output = self.decoder(target_input, encoder_output, decoder_mask, encoder_mask)

    output = self.output_projection(decoder_output)

    return output

## instantiate the class

model = Transformer(vocab_size=shared_vocab_size, target_vocab_size=shared_vocab_size, max_len=max_len, max_target_length=max_target_length, d_model=d_model, num_heads=num_heads, d_ff=d_ff, eps=eps, num_layers=num_layers, pad_token_id=PAD_ID, dropout=dropout_rate)

## move model to device

model = model.to(device)

## xavier initialization is used for embedding layer
# embedding layer and output projection layer use same set of parameters also called as weight tying.
# normally embedding layers use normal distribution for initialization but since here weights are shared between embedding and output projection we use xavier initialization
# if we dont use xavier due to parameter sharing output projection layer weights explode in floating point 16 bit precision and cause output logits to be NaN and gradients to be NaN

nn.init.xavier_uniform_(model.encoder.encoder_input_embeddings.embedding.weight)

## weight tying
# done for both encoder and decoder embedding layers and output projection
# helps in reduction of trainable parameters

model.encoder.encoder_input_embeddings.embedding.weight = model.decoder.decoder_input_embeddings.embedding.weight
model.output_projection.weight = model.decoder.decoder_input_embeddings.embedding.weight

## check number of parameters

params = sum(p.numel() for p in model.parameters())

print(f" parameters: {params:,}")

## uses label smoothing 
# reduces overconfidence in prediction

loss_fn = nn.CrossEntropyLoss(ignore_index=PAD_ID, label_smoothing=0.1)

## learning rate scheduler with warmup steps

warmup_steps = 4000

def get_lr_multiplier(step):
    if step == 0:
        step = 1
    return (d_model ** -0.5) * min(step ** -0.5, step * (warmup_steps ** -1.5))

optimizer = optim.Adam(
    model.parameters(),
    lr=1.0,
    betas=(0.9, 0.98),
    eps=eps,
    weight_decay=1e-4
)

scheduler = lr_scheduler.LambdaLR(
    optimizer,
    lr_lambda=get_lr_multiplier
)

## validation loss function

def evaluate_loss(model, val_loader, loss_fn, device):

  model.eval()

  total_loss = 0

  with torch.no_grad():

    for source_input, target in val_loader:

      source_input = source_input.to(device)

      target = target.to(device)

      target_input = target[:, :-1]

      target_output = target[:, 1:]

      logits = model(source_input, target_input)

      logits = logits.transpose(1, 2)

      loss = loss_fn(logits, target_output)

      total_loss += loss.item()

  return total_loss / len(val_loader)

## training loop
# uses mixed precision floating point 16 bit for training

scaler = torch.amp.GradScaler('cuda')
best_val_loss = float('inf')

for epoch in range(epochs):
    model.train()
    epoch_loss = 0

    for source_input, target in train_loader:
        # non_blocking=True pairs with pin_memory=True for faster transfers
        source_input = source_input.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        target_input = target[:, :-1]
        target_output = target[:, 1:]

        optimizer.zero_grad()

        # Enables A100 Tensor Cores
        with torch.amp.autocast('cuda'):
            logits = model(source_input, target_input)
            logits = logits.transpose(1, 2)
            loss = loss_fn(logits, target_output)

        scaler.scale(loss).backward()
        scaler.step(optimizer)

        old_scale = scaler.get_scale()
        scaler.update()
        new_scale = scaler.get_scale()

        if new_scale >= old_scale:
            scheduler.step()

        epoch_loss += loss.item()

    train_loss = epoch_loss / len(train_loader)
    val_loss = evaluate_loss(model, val_loader, loss_fn, device)

    print(f"Epoch {epoch + 1}/{epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), "/content/drive/MyDrive/transformer_en_fr.pt")
        print("--> Saved new best model.")

## beam search decoding

def beam_search_decode(
    model,
    source_input,
    bos_token_id,
    eos_token_id,
    max_length,
    device,
    beam_size=4,
    alpha=0.6
):
    model.eval()
    source_input = source_input.to(device)

    # 1. Run the encoder ONCE for all beams
    with torch.no_grad():
        encoder_mask = create_padding_mask(source_input, model.pad_token_id)
        encoder_output = model.encoder(source_input, encoder_mask)

    # 2. Duplicate the encoder output for our parallel beams
    encoder_output = encoder_output.repeat(beam_size, 1, 1)
    encoder_mask = encoder_mask.repeat(beam_size, 1, 1, 1)

    # 3. Initialize beams with [BOS] token
    generated = torch.full((beam_size, 1), bos_token_id, dtype=torch.long, device=device)

    # Beam scores: First beam starts at 0, others at -infinity
    # so they don't all pick the exact same starting word
    beam_scores = torch.zeros(beam_size, device=device)
    beam_scores[1:] = -1e9

    finished_beams = []

    with torch.no_grad():
        for step in range(max_length):
            # Run decoder
            decoder_mask = create_decoder_attention_mask(generated, model.pad_token_id)
            decoder_output = model.decoder(generated, encoder_output, decoder_mask, encoder_mask)

            # Get logits for the very last generated token
            logits = model.output_projection(decoder_output[:, -1, :])
            log_probs = torch.log_softmax(logits, dim=-1)

            vocab_size = log_probs.shape[-1]

            # Add current beam scores to the new probabilities
            next_scores = beam_scores.unsqueeze(1) + log_probs
            next_scores = next_scores.view(-1) # Flatten to 1D

            # Find the top 'k' combinations across ALL beams
            topk_scores, topk_indices = torch.topk(next_scores, beam_size)

            # Map flat indices back to which beam they came from, and which token they are
            beam_indices = topk_indices // vocab_size
            token_indices = topk_indices % vocab_size

            # Update our sequences
            next_generated = torch.cat(
                [generated[beam_indices], token_indices.unsqueeze(1)],
                dim=-1
            )
            beam_scores = topk_scores

            # Check if any beams hit the [EOS] token
            new_generated = []
            new_scores = []

            for i in range(beam_size):
                if token_indices[i].item() == eos_token_id:
                    # Apply Vaswani's Length Penalty
                    length = next_generated[i].shape[0]
                    length_penalty = ((5.0 + length) / 6.0) ** alpha

                    finished_beams.append({
                        "seq": next_generated[i],
                        "score": beam_scores[i] / length_penalty
                    })

                    # Kill this beam for future steps so it doesn't keep running
                    beam_scores[i] = -1e9

                new_generated.append(next_generated[i])
                new_scores.append(beam_scores[i])

            generated = torch.stack(new_generated)
            beam_scores = torch.stack(new_scores)

            # Stop if all beams have finished
            if all(score == -1e9 for score in beam_scores):
                break

    # If we hit max_length and some beams never finished, collect them anyway
    if len(finished_beams) == 0:
        for i in range(beam_size):
            length = generated[i].shape[0]
            length_penalty = ((5.0 + length) / 6.0) ** alpha
            finished_beams.append({
                "seq": generated[i],
                "score": beam_scores[i] / length_penalty
            })

    # Sort finished branches by their penalized score
    finished_beams.sort(key=lambda x: x["score"], reverse=True)

    # Return the absolute best sequence branch (formatted to match the old shape)
    return finished_beams[0]["seq"].unsqueeze(0)

## translate test sentence function

def translate(source_text, model, source_tokenizer, target_tokenizer, device):

  source_ids = source_tokenizer.encode(source_text, out_type=int, add_bos=True, add_eos=True)

  source_tensor = torch.tensor(source_ids, dtype=torch.long, device=device).unsqueeze(0)

  generated = beam_search_decode(model, source_tensor, BOS_ID, EOS_ID, max_target_length, device)

  translation = target_tokenizer.decode(generated.squeeze(0).tolist())

  return translation

## test data filter

test_data = test_data.filter(
    lambda example:
        len(shared_tokenizer.encode(
            example["translation"]["en"],
            out_type=int,
            add_bos=True,
            add_eos=True
        )) <= max_len
        and
        len(shared_tokenizer.encode(
            example["translation"]["fr"],
            out_type=int,
            add_bos=True,
            add_eos=True
        )) <= max_target_length
)

## evaluate test sentence

model.load_state_dict(torch.load("/content/drive/MyDrive/transformer_en_fr.pt", weights_only=True))

model.eval()

example_test = test_data[10]
source_text = example_test["translation"]["en"]
reference_text = example_test["translation"]["fr"]

with torch.no_grad():
    prediction = translate(source_text, model, shared_tokenizer, shared_tokenizer, device)

print("English:", source_text)
print("French:", reference_text)
print("Prediction:", prediction)

## use bleu evaluation metric with smoothing function
# smoothing function to avoid 0 multiplication errors

from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction

def evaluate_bleu(model, test_data, source_tokenizer, target_tokenizer, device, max_length):
    model.eval()
    references = []
    hypotheses = []

    chencherry = SmoothingFunction()

    for example in test_data.select(range(100)):
        source_text = example["translation"]["en"]
        target_text = example["translation"]["fr"]

        source_ids = source_tokenizer.encode(source_text, out_type=int, add_bos=True, add_eos=True)
        source_tensor = torch.tensor(source_ids, dtype=torch.long, device=device).unsqueeze(0)

        generated = beam_search_decode(model, source_tensor, BOS_ID, EOS_ID, max_length, device)
        prediction = target_tokenizer.decode(generated.squeeze(0).tolist())

        reference_tokens = target_text.strip().split()
        prediction_tokens = prediction.strip().split()

        references.append([reference_tokens])
        hypotheses.append(prediction_tokens)

    bleu = corpus_bleu(references, hypotheses, smoothing_function=chencherry.method1)

    return bleu

## print bleu score over a set of test examples

bleu_score = evaluate_bleu(model, test_data, shared_tokenizer, shared_tokenizer, device, max_target_length)

print(f"BLEU: {bleu_score * 100:.2f}")

## install onnx runtime

!pip install onnxruntime

## import onnxruntime

import onnxruntime as ort

## load model

model.load_state_dict(torch.load("/content/drive/MyDrive/transformer_en_fr.pt", weights_only=True))

## install other dependencies

!pip install onnx onnxscript

## export model to onnx(open neural network exchange) format for inference purposes

model.eval()

# 1. Prepare model for export (CPU prevents hardware-specific errors)
model.to('cpu')

# 2. Create dummy inputs (batch_size=1, sequence_lengths=12 and 5)
dummy_source = torch.randint(0, 25000, (1, 12), dtype=torch.long)
dummy_target = torch.randint(0, 25000, (1, 5), dtype=torch.long)

# 3. Export to ONNX
torch.onnx.export(
    model,
    (dummy_source, dummy_target),
    "transformer_translation.onnx",
    export_params=True,
    opset_version=14,
    do_constant_folding=True,
    input_names=["source_ids", "target_ids"],
    output_names=["logits"],
    # dynamic_axes allows users to input sentences of any length, not just 12 or 5
    dynamic_axes={
        "source_ids": {0: "batch_size", 1: "src_seq_len"},
        "target_ids": {0: "batch_size", 1: "tgt_seq_len"},
        "logits": {0: "batch_size", 1: "tgt_seq_len"}
    }
)

print("Export complete. Download transformer_translation.onnx from the Colab file browser.")

