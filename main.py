from __future__ import print_function

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import savgol_filter

from six.moves import xrange

import umap

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.optim as optim
# %% md
## Create run dir and init logging
# %%
from datetime import datetime
import os
import logging

run_id = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
run_dir = f'./runs/{run_id}'
os.makedirs(run_dir, exist_ok=True)
min_loss_model_path = f'{run_dir}/min-loss-model.pt'
final_model_path = f'{run_dir}/final-model.pt'

logger = logging.getLogger(f'{run_id}')
logger.setLevel(logging.DEBUG)

ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
logger.addHandler(ch)

fh = logging.FileHandler(filename=f'{run_dir}/info.log', encoding='utf-8')
fh.setLevel(logging.INFO)
logger.addHandler(fh)

fh = logging.FileHandler(filename=f'{run_dir}/debug.log', encoding='utf-8')
fh.setLevel(logging.DEBUG)
logger.addHandler(fh)

# %%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# %% md
## Load Data
# %%
import gensim.downloader as api

model_name = 'glove-wiki-gigaword-50'
wv = api.load(model_name)
# %%
from torch.utils.data import Dataset, DataLoader


class WordEmbeddingDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


# %%
import ssl

ssl._create_default_https_context = ssl._create_unverified_context
import nltk

nltk.download('averaged_perceptron_tagger')

all_data = wv.vectors[:20000]
pos = nltk.pos_tag(wv.index_to_key[:len(all_data)])
all_words = np.array(wv.index_to_key[:len(all_data)])
idx = np.array([i for i, v in enumerate(all_data) if
                pos[i][1] in ['NN', 'JJ', 'RB', 'VB'] and all_words[i] not in ['"', "n't", '_', '-', 'u.s.', '%', 'i']])
all_data = all_data[idx]
all_words = all_words[idx]
all_data_len = len(all_data)
training_data_len = int(all_data_len * 0.8)
training_data = WordEmbeddingDataset(all_data[:training_data_len])
validation_data = WordEmbeddingDataset(all_data[training_data_len:])
# %%
data_norms = np.linalg.norm(training_data.data, axis=1)
data_variance = np.var(
    (data_norms - np.min(data_norms)) /
    (np.max(data_norms) - np.min(data_norms))
)
cosine_similarity_eps = 1e-10
# %% md
## Vector Quantizer Layer


class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost):
        super(VectorQuantizer, self).__init__()

        self._embedding_dim = embedding_dim
        self._num_embeddings = num_embeddings

        self._embedding = nn.Embedding(self._num_embeddings, self._embedding_dim)
        self._embedding.weight.data.uniform_(-1 / self._num_embeddings, 1 / self._num_embeddings)
        self._commitment_cost = commitment_cost

    def forward(self, inputs):
        inputs = inputs.contiguous()
        input_shape = inputs.shape

        # Flatten input
        flat_input = inputs.view(-1, self._embedding_dim)

        # Calculate distances
        distances = (torch.sum(flat_input ** 2, dim=1, keepdim=True)
                     + torch.sum(self._embedding.weight ** 2, dim=1)
                     - 2 * torch.matmul(flat_input, self._embedding.weight.t()))

        # Encoding
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self._num_embeddings, device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1)

        # Quantize and unflatten
        quantized = torch.matmul(encodings, self._embedding.weight).view(input_shape)

        # Loss
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        q_latent_loss = F.mse_loss(quantized, inputs.detach())
        loss = q_latent_loss + self._commitment_cost * e_latent_loss

        quantized = inputs + (quantized - inputs).detach()
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        # convert quantized from BHWC -> BCHW
        return loss, quantized.contiguous(), perplexity, encodings


# %% md

class VectorQuantizerEMA(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost, decay, epsilon=1e-5):
        super(VectorQuantizerEMA, self).__init__()

        self._embedding_dim = embedding_dim
        self._num_embeddings = num_embeddings

        self._embedding = nn.Embedding(self._num_embeddings, self._embedding_dim)
        self._embedding.weight.data.normal_()
        self._commitment_cost = commitment_cost

        self.register_buffer('_ema_cluster_size', torch.zeros(num_embeddings))
        self._ema_w = nn.Parameter(torch.Tensor(num_embeddings, self._embedding_dim))
        self._ema_w.data.normal_()

        self._decay = decay
        self._epsilon = epsilon

    def forward(self, inputs):
        inputs = inputs.contiguous()
        input_shape = inputs.shape

        # Flatten input
        flat_input = inputs.view(-1, self._embedding_dim)

        # Calculate distances
        distances = (torch.sum(flat_input ** 2, dim=1, keepdim=True)
                     + torch.sum(self._embedding.weight ** 2, dim=1)
                     - 2 * torch.matmul(flat_input, self._embedding.weight.t()))

        # Encoding
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self._num_embeddings, device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1)

        # Quantize and unflatten
        quantized = torch.matmul(encodings, self._embedding.weight).view(input_shape)

        # Use EMA to update the embedding vectors
        if self.training:
            self._ema_cluster_size = self._ema_cluster_size * self._decay + \
                                     (1 - self._decay) * torch.sum(encodings, 0)

            # Laplace smoothing of the cluster size
            n = torch.sum(self._ema_cluster_size.data)
            self._ema_cluster_size = (
                    (self._ema_cluster_size + self._epsilon)
                    / (n + self._num_embeddings * self._epsilon) * n)

            dw = torch.matmul(encodings.t(), flat_input)
            self._ema_w = nn.Parameter(self._ema_w * self._decay + (1 - self._decay) * dw)

            self._embedding.weight = nn.Parameter(self._ema_w / self._ema_cluster_size.unsqueeze(1))

        # Loss
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        loss = self._commitment_cost * e_latent_loss

        # Straight Through Estimator
        quantized = inputs + (quantized - inputs).detach()
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        return loss, quantized.contiguous(), perplexity, encodings



## Encoder & Decoder Architecture
class Residual(nn.Module):
    def __init__(self, num_hiddens, num_residual_hiddens):
        super(Residual, self).__init__()
        self._block = nn.Sequential(
            nn.ReLU(True),
            nn.Linear(in_features=num_hiddens,
                      out_features=num_residual_hiddens),
            nn.ReLU(True),
            nn.Linear(in_features=num_residual_hiddens,
                      out_features=num_hiddens)
        )

    def forward(self, x):
        return x + self._block(x)


class ResidualStack(nn.Module):
    def __init__(self, num_hiddens, num_residual_layers, num_residual_hiddens):
        super(ResidualStack, self).__init__()
        self._num_residual_layers = num_residual_layers
        self._layers = nn.ModuleList([Residual(num_hiddens, num_residual_hiddens)
                                      for _ in range(self._num_residual_layers)])

    def forward(self, x):
        for i in range(self._num_residual_layers):
            x = self._layers[i](x)
        return F.relu(x)


# %%
class Encoder(nn.Module):
    def __init__(self, num_hiddens, num_residual_layers, num_residual_hiddens, embedding_dim):
        super(Encoder, self).__init__()

        self._lin_1 = nn.Linear(in_features=embedding_dim,
                                out_features=num_hiddens)

        self._lin_2 = nn.Linear(in_features=num_hiddens,
                                out_features=num_hiddens)

        self._lin_3 = nn.Linear(in_features=num_hiddens,
                                out_features=num_hiddens)

        self._residual_stack = ResidualStack(num_hiddens=num_hiddens,
                                             num_residual_layers=num_residual_layers,
                                             num_residual_hiddens=num_residual_hiddens)

    def forward(self, inputs):
        x = self._lin_1(inputs)
        x = F.relu(x)

        x = self._lin_2(x)
        x = F.relu(x)

        x = self._lin_2(x)
        x = F.relu(x)

        x = self._lin_2(x)
        x = F.relu(x)

        x = self._lin_2(x)
        x = F.relu(x)

        x = self._lin_3(x)
        return self._residual_stack(x)


# %%
class Decoder(nn.Module):
    def __init__(self, embedding_dim, definition_len):
        super(Decoder, self).__init__()
        self._embedding_dim = embedding_dim
        self._definition_len = definition_len

    def forward(self, inputs):
        definition = inputs.view((-1, self._definition_len, self._embedding_dim))
        recon = torch.mean(definition, 1)
        return recon, definition


batch_size = 1000
num_training_updates = 100000

num_hiddens = 128
num_residual_hiddens = 32
num_residual_layers = 2

embedding_dim = 50
num_embeddings = training_data_len
definition_len = 5

commitment_cost = 0.25

decay = 0.99

learning_rate = 1e-5

logger.info(f'model_name={model_name}')
logger.info(f'batch_size={batch_size}')
logger.info(f'num_training_updates={num_training_updates}')
logger.info(f'num_hiddens={num_hiddens}')
logger.info(f'num_residual_hiddens={num_residual_hiddens}')
logger.info(f'embedding_dim={embedding_dim}')
logger.info(f'num_embeddings={num_embeddings}')
logger.info(f'definition_len={definition_len}')
logger.info(f'commitment_cost={commitment_cost}')
logger.info(f'decay={decay}')
logger.info(f'learning_rate={learning_rate}')
# logger.info(f'cosine_similarity_eps={cosine_similarity_eps}')
logger.info('')
# %%
training_loader = DataLoader(training_data,
                             batch_size=batch_size,
                             shuffle=True,
                             pin_memory=True)
# %%
validation_loader = DataLoader(validation_data,
                               batch_size=32,
                               shuffle=True,
                               pin_memory=True)


# %%
class Model(nn.Module):
    def __init__(self, num_hiddens, definition_len, num_residual_layers, num_residual_hiddens,
                 num_embeddings, embedding_dim, commitment_cost, decay=0):
        super(Model, self).__init__()

        self._encoder = Encoder(num_hiddens,
                                num_residual_layers,
                                num_residual_hiddens,
                                embedding_dim)
        self._pre_vq_lin = nn.Linear(in_features=num_hiddens,
                                     out_features=embedding_dim * definition_len)
        if decay > 0.0:
            self._vq_vae = VectorQuantizerEMA(num_embeddings, embedding_dim,
                                              commitment_cost, decay)
        else:
            self._vq_vae = VectorQuantizer(num_embeddings, embedding_dim,
                                           commitment_cost)
        self._decoder = Decoder(embedding_dim, definition_len)

    def forward(self, x):
        z = self._encoder(x)
        z = self._pre_vq_lin(z)
        loss, quantized, perplexity, _ = self._vq_vae(z)
        x_recon, x_definition = self._decoder(quantized)

        return loss, x_recon, x_definition, perplexity


# %%
model = Model(num_hiddens, definition_len, num_residual_layers, num_residual_hiddens,
              num_embeddings, embedding_dim,
              commitment_cost, decay).to(device)

logger.info(f'model=\n{model}')
# %%
optimizer = optim.Adam(model.parameters(), lr=learning_rate, amsgrad=False)
# %%

# %%
model.train()
train_res_recon_error = []
train_vq_loss = []
train_res_perplexity = []
min_recon_error = 1000

for i in xrange(num_training_updates):
    data = next(iter(training_loader))
    data = data.to(device)
    optimizer.zero_grad()

    vq_loss, data_recon, data_definition, perplexity = model(data)
    recon_error = F.mse_loss(data_recon, data)  # / data_variance
    loss = recon_error + vq_loss
    loss.backward()

    optimizer.step()

    train_res_recon_error.append(recon_error.item())
    train_vq_loss.append(vq_loss.item())
    train_res_perplexity.append(perplexity.item())

    if (i + 1) % 100 == 0:
        if recon_error < min_recon_error:
            min_recon_error = recon_error
            torch.save(model, min_loss_model_path)
        logger.debug('%d iterations' % (i + 1))
        recon_error = np.mean(train_res_recon_error[-100:])
        logger.debug('recon_error: %.3f' % recon_error)
        logger.debug('vq_loss: %.3f' % np.mean(train_vq_loss[-100:]))
        logger.debug('perplexity: %.3f' % np.mean(train_res_perplexity[-100:]))
        logger.debug('')

# Write final result
if recon_error < min_recon_error:
    min_recon_error = recon_error

torch.save(model, final_model_path)
logger.info('%d iterations' % (i + 1))
recon_error = np.mean(train_res_recon_error[-100:])
logger.info('recon_error: %.3f' % recon_error)
logger.info('min_recon_error: %.3f' % min_recon_error)
logger.info('vq_loss: %.3f' % np.mean(train_vq_loss[-100:]))
logger.info('perplexity: %.3f' % np.mean(train_res_perplexity[-100:]))
logger.info('')
# %%
model = torch.load(min_loss_model_path)
# %% md
## Plot Loss
# %%
train_res_recon_error_smooth = savgol_filter(train_res_recon_error, 201, 7)
train_res_vq_loss_smooth = savgol_filter(train_vq_loss, 201, 7)
train_res_perplexity_smooth = savgol_filter(train_res_perplexity, 201, 7)
# %%
f = plt.figure(figsize=(16, 8))
ax = f.add_subplot(2, 2, 1)
ax.plot(train_res_recon_error_smooth)
ax.set_yscale('log')
# ax.set_title('Smoothed cosine embedding loss.')
ax.set_xlabel('iteration')

ax = f.add_subplot(2, 2, 2)
ax.plot(train_res_vq_loss_smooth)
ax.set_yscale('log')
ax.set_title('Smoothed VQ loss.')
ax.set_xlabel('iteration')

ax = f.add_subplot(2, 2, 3)
ax.plot(train_res_perplexity_smooth)
ax.set_title('Smoothed Average codebook usage (perplexity).')
ax.set_xlabel('iteration')

plt.savefig(f'{run_dir}/loss.png')
# %% md
## Auxilary functions
# %%
cosine_similarity = nn.CosineSimilarity(dim=0, eps=cosine_similarity_eps)


def closest_vector(v):
    i = np.argmin([F.pairwise_distance(v, torch.tensor(w).detach().cpu()) for w in all_data])
    return all_data[i]


def closest_word(v):
    i = np.argmin([F.pairwise_distance(v, torch.tensor(w).detach().cpu()) for w in all_data])
    return all_words[i]


def tensor_to_closest_word(t):
    return closest_word(t)


def tensor_to_embedding_closest_word(v):
    i = np.argmin([F.pairwise_distance(v, w) for w in model._vq_vae._embedding.weight.detach().cpu()])
    return tensor_to_closest_word(model._vq_vae._embedding.weight[i].detach().cpu())


# %% md
## View Reconstructions
# %%
model.eval()

valid_originals = next(iter(validation_loader))
valid_originals = valid_originals.to(device)

vq_output_eval = model._pre_vq_lin(model._encoder(valid_originals))
_, valid_quantize, _, _ = model._vq_vae(vq_output_eval)
valid_reconstructions, valid_definition = model._decoder(valid_quantize)
logger.info('validation')
for i in range(10):
    orig_word = tensor_to_closest_word(valid_originals[i].detach().cpu())
    recon_word = tensor_to_closest_word(valid_reconstructions[i].detach().cpu())
    def_text = ' '.join([tensor_to_embedding_closest_word(v.detach().cpu()) for v in valid_definition[i]])
    logger.info(f'{orig_word} to {recon_word}: {def_text}')
# %%
train_originals = next(iter(training_loader))
train_originals = train_originals.to(device)

vq_output_eval = model._pre_vq_lin(model._encoder(train_originals))
_, train_quantize, _, _ = model._vq_vae(vq_output_eval)
train_reconstructions, train_definition = model._decoder(train_quantize)
logger.info('train')
for i in range(10):
    orig_word = tensor_to_closest_word(train_originals[i].detach().cpu())
    recon_word = tensor_to_closest_word(train_reconstructions[i].detach().cpu())
    def_text = ' '.join([tensor_to_embedding_closest_word(v.detach().cpu()) for v in train_definition[i]])
    logger.info(f'{orig_word} to {recon_word}: {def_text}')
# %% md
## View Embedding
# %%
proj = umap.UMAP(n_neighbors=3,
                 min_dist=0.1,
                 metric='cosine').fit_transform(model._vq_vae._embedding.weight.data.cpu())
# %%
plt.scatter(proj[:, 0], proj[:, 1], alpha=0.3)
plt.savefig(f'{run_dir}/embedding.png')
# %%
for i in range(20):
    logger.debug(
        f'{i}:\t{tensor_to_closest_word(model._vq_vae._embedding.weight.data[i].detach().cpu())} {model._vq_vae._embedding.weight.data[i].tolist()}')
# %%
logging.shutdown()