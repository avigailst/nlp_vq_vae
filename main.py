from __future__ import print_function
from sklearn.cluster import KMeans
import closest_word as cw
import vq_model as vm
from scipy.signal import savgol_filter
from six.moves import xrange
import wandb
import umap
import torch.nn.functional as F
import torch.optim as optim
import a_data as ad
from datetime import datetime
import os
import logging
import torch
import numpy as np
from kmeans_pytorch import kmeans, kmeans_predict
from matplotlib import pyplot as plt

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

def tensor_to_nth_embedding_closest_word(v, n):
    i = np.argmin([F.pairwise_distance(v, w) for w in model._vq_vae._embedding.weight.detach().cpu()])
    return cw.tensor_to_nth_closest_word(ad.all_words, ad.all_data, model._vq_vae._embedding.weight[i].detach().cpu(), n)


cosine_similarity_eps = 1e-10



logger.info(f'model_name={ad.model_name}')
logger.info(f'batch_size={wandb.config["batch_size"]}')
logger.info(f'num_training_updates={wandb.config["num_training_updates"]}')
logger.info(f'num_hiddens={wandb.config["num_hiddens"]}')
logger.info(f'num_residual_hiddens={wandb.config["num_residual_hiddens"]}')
logger.info(f'embedding_dim={wandb.config["embedding_dim"]}')
logger.info(f'num_embeddings={wandb.config["num_embeddings"]}')
logger.info(f'definition_len={wandb.config["definition_len"]}')
logger.info(f'commitment_cost={wandb.config["commitment_cost"]}')
logger.info(f'decay={wandb.config["decay"]}')
logger.info(f'learning_rate={wandb.config["learning_rate"]}')
# logger.info(f'cosine_similarity_eps={cosine_similarity_eps}')
logger.info('')
# %%



# %%



# %%
model = vm.Model(wandb.config["num_hiddens"], wandb.config["definition_len"],
              wandb.config["num_residual_layers"], wandb.config["num_residual_hiddens"],
              wandb.config["num_embeddings"], wandb.config["embedding_dim"],
              wandb.config["commitment_cost"], wandb.config["decay"]).to(device)

logger.info(f'model=\n{model}')
# %%
optimizer = optim.Adam(model.parameters(), lr=wandb.config["learning_rate"], amsgrad=False)
# %%
min_loss_model_path = './runs/2022-02-07-05-38-13/min-loss-model.pt'
model = torch.load(min_loss_model_path)
# %%
model.train()
train_res_recon_error = []
train_vq_loss = []
train_res_perplexity = []
min_recon_error = 1000

for i in xrange(wandb.config["num_training_updates"]):
    data = next(iter(ad.training_loader))
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

model.eval()

valid_originals = next(iter(ad.validation_loader))
valid_originals = valid_originals.to(device)
x = torch.from_numpy(ad.training_data.data)

# _, cluster_centers = kmeans(
#     X= x, num_clusters=ad, distance='euclidean', device=device
# )



vq_output_eval = model._pre_vq_lin(model._encoder(valid_originals))
_, valid_quantize, _, _ = model._vq_vae(vq_output_eval)
valid_reconstructions, valid_definition = model._decoder(valid_quantize)
logger.info('validation')
for i in range(10):
    orig_word = cw.tensor_to_nth_closest_word(ad.all_words, ad.all_data, valid_originals[i].detach().cpu(), 0)
    recon_word = cw.tensor_to_nth_closest_word(ad.all_words, ad.all_data, valid_reconstructions[i].detach().cpu(), 0)
    def_text = ' '.join([tensor_to_nth_embedding_closest_word(v.detach().cpu(), 0) for v in valid_definition[i]])
    y = valid_originals[i].data.unsqueeze(1)
    # k_mean = kmeans_predict(y, cluster_centers, 'euclidean', device=device)
    # k_mean_word = cw.tensor_to_nth_closest_word(ad.all_words, ad.all_data, k_mean, 0)
    logger.info(f'1- {orig_word} to {recon_word}: {def_text}')
    # logger.info(f'kmean- {k_mean_word}')

    recon_word = cw.tensor_to_nth_closest_word(ad.all_words, ad.all_data, valid_reconstructions[i].detach().cpu(), 1)
    def_text = ' '.join([tensor_to_nth_embedding_closest_word(v.detach().cpu(), 1) for v in valid_definition[i]])
    logger.info(f'2- {orig_word} to {recon_word}: {def_text}')


# %%
train_originals = next(iter(ad.training_loader))
train_originals = train_originals.to(device)


vq_output_eval = model._pre_vq_lin(model._encoder(train_originals))
_, train_quantize, _, _ = model._vq_vae(vq_output_eval)
train_reconstructions, train_definition = model._decoder(train_quantize)
logger.info('train')
for i in range(10):
    orig_word = cw.tensor_to_nth_closest_word(ad.all_words, ad.all_data, train_originals[i].detach().cpu(), 0)
    recon_word = cw.tensor_to_nth_closest_word(ad.all_words, ad.all_data, train_reconstructions[i].detach().cpu(), 0)
    def_text = ' '.join([tensor_to_nth_embedding_closest_word(v.detach().cpu(),0) for v in train_definition[i]])
    y = train_originals[i].data.unsqueeze(1)
    # k_mean = kmeans_predict(y, cluster_centers, 'euclidean',
    #                                device=device)
    # k_mean_word = cw.tensor_to_nth_closest_word(ad.all_words, ad.all_data, k_mean, 0)
    logger.info(f'1- {orig_word} to {recon_word}: {def_text}')

    # logger.info(f'kmean- {k_mean_word}')

    recon_word = cw.tensor_to_nth_closest_word(ad.all_words, ad.all_data, train_reconstructions[i].detach().cpu(), 1)
    def_text = ' '.join([tensor_to_nth_embedding_closest_word(v.detach().cpu(), 1) for v in train_definition[i]])
    logger.info(f'2- {orig_word} to {recon_word}: {def_text}')
    # logger.info(f'{vq_output_eval[i].detach().cpu().reshape(-1, embedding_dim)}')
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
# for i in range(20):
#     logger.debug(
#         f'{i}:\t{tensor_to_closest_word(model._vq_vae._embedding.weight.data[i].detach().cpu())} {model._vq_vae._embedding.weight.data[i].tolist()}')
# # %%

embedding_word_list = [cw.tensor_to_nth_closest_word(ad.all_words, ad.all_data,v.detach().cpu(), 0) for v in model._vq_vae._embedding.weight.data]
embedding_word_set = set(embedding_word_list)
logger.info(f'There are {len(embedding_word_set)}/{len(embedding_word_list)} ({len(embedding_word_set)*100./len(embedding_word_list)}%) unique words')

logging.shutdown()