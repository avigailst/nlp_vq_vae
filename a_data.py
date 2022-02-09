import torch
import ssl
import numpy as np
import wandb
import nltk
import gensim.downloader as api
from torch.utils.data import Dataset, DataLoader



ssl._create_default_https_context = ssl._create_unverified_context

model_name = 'glove-wiki-gigaword-50'
wv = api.load(model_name)

nltk.download('averaged_perceptron_tagger')

class WordEmbeddingDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

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


wandb.init(project="nlp_vq_vae", entity="avigailst")

wandb.config = {
    "batch_size" : 1000,
    "num_training_updates" : 100000,
    "num_hiddens" : 128,
    "num_residual_hiddens" : 32,
    "num_residual_layers" : 2,
    "embedding_dim" : 50,
    "num_embeddings" : training_data_len // 10,
    "definition_len" : 5,
    "commitment_cost" : 0.1,
    "decay" : 0.999,
    "learning_rate" : 1e-6
}

data_std = torch.tensor(np.std(training_data.data, 0))
data_mean = torch.tensor(np.mean(training_data.data, 0))
data_norms = np.linalg.norm(training_data.data, axis=1)
data_variance = np.var(
    (data_norms - np.min(data_norms)) /
    (np.max(data_norms) - np.min(data_norms))
)

training_loader = DataLoader(training_data,
                             batch_size=1, #wandb.config["batch_size"],
                             shuffle=True,
                             pin_memory=True)

validation_loader = DataLoader(validation_data,
                               batch_size=32,
                               shuffle=True,
                               pin_memory=True)

