import torch
import copy

class FCF(torch.nn.Module):
    def __init__(self, config):
        super(FCF, self).__init__()
        self.config = config
        self.num_items = config['num_items']
        self.latent_dim = config['latent_dim']

        self.embedding_item = torch.nn.Embedding(num_embeddings=self.num_items, embedding_dim=self.latent_dim)
        self.affine_output = torch.nn.Linear(in_features=self.latent_dim, out_features=1)
        self.logistic = torch.nn.Sigmoid()

    def forward(self, item_indices):
        item_embedding = self.embedding_item(item_indices)
        logits = self.affine_output(item_embedding)
        rating = self.logistic(logits)
        return rating

    def forward_test(self, user_id, item_indices):
        return self.forward(item_indices)

    def init_weight(self):
        pass

    def load_pretrain_weights(self):
        pass

