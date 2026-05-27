import torch
import torch.nn as nn

class MF(torch.nn.Module):
    def __init__(self, config):
        super(MF, self).__init__()
        self.config = config

        self.num_users = config['num_users']
        self.num_items = config['num_items']
        self.latent_dim = config['latent_dim']
        self.__init_weight(config)
        self.logistic = torch.nn.Sigmoid()

    def __init_weight(self, config):

        self.embedding_user = torch.nn.Embedding(num_embeddings=self.num_users, embedding_dim=self.latent_dim)
        self.embedding_item = torch.nn.Embedding(num_embeddings=self.num_items, embedding_dim=self.latent_dim)
        nn.init.normal_(self.embedding_user.weight, std=0.1)
        nn.init.normal_(self.embedding_item.weight, std=0.1)

    def forward(self, user_id, item_id):
        users_emb = self.embedding_user(user_id)
        items_emb = self.embedding_item(item_id)
        logits = (users_emb * items_emb).sum(dim=1)
        rating = self.logistic(logits)
        return rating
        #
        # neg_emb = self.embedding_item(neg)
        # # reg_loss = (1/2) * (users_emb.norm(2).pow(2) + pos_emb.norm(2).pow(2) + neg_emb.norm(2).pow(2)) / float(len(users))
        # pos_scores = torch.matmul(user_emb, pos_emb.T)
        # pos_label = torch.ones_like(pos_scores)
        # neg_scores = torch.mul(user_emb, neg_emb.T)
        # neg_label = torch.zeros_like(neg_scores)
        # loss = 1/2 * (self.MSEloss(pos_scores, pos_label) + self.MSEloss(neg_scores, neg_label))
        #
        # return loss

    def forward_test(self, user_id, item_id):
        return self.forward(user_id, item_id)

    def getUsersRating(self, users, items):
        users_emb = self.embedding_user.weight[users]
        items_emb = self.embedding_item.weight[items]
        rating = self.f(torch.matmul(users_emb, items_emb.t()))
        return rating

    def getItemSimilarity(self):
        items_emb = self.embedding_item.weight
        similarity_matrix  = torch.mm(items_emb, items_emb.t())

        return similarity_matrix
