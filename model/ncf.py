import torch


class NCF(torch.nn.Module):
    def __init__(self, config):
        super(NCF, self).__init__()
        self.config = config
        self.num_users = config['num_users']
        self.num_items = config['num_items']
        self.latent_dim = config['latent_dim']

        self.mlp_user_embeddings = torch.nn.Embedding(num_embeddings=self.num_users, embedding_dim=2*self.latent_dim )
        self.mlp_item_embeddings = torch.nn.Embedding(num_embeddings=self.num_items, embedding_dim=2*self.latent_dim )
        self.gmf_user_embeddings = torch.nn.Embedding(num_embeddings=self.num_users, embedding_dim=2*self.latent_dim )
        self.gmf_item_embeddings = torch.nn.Embedding(num_embeddings=self.num_items, embedding_dim=2*self.latent_dim )
        self.mlp = torch.nn.Sequential(torch.nn.Linear(4*self.latent_dim , 2*self.latent_dim ),
            torch.nn.ReLU(),
            torch.nn.Linear(2*self.latent_dim , self.latent_dim ),
            torch.nn.ReLU(),
            torch.nn.Linear(self.latent_dim , self.latent_dim //2),
            torch.nn.ReLU()
            )
        self.gmf_out = torch.nn.Linear(2*self.latent_dim , 1)
        self.gmf_out.weight = torch.nn.Parameter(torch.ones(1, 2*self.latent_dim ))
        self.mlp_out = torch.nn.Linear(self.latent_dim //2, 1)
        self.output_logits = torch.nn.Linear(self.latent_dim , 1)
        self.model_blending = 0.5           # alpha parameter, equation 13 in the paper
        self.initialize_weights()
        self.join_output_weights()
        self.logistic = torch.nn.Sigmoid()

    def initialize_weights(self):
        torch.nn.init.normal_(self.mlp_user_embeddings.weight, std=0.01)
        torch.nn.init.normal_(self.mlp_item_embeddings.weight, std=0.01)
        torch.nn.init.normal_(self.gmf_user_embeddings.weight, std=0.01)
        torch.nn.init.normal_(self.gmf_item_embeddings.weight, std=0.01)
        for layer in self.mlp:
            if isinstance(layer, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(layer.weight)
        torch.nn.init.kaiming_uniform_(self.gmf_out.weight, a=1)
        torch.nn.init.kaiming_uniform_(self.mlp_out.weight, a=1)

    def forward(self, user_id, item_id):
        gmf_product = self.gmf_forward(user_id, item_id)
        mlp_output = self.mlp_forward(user_id, item_id)
        logits = self.output_logits(torch.cat([gmf_product, mlp_output], dim=1)).view(-1)
        rating = self.logistic(logits)
        return rating

    def forward_test(self, user_id, item_id):
        return self.forward(user_id, item_id)

    def gmf_forward(self, user_id, item_id):
        user_emb = self.gmf_user_embeddings(user_id)
        item_emb = self.gmf_item_embeddings(item_id)
        return torch.mul(user_emb, item_emb)

    def mlp_forward(self, user_id, item_id):
        user_emb = self.mlp_user_embeddings(user_id)
        item_emb = self.mlp_item_embeddings(item_id)
        return self.mlp(torch.cat([user_emb, item_emb], dim=1))

    def join_output_weights(self):
        W = torch.nn.Parameter(torch.cat((self.model_blending*self.gmf_out.weight, (1-self.model_blending)*self.mlp_out.weight), dim=1))
        self.output_logits.weight = W

    def layer_setter(self, model, model_copy):
        for m, mc in zip(model.parameters(), model_copy.parameters()):
            mc.data[:] = m.data[:]

    def load_server_weights(self, server_model):
        self.layer_setter(server_model.mlp_item_embeddings, self.mlp_item_embeddings)
        self.layer_setter(server_model.gmf_item_embeddings, self.gmf_item_embeddings)
        self.layer_setter(server_model.mlp, self.mlp)
        self.layer_setter(server_model.gmf_out, self.gmf_out)
        self.layer_setter(server_model.mlp_out, self.mlp_out)
        self.layer_setter(server_model.output_logits, self.output_logits)
