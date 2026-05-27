import torch.optim

from utils import *
import random
import copy
from engine.base import BaseEngine
from model.cgscf import *
from sklearn.cluster import KMeans

class CGFedRecEngine(BaseEngine):
    """Meta Engine for training & evaluating NCF model

    Note: Subclass should implement self.model !
    """

    def __init__(self, config):
        super(CGFedRecEngine, self).__init__(config)
        self.model = CGSCF(config)
        if config['use_cuda'] is True:
            self.model.cuda()
        self.labels = torch.tensor([1, 2])
        self.reg = config['reg']
        self.cluster_history = {}
        self.kmeans = KMeans(n_clusters=self.config['item_cluster'])

    def fed_train_single_batch(self, model_client, batch_data, optimizers):
        """train a batch and return an updated model."""
        # load batch data.
        _, items, ratings = batch_data[0], batch_data[1], batch_data[2]
        ratings = ratings.float()

        if self.config['use_cuda'] is True:
            items, ratings = items.cuda(), ratings.cuda()
            labels = self.labels.cuda()

        optimizer, optimizer_i = optimizers
        # update score function.
        optimizer.zero_grad()
        ratings_pred, supcon = model_client(items, self.labels)

        loss = self.crit(ratings_pred.view(-1), ratings)
        loss = loss + self.reg * supcon
        loss.backward()
        optimizer.step()

        # update item embedding.
        optimizer_i.zero_grad()
        ratings_pred, supcon = model_client(items, self.labels)

        loss_i = self.crit(ratings_pred.view(-1), ratings)
        loss_i = loss_i + self.reg * supcon
        loss_i.backward()
        optimizer_i.step()

        return model_client, loss_i.item()

    def _upload_param(self, user, client_param):
        # Upload non-user parameters
        self.client_model_params[user] = copy.deepcopy(client_param)
        self.round_participant_params[user] = copy.deepcopy(client_param)
        for key in client_param.keys():
            self.client_model_params[user][key] = self.client_model_params[user][key].data.cpu()
            if 'item' not in key:
                del self.round_participant_params[user][key]
            else:
                self.round_participant_params[user][key] = self.round_participant_params[user][key].data.cpu()

    def _set_optimizer(self, model):
        if self.learner.lower() == 'adam':
            optimizer = torch.optim.Adam(model.affine_output.parameters(),
                                        lr=self.config['lr'],
                                        weight_decay=self.config['l2_regularization'])  # MLP optimizer
            # optimizer_i is responsible for updating item embedding.
            optimizer_i = torch.optim.Adam(model.embedding_item.parameters(),
                                          lr=self.config['lr'] * self.config['num_items'] * self.config['lr_eta'],
                                          weight_decay=self.config['l2_regularization'])  # Item optimizer
        elif self.learner.lower() == 'sgd':
            optimizer = torch.optim.SGD(model.affine_output.parameters(),
                                         lr=self.config['lr'],
                                         weight_decay=self.config['l2_regularization'])  # MLP optimizer
            # optimizer_i is responsible for updating item embedding.
            optimizer_i = torch.optim.SGD(model.embedding_item.parameters(),
                                           lr=self.config['lr'] * self.config['num_items'] * self.config['lr_eta'],
                                           weight_decay=self.config['l2_regularization'])  # Item optimizer
        elif self.learner.lower() == 'adagrad':
            optimizer = torch.optim.Adagrad(model.affine_output.parameters(),
                                        lr=self.config['lr'],
                                        weight_decay=self.config['l2_regularization'])  # MLP optimizer
            # optimizer_i is responsible for updating item embedding.
            optimizer_i = torch.optim.Adagrad(model.embedding_item.parameters(),
                                          lr=self.config['lr'] * self.config['num_items'] * self.config['lr_eta'],
                                          weight_decay=self.config['l2_regularization'])
        elif self.learner.lower() == 'rmsprop':
            optimizer = torch.optim.RMSprop(model.affine_output.parameters(),
                                        lr=self.config['lr'],
                                        weight_decay=self.config['l2_regularization'])  # MLP optimizer
            # optimizer_i is responsible for updating item embedding.
            optimizer_i = torch.optim.RMSprop(model.embedding_item.parameters(),
                                          lr=self.config['lr'] * self.config['num_items'] * self.config['lr_eta'],
                                          weight_decay=self.config['l2_regularization'])  # Item optimizer
        else:
            self.logger.warning('未识别的优化器，使用默认的Adam优化器')
            optimizer = torch.optim.Adam(model.affine_output.parameters(),
                                         lr=self.config['lr'],
                                         weight_decay=self.config['l2_regularization'])  # MLP optimizer
            # optimizer_i is responsible for updating item embedding.
            optimizer_i = torch.optim.Adam(model.embedding_item.parameters(),
                                           lr=self.config['lr'] * self.config['num_items'] * self.config['lr_eta'],
                                           weight_decay=self.config['l2_regularization'])  # Item optimizer

        return [optimizer, optimizer_i]

    def _set_client(self, user):
        model_client = copy.deepcopy(self.model)
        # for the first round, client models copy initialized parameters directly.
        # for other rounds, client models receive updated item embedding and score function from server.
        user_param_dict = copy.deepcopy(self.model.state_dict())
        # 加载模型
        if user in self.client_model_params.keys():
            # 加载本地模型
            for key in self.client_model_params[user].keys():
                user_param_dict[key] = copy.deepcopy(self.client_model_params[user][key].data)

        model_client.load_state_dict(user_param_dict)
        return model_client

    def fed_train_a_round(self, all_train_data, round_id):
        """train a round."""
        # sample users participating in single round.
        if self.config['clients_sample_method'] == 'random':
            self.participants = self._client_sample()
        else:
            if round_id == 0:
                self.participants = self._client_sample()

        # store all the users' train loss
        all_loss = {}
        result = {}
        for user in tqdm(self.participants):
            model_client = self._set_client(user)
            optimizer = self._set_optimizer(model_client)
            user_train_data = [all_train_data[0][user], all_train_data[1][user], all_train_data[2][user]]
            user_dataloader = self.instance_user_train_loader(user_train_data)
            all_loss[user] = self._model_train(model_client, user_dataloader, optimizer)
            self._upload_param(user, model_client.state_dict())

        self.aggregate_clients_params()

        self.item_clusters = self.kmeans.fit_predict(self.server_model_param['embedding_item.weight'].data)
        # self.labels=torch.randint(low=0, high=k_opt, size=(round_participant_params[0]['embedding_item.weight'].shape[0],))
        self.labels = copy.deepcopy(torch.tensor(self.item_clusters, dtype=torch.long))

        result['loss'] = all_loss
        result['item_cluster'] = self.item_clusters
        return result