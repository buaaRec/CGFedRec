import torch.optim

from utils import *
import random
import copy
from engine.base import BaseEngine
from model.fcf import *


class PFedRecEngine(BaseEngine):
    """Meta Engine for training & evaluating NCF model

    Note: Subclass should implement self.model !
    """

    def __init__(self, config):
        super(PFedRecEngine, self).__init__(config)
        self.model = FCF(config)
        if config['use_cuda'] is True:
            self.model.cuda()

    def fed_train_single_batch(self, model_client, batch_data, optimizers):
        """train a batch and return an updated model."""
        # load batch data.
        _, items, ratings = batch_data[0], batch_data[1], batch_data[2]
        ratings = ratings.float()

        if self.config['use_cuda'] is True:
            items, ratings = items.cuda(), ratings.cuda()

        optimizer, optimizer_i = optimizers
        # update score function.
        optimizer.zero_grad()
        ratings_pred = model_client(items)
        loss = self.crit(ratings_pred.view(-1), ratings)
        loss.backward()
        optimizer.step()

        # update item embedding.
        optimizer_i.zero_grad()
        ratings_pred = model_client(items)
        loss_i = self.crit(ratings_pred.view(-1), ratings)
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