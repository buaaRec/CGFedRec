from abc import ABC, abstractmethod
from utils import *
from metrics import MetronAtK
import random
import copy
from data import UserItemRatingDataset
from torch.utils.data import DataLoader
from quantize import *

class BaseEngine(object):
    """Meta Engine for training & evaluating NCF model

    Note: Subclass should implement self.model !
    """

    def __init__(self, config):
        self.config = config  # model configuration
        self._metron = MetronAtK(top_k=config['top_k'])
        self.learner = config["optimizer"]
        self.weight_decay = config['l2_regularization']
        self.server_model_param = {}
        self.client_model_params = {}
        self.round_participant_params = {}
        self.participants = None
        # implicit feedback
        self.crit = torch.nn.BCELoss()

    def instance_user_train_loader(self, user_train_data):
        """instance a user's train loader."""
        dataset = UserItemRatingDataset(user_tensor=torch.LongTensor(user_train_data[0]),
                                        item_tensor=torch.LongTensor(user_train_data[1]),
                                        target_tensor=torch.FloatTensor(user_train_data[2]))
        return DataLoader(dataset, batch_size=self.config['batch_size'], shuffle=True)

    @abstractmethod
    def fed_train_single_batch(self, model_client, batch_data, optimizer):
        """train a batch and return an updated model."""
        users, items, ratings = batch_data[0], batch_data[1], batch_data[2]
        # x = torch.stack([users, items], dim=1)
        ratings = ratings.float()

        if self.config['use_cuda'] is True:
            users, items, ratings = users.cuda(), items.cuda(), ratings.cuda()

        # update score function.
        optimizer.zero_grad()
        ratings_pred = model_client(users, items)
        loss = self.crit(ratings_pred.view(-1), ratings)
        loss.backward()
        optimizer.step()

        return model_client, loss.item()

    def aggregate_clients_params(self):
        """receive client models' parameters in a round, aggregate them and store the aggregated result for server."""
        # aggregate item embedding and score function via averaged aggregation.
        t = 0
        for user in self.round_participant_params.keys():
            user_params = self.round_participant_params[user]
            if self.config['bit'] > 0:
                user_params = quantize_state_dict(self.round_participant_params[user], self.config['bit'])
            if t == 0:
                self.server_model_param = copy.deepcopy(user_params)
            else:
                for key in user_params.keys():
                    self.server_model_param[key].data += user_params[key].data
            t += 1
        for key in self.server_model_param.keys():
            self.server_model_param[key].data = self.server_model_param[key].data / t

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
            # 加载全局模型
            if self.config['bit'] > 0:
                self.server_model_param = quantize_state_dict(self.server_model_param, self.config['bit'])

            for key in self.server_model_param.keys():
                user_param_dict[key] = copy.deepcopy(self.server_model_param[key].data)

        model_client.load_state_dict(user_param_dict)
        return model_client

    def _set_optimizer(self, model):
        """初始化优化器，为不同参数设置不同学习率"""
        if self.learner.lower() == 'adam':
            optimizer = torch.optim.Adam(model.parameters(),lr=self.config['lr'], weight_decay=self.weight_decay)
        elif self.learner.lower() == 'sgd':
            optimizer = torch.optim.SGD(model.parameters(),lr=self.config['lr'], weight_decay=self.weight_decay)
        elif self.learner.lower() == 'adagrad':
            optimizer = torch.optim.Adagrad(model.parameters(),lr=self.config['lr'], weight_decay=self.weight_decay)
        elif self.learner.lower() == 'rmsprop':
            optimizer = torch.optim.RMSprop(model.parameters(),lr=self.config['lr'], weight_decay=self.weight_decay)
        else:
            self.logger.warning('未识别的优化器，使用默认的Adam优化器')
            optimizer = torch.optim.Adam(model.parameters(), lr=self.config['lr'], weight_decay=self.weight_decay)
        return optimizer

    def _upload_param(self, user, client_param):
        # Upload non-user parameters
        self.round_participant_params[user] = copy.deepcopy(client_param)
        self.client_model_params[user] = copy.deepcopy(client_param)
        for key in client_param.keys():
            self.client_model_params[user][key] = self.client_model_params[user][key].data.cpu()
            if 'user' in key:
                del self.round_participant_params[user][key]
            else:
                self.round_participant_params[user][key] = self.round_participant_params[user][key].data.cpu()

    def _model_train(self, model_client, user_dataloader, optimizer):
        model_client.train()
        sample_num = 0
        loss = 0
        # update client model.
        for epoch in range(self.config['local_epoch']):
            for batch_id, batch in enumerate(user_dataloader):
                assert isinstance(batch[0], torch.LongTensor)
                model_client, loss_r = self.fed_train_single_batch(model_client, batch, optimizer)
                loss += loss_r * len(batch[0])
                sample_num += len(batch[0])
        return loss / sample_num

    def _client_sample(self):
        if self.config['clients_sample_ratio'] <= 1:
            num_participants = int(self.config['num_users'] * self.config['clients_sample_ratio'])
            participants = random.sample(range(self.config['num_users']), num_participants)
        else:
            participants = random.sample(range(self.config['num_users']), self.config['clients_sample_num'])
        return participants

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
        result['loss'] = all_loss
        return result

    def fed_evaluate(self, evaluate_data):
        # evaluate all client models' performance using testing data.
        participants = sorted(list(self.client_model_params.keys()))
        test_users = evaluate_data[0][torch.isin(evaluate_data[0], torch.tensor(participants))]
        test_items = evaluate_data[1][torch.isin(evaluate_data[0], torch.tensor(participants))]
        negative_users = evaluate_data[2][torch.isin(evaluate_data[2], torch.tensor(participants))]
        negative_items = evaluate_data[3][torch.isin(evaluate_data[2], torch.tensor(participants))]


        if self.config['use_cuda'] is True:
            test_users = test_users.cuda()
            test_items = test_items.cuda()
            negative_users = negative_users.cuda()
            negative_items = negative_items.cuda()
        test_scores = None
        negative_scores = None
        all_loss = {}
        for idx in range(len(participants)):
            user_model = self._set_client(participants[idx])
            user_model.eval()
            with torch.no_grad():
                test_user = test_users[idx: idx + 1]
                test_item = test_items[idx: idx + 1]
                negative_user = negative_users[torch.where(negative_users == test_user)]
                negative_item = negative_items[torch.where(negative_users == test_user)]
                test_score = user_model.forward_test(test_user, test_item)
                negative_score = user_model.forward_test(negative_user, negative_item)
                if test_scores is None:
                    test_scores = test_score
                    negative_scores = negative_score
                else:
                    test_scores = torch.cat((test_scores, test_score))
                    negative_scores = torch.cat((negative_scores, negative_score))
                ratings = torch.zeros(len(negative_user) + 1)
                ratings[0] = 1
                if self.config['use_cuda'] is True:
                    ratings = ratings.cuda()
                ratings_pred = torch.cat((test_score, negative_score))
                loss = self.crit(ratings_pred.view(-1), ratings)
            all_loss[participants[idx]] = loss.item()
        if self.config['use_cuda'] is True:
            test_users = test_users.cpu()
            test_items = test_items.cpu()
            test_scores = test_scores.cpu()
            negative_users = negative_users.cpu()
            negative_items = negative_items.cpu()
            negative_scores = negative_scores.cpu()
        self._metron.subjects = [test_users.data.view(-1).tolist(),
                                 test_items.data.view(-1).tolist(),
                                 test_scores.data.view(-1).tolist(),
                                 negative_users.data.view(-1).tolist(),
                                 negative_items.data.view(-1).tolist(),
                                 negative_scores.data.view(-1).tolist()]
        hit_ratio, ndcg, recall, precision = self._metron.cal_hit_ratio(), self._metron.cal_ndcg(), self._metron.cal_recall(), self._metron.cal_precision()
        re = {'hr': hit_ratio, 'ndcg': ndcg, 'precision': precision, 'loss': all_loss}
        return re