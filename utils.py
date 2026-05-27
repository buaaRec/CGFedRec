"""
    Some handy functions for pytroch model training ...
"""
import copy

import numpy as np
import torch
import logging
import random
import importlib
from sklearn.metrics import pairwise_distances
from sklearn.cluster import SpectralClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from tqdm import tqdm

import torch.nn.functional as F

# Checkpoints
def save_checkpoint(model, model_dir):
    torch.save(model.state_dict(), model_dir)

def get_engine(model_name):
    """根据模型名称自动选择模型类

    Args:
        model_name (str): 模型名称

    Returns:
        Recommender: 模型类

    Raises:
        ImportError: 当模块未找到时
        AttributeError: 当模型类未找到时
    """
    try:
        model_file_name = model_name.lower()
        module_path = ".".join(["engine", model_file_name])
        if importlib.util.find_spec(module_path, __name__):
            model_module = importlib.import_module(module_path, __name__)
            model_class = getattr(model_module, f"{model_name}Engine")
            return model_class
        raise ImportError(f"Module {module_path} not found")
    except (ImportError, AttributeError) as e:
        raise type(e)(f"Failed to load model {model_name}: {str(e)}")

def construct_user_relation_graph_via_item(round_user_params, item_num, latent_dim, similarity_metric):
    # prepare the item embedding array.
    item_embedding = np.zeros((len(round_user_params), item_num * latent_dim), dtype='float32')
    for user in round_user_params.keys():
        item_embedding[user] = round_user_params[user]['embedding_item.weight'].numpy().flatten()
    # construct the user relation graph.
    adj = pairwise_distances(item_embedding, metric=similarity_metric)
    if similarity_metric == 'cosine':
        return adj
    else:
        return -adj


def select_topk_neighboehood(user_realtion_graph, neighborhood_size, neighborhood_threshold):
    topk_user_relation_graph = np.zeros(user_realtion_graph.shape, dtype='float32')
    if neighborhood_size > 0:
        for user in range(user_realtion_graph.shape[0]):
            user_neighborhood = user_realtion_graph[user]
            topk_indexes = user_neighborhood.argsort()[-neighborhood_size:][::-1]
            for i in topk_indexes:
                topk_user_relation_graph[user][i] = 1/neighborhood_size
    else:
        similarity_threshold = np.mean(user_realtion_graph)*neighborhood_threshold
        for i in range(user_realtion_graph.shape[0]):
            high_num = np.sum(user_realtion_graph[i] > similarity_threshold)
            if high_num > 0:
                for j in range(user_realtion_graph.shape[1]):
                    if user_realtion_graph[i][j] > similarity_threshold:
                        topk_user_relation_graph[i][j] = 1/high_num
            else:
                topk_user_relation_graph[i][i] = 1

    return topk_user_relation_graph


def MP_on_graph(round_user_params, item_num, latent_dim, topk_user_relation_graph, layers):
    # prepare the item embedding array.
    item_embedding = np.zeros((len(round_user_params), item_num*latent_dim), dtype='float32')
    for user in round_user_params.keys():
        item_embedding[user] = round_user_params[user]['embedding_item.weight'].numpy().flatten()

    # aggregate item embedding via message passing.
    aggregated_item_embedding = np.matmul(topk_user_relation_graph, item_embedding)
    for layer in range(layers-1):
        aggregated_item_embedding = np.matmul(topk_user_relation_graph, aggregated_item_embedding)

    # reconstruct item embedding.
    item_embedding_dict = {}
    for user in round_user_params.keys():
        item_embedding_dict[user] = torch.from_numpy(aggregated_item_embedding[user].reshape(item_num, latent_dim))
    item_embedding_dict['global'] = sum(item_embedding_dict.values())/len(round_user_params)
    return item_embedding_dict

def compute_regularization(model, parameter_label):
    reg_fn = torch.nn.MSELoss(reduction='mean')
    for name, param in model.named_parameters():
        if name == 'embedding_item.weight':
            reg_loss = reg_fn(param, parameter_label)
            return reg_loss


def resume_checkpoint(model, model_dir, device_id):
    state_dict = torch.load(model_dir,
                            map_location=lambda storage, loc: storage.cuda(device=device_id))  # ensure all storage are on gpu
    model.load_state_dict(state_dict)

def n_bit_quantize(emb, n_bits=4, range=0):
    """
    将 n*m 的 tensor 转化为 n-bit 量化版本
    :param tensor: 输入的浮点型 torch.Tensor
    :param n_bits: 量化位数 (如 4-bit, 8-bit)
    :return: 量化后的整数张量, scale, zero_point
    """
    # 1. 确定量化范围 [0, 2^n - 1]
    q_min = 0
    q_max = 2 ** n_bits - 1

    # 2. 计算每一行的 min 和 max (为了更精细的量化，通常按行/通道进行)
    # 如果需要全局量化，去掉 dim=1 的参数即可
    if range == 0:
        t_min = emb.min(dim=1, keepdim=True)[0]
        t_max = emb.max(dim=1, keepdim=True)[0]
    else:
        t_min = emb.min(keepdim=True)[0]
        t_max = emb.max(keepdim=True)[0]
    # 3. 计算缩放因子 scale
    # 防止分母为 0
    scale = (t_max - t_min) / (q_max - q_min)
    scale = torch.clamp(scale, min=1e-8)

    # 4. 映射到整数空间并取整
    # x_q = round((x - min) / scale)
    quantized_tensor = torch.round((emb - t_min) / scale)

    # 5. 截断到目标位数范围并转为整数类型
    quantized_tensor = torch.clamp(quantized_tensor, q_min, q_max).to(torch.uint8)

    dequantized = quantized_tensor.float() * scale + t_min

    return dequantized

# Hyper params
def use_cuda(enabled, device_id=0):
    if enabled:
        assert torch.cuda.is_available(), 'CUDA is not available'
        torch.cuda.set_device(device_id)

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

    # def set_seeds(seed):
    #     np.random.seed(seed)
    #     torch.manual_seed(seed)
    #     if torch.cuda.is_available():
    #         torch.cuda.manual_seed(seed)
    #         torch.cuda.manual_seed_all(seed)
    #     torch.backends.cudnn.benchmark = False
    #     torch.backends.cudnn.deterministic = True
    #     torch.backends.cudnn.enabled = False


def initLogging(logFilename):
    """Init for logging
    """
    logging.basicConfig(
                    level    = logging.DEBUG,
                    format='%(asctime)s-%(levelname)s-%(message)s',
                    datefmt  = '%y-%m-%d %H:%M',
                    filename = logFilename,
                    filemode = 'w');
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s-%(levelname)s-%(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)