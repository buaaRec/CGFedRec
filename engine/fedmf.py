from utils import *
import random
import copy
from engine.base import BaseEngine
from model.mf import *

class FedMFEngine(BaseEngine):
    """Meta Engine for training & evaluating NCF model

    Note: Subclass should implement self.model !
    """

    def __init__(self, config):
        super(FedMFEngine, self).__init__(config)
        self.model = MF(config)
        if config['use_cuda'] is True:
            self.model.cuda()