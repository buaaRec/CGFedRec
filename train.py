import datetime
import os
from data import SampleGenerator
import warnings
warnings.filterwarnings('ignore')
from config import *
from read_data import load_data
from early_stop import *
from utils import *

# Logging.
path = 'log/'
current_time = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
logname = os.path.join(path, current_time+'.txt')
initLogging(logname)
rating = load_data(config['dataset'])

logging.info('Range of userId is [{}, {}]'.format(rating.userId.min(), rating.userId.max()))
logging.info('Range of itemId is [{}, {}]'.format(rating.itemId.min(), rating.itemId.max()))
logging.info(str(args))

# DataLoader for training
sample_generator = SampleGenerator(ratings=rating, neg_num=config['neg_num'], evaluate=config['evaluate'])
validate_data = sample_generator.validate_data
test_data = sample_generator.test_data

engine = get_engine(config['alias'])(config)

early_stopper = EarlyStopping(mode="max", patience=10)

train_result = []

for round in range(config['num_round']):
    tmp = {}
    logging.info('-' * 80)
    logging.info('Round {} starts !'.format(round))

    all_train_data = sample_generator.store_all_train_data(config['num_negative'])
    num_inter = sample_generator.num_inter

    logging.info('-' * 80)
    logging.info('Training phase!')
    tr_re = engine.fed_train_a_round(all_train_data,round_id=round)
    tmp['train'] = tr_re
    logging.info(f'[Training Epoch {round} Loss = {sum(tr_re["loss"].values())/len(tr_re["loss"].keys())}]')

    logging.info('-' * 80)
    test_re = engine.fed_evaluate(test_data)
    tmp['test'] = test_re

    hr_str = ', '.join([f'{v:.8f}' for i, v in enumerate(test_re['hr'])])
    ndcg_str = ', '.join([f'{v:.8f}' for i, v in enumerate(test_re['ndcg'])])
    logging.info(f'[Testing Epoch {round}] HR = {{{hr_str}}}, NDCG = {{{ndcg_str}}}')

    logging.info('-' * 80)
    logging.info('Validating phase!')
    val_re = engine.fed_evaluate(validate_data)
    tmp['val'] = val_re

    train_result.append(tmp)

    hr_str = ', '.join([f'{v:.8f}' for i, v in enumerate(val_re['hr'])])
    ndcg_str = ', '.join([f'{v:.8f}' for i, v in enumerate(val_re['ndcg'])])
    logging.info(f'[Validating Epoch {round}] HR = {{{hr_str}}}, NDCG = {{{ndcg_str}}}')

    model_params = [engine.client_model_params, engine.server_model_param]
    if early_stopper.step(val_re['hr'][0], test_re, model_params):
        print("Early stopping triggered")
        break

# current_time = datetime.datetime.now().strftime('%Y-%m-%d %H-%M-%S')
if config['save']:
    path = f'./checkpoint/'+current_time+ '-' + config['alias'] + '-' + str(config['dataset'])+'-params.npy'
    early_stopper.save(path)
    path = f'./metric/' + current_time + '-' + config['alias'] + '-' + str(config['dataset']) + '-metric.npy'
    np.save(path, np.array(train_result))

log_str = (
    f"{current_time}-"
    f"{config['alias']}-"
    f"dataset:{config['dataset']}-"
    f"lr:{config['lr']}-"
    f"optimizer:{config['optimizer']}-"
    f"batch_size:{config['batch_size']}-"
    f"local_epoch:{config['local_epoch']}-"
    f"num_round:{config['num_round']}-"
    f"latent_dim:{config['latent_dim']}-"
    f"num_negative:{config['num_negative']}-"
    f"neg_num:{config['neg_num']}-"
    f"clients_sample_ratio:{config['clients_sample_ratio']}-"
    f"clients_sample_num:{config['clients_sample_num']}-"
    f"l2:{config['l2_regularization']}-"
    f"seed:{config['seed']}-"
    f"eval:{config['evaluate']}-"
    f"bit:{config['bit']}-"
    f"hr:{early_stopper.best_result['hr']}-"
    f"ndcg:{early_stopper.best_result['ndcg']}"
)
file_name = "sh_result/"+config['alias']+'-'+config['dataset']+".txt"
with open(file_name, 'a') as file:
    file.write(log_str + '\n')

