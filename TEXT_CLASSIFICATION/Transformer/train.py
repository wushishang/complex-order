# train.py
import random
import torch.optim as optim
from torch import nn
from tqdm import tqdm

from common.helper import log_stats, Util
from common.json_dump import JsonDump
from util.constants import TC_ExperimentData
from utils import *
from config import Config
# from static_config import Static_Config
from model_transformer.transformer_wo import Transformer_wo
from model_transformer.PE_reduce import Transformer_PE_reduce
from model_transformer.TPE_reduce import Transformer_TPE_reduce
from model_transformer.Complex_vanilla import Transformer_Complex_vanilla
from model_transformer.Complex_order import Transformer_Complex_order


namedclass = {'Transformer_wo': Transformer_wo, 'Transformer_PE_reduce': Transformer_PE_reduce,
              'Transformer_TPE_reduce': Transformer_TPE_reduce,
              'Transformer_Complex_vanilla': Transformer_Complex_vanilla,
              'Transformer_Complex_order':Transformer_Complex_order}

acc_flod=[]
cfg = Config()

# Initialize loggers
logger_stats = JsonDump(cfg.log_file_name())
epoch_stats = JsonDump(cfg.stats_file_name())
output_stats = JsonDump(cfg.output_file_name())


def set_seed(seed, logger_stats):
    """
    Set seeds for reproducibility

    Ref: https://github.com/pytorch/pytorch/issues/7068#issuecomment-484918113
    """
    assert isinstance(seed, int)

    if seed >= 0:
        log_stats(logger_stats, "Random seed info", seed=cfg.seed_val)
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
        # torch.backends.cudnn.benchmark = False
        # torch.backends.cudnn.deterministic = True
        # torch.backends.cudnn.enabled = False
    else:
        log_stats(logger_stats, "Random seed info: random seeds have NOT been set.")

def dev_point_wise():
    # Set seeds
    set_seed(cfg.seed_val, logger_stats)

    train_file = "../data/" + cfg.experiment_data.name
    dataset = Dataset(cfg)
    dataset.load_data(train_file, cfg.experiment_data.name)

    model = namedclass[cfg.model_cfg.get_class_name()](cfg, len(dataset.vocab))
    n_all_param = sum([p.nelement() for p in model.parameters()])
    print('#params = {}'.format(n_all_param))
    if torch.cuda.is_available():
        model.cuda()
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=cfg.learning_rate)
    NLLLoss = nn.CrossEntropyLoss()
    model.add_optimizer(optimizer)
    model.add_loss_op(NLLLoss)
    
    train_losses = []
    val_accuracies = []
    # max_score = 0.1
    acc_max = 0.0000
    best_epoch = 0

    log_stats(logger_stats, "---------Training Model---------", model=model, optimizer=optimizer)
    for epoch in tqdm(range(cfg.num_epochs), desc="Epochs"):
        train_loss,val_accuracy = model.run_epoch(dataset.train_iterator, dataset.val_iterator, epoch)
        train_losses.append(train_loss)
        # val_accuracies.append(val_accuracy)
        train_acc = evaluate_model(model, dataset.train_iterator)
        val_acc = evaluate_model(model, dataset.val_iterator)
        # print ('Final Training Accuracy: {:.4f}'.format(train_acc))
        # print ('Final Validation Accuracy: {:.4f}'.format(val_acc))

        _es = Util.dictize(epoch=epoch
                           , loss_train=float(np.mean(np.array(train_loss)))
                           , acc_train=train_acc
                           , acc_val=val_acc)
        if isinstance(model, Transformer_PE_reduce):
            _es['pe_variance'], _es['pe_norm'] = get_pe_variance(model.src_embed[1].pe.weight)

        if val_acc > acc_max:
            torch.save(model, cfg.checkpoint_file_name())
            acc_max = val_acc
            best_epoch = epoch
            if isinstance(model, Transformer_PE_reduce):
                _es['best_pe_variance'], _es['best_pe_norm'] = _es['pe_variance'], _es['pe_norm']
        _es.update(Util.dictize(best_val_metric=acc_max, best_epoch=best_epoch))

        epoch_stats.add(**_es)

    acc_flod.append(acc_max)


if __name__ == '__main__':
    if cfg.experiment_data in (TC_ExperimentData.TREC_transformer, TC_ExperimentData.sst2_transformer):
        dev_point_wise()
        train_file = "../data/" + cfg.experiment_data.name
        dataset = Dataset(cfg)
        dataset.load_data(train_file, cfg.experiment_data.name)
        model = torch.load(cfg.checkpoint_file_name())
        test_acc = evaluate_model(model, dataset.test_iterator)
        print ('Final Test Accuracy: {:.4f}'.format(test_acc))
        _es_test = {'test_acc':test_acc}
        if isinstance(model, Transformer_PE_reduce):
            _es_test['best_pe_variance'], _es_test['best_pe_norm'] = get_pe_variance(model.src_embed[1].pe.weight)
        output_stats.add(**_es_test)
    else:
        for i in range(1, cfg.n_fold + 1):
            print("{} cross validation ".format(i))
            dev_point_wise()
        print("the average acc {}".format(np.mean(acc_flod)))