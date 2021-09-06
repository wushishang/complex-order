# train.py
import copy
import random
import time

import torch.optim as optim
from torch import nn
from torch.backends import cudnn
from tqdm import tqdm
import numpy as np

from common.helper import log_stats, Util, print_stats
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

class Train:

    @classmethod
    def set_seed(cls, seed, cfg, logger_stats, multiple_GPU=False, freeze_cudnn=False):
        """
        Set seeds for reproducibility
        Ref: https://github.com/pytorch/pytorch/issues/7068#issuecomment-484918113

        Note that seeds would be reset if a model is saved and later loaded for continued training, which
        would lead to different results.
        """
        assert isinstance(seed, int)

        if seed >= 0:
            log_stats(logger_stats, "Random seed info", seed=cfg.seed_val)
            random.seed(seed)
            os.environ['PYTHONHASHSEED'] = str(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            if multiple_GPU:
                torch.cuda.manual_seed_all(seed)
            if freeze_cudnn:
                cudnn.benchmark = False
                cudnn.deterministic = True
                cudnn.enabled = False
        else:
            log_stats(logger_stats, "Random seed info: random seeds have NOT been set.")

    @classmethod
    def get_model(cls, cfg, dataset, logger_stats):
        model = namedclass[cfg.model_cfg.get_class_name()](cfg, len(dataset.vocab))
        n_all_param = sum([p.nelement() for p in model.parameters()])
        log_stats(logger_stats, 'Model specifications:', num_params=n_all_param)
        if torch.cuda.is_available():
            model.cuda()

        return model

    @classmethod
    def load_state(cls, cfg, model, optimizer, logger_stats):
        checkpoint = cfg.checkpoint_file_name()
        print_stats("Checkpoint filename:", checkpoint=checkpoint)
        if os.path.exists(checkpoint):
            # Properly load the checkpoint file saved from another device to the local device
            # see: https://github.com/pytorch/pytorch/issues/10622#issuecomment-474733769
            if torch.cuda.is_available():
                map_location = lambda storage, loc: storage.cuda()
            else:
                map_location = 'cpu'
            state_dict = torch.load(checkpoint, map_location=map_location)

            model.load_state_dict(state_dict['model'])
            if state_dict['best_model'] is not None:  # best model exists only when validation data are available
                best_model_dict = state_dict['best_model']
            else:
                best_model_dict = None
            optimizer.load_state_dict(state_dict['optimizer'])
            best_val_metric = state_dict['best_val_metric']
            # patience = state_dict['patience']
            # patience_increase = state_dict['patience_increase']
            epoch = state_dict['epoch']
            best_epoch = state_dict['best_epoch']
            training_time = state_dict['training_time']
            # if cfg.testing:
            #     assert patience - epoch <= 1, "Training must be finished before testing!"
            log_stats(logger_stats,  # "testing" if cfg.testing else "restarting_optimization",
                      best_val_metric=best_val_metric,  # patience=patience, patience_increase=patience_increase,
                      epoch=epoch, training_time=training_time)
        else:
            best_val_metric = 0.
            best_model_dict = None
            # patience = cfg.num_epochs  # look as these many epochs
            # patience_increase = cfg.patience_increase  # wait these many epochs longer once validation error stops reducing
            epoch = -1
            best_epoch = -1
            training_time = 0
            log_stats(logger_stats, "starting_optimization",
                      best_val_metric=best_val_metric,  # patience=patience, patience_increase=patience_increase,
                      epoch=epoch, training_time=0)
        return model, best_model_dict, optimizer, best_val_metric, epoch, best_epoch, training_time  # patience, patience_increase,

    @classmethod
    def dev_point_wise(cls, cfg, logger_stats, epoch_stats):
        # Set seeds
        cls.set_seed(cfg.seed_val, cfg, logger_stats)

        # Initialize data
        train_file = "../data/" + cfg.experiment_data.name
        dataset = Dataset(cfg)
        dataset.load_data(train_file, cfg.experiment_data.name)

        # Initialize model
        model = cls.get_model(cfg, dataset, logger_stats)
        optimizer = optim.Adam(model.parameters(), lr=cfg.learning_rate)
        NLLLoss = nn.CrossEntropyLoss()
        # model.add_optimizer(optimizer)
        model.add_loss_op(NLLLoss)
        best_model = copy.deepcopy(model)  # will load the final state at the end of training for model evaluation

        model, best_model_dict, optimizer, best_val_metric \
            , epoch, best_epoch, training_time = cls.load_state(cfg, model, optimizer, logger_stats)

        # train_losses = []
        # max_score = 0.1

        # TODO: 1. Save checkpoint for continued training and evaluation (finished)
        #       2. Add the original mode (fixed PE)
        #       3. Add regularization
        #       4. Add different orderings for testing
        #       5. Merge PE_reduce and wo

        log_stats(logger_stats, "---------Training Model---------", model=model, optimizer=optimizer)
        while cfg.num_epochs - epoch > 1:
            epoch = epoch + 1

            for epoch in tqdm(range(epoch, cfg.num_epochs), desc="Epochs"):
                start_time = time.time()
                # TODO: remove the unused 'val_accuracy'
                train_loss, val_accuracy = model.run_epoch(dataset.train_iterator, epoch, optimizer)
                epoch_time = time.time() - start_time
                training_time += epoch_time

                # train_losses.append(train_loss)
                train_acc = evaluate_model(model, dataset.train_iterator)
                val_acc = evaluate_model(model, dataset.val_iterator)
                _es = Util.dictize(epoch=epoch
                                   , loss_train=float(np.mean(np.array(train_loss)))
                                   , acc_train=train_acc
                                   , acc_val=val_acc)
                if isinstance(model, Transformer_PE_reduce):
                    _es['pe_variance'], _es['pe_norm'] = get_pe_variance(model.src_embed[1].pe.weight)

                if val_acc > best_val_metric:
                    best_val_metric = val_acc
                    best_epoch = epoch
                    if isinstance(model, Transformer_PE_reduce):
                        # FIXME: only track PE up to max_sen_len
                        _es['best_pe_variance'], _es['best_pe_norm'] = _es['pe_variance'], _es['pe_norm']
                    # Save Weights
                    best_model_dict = model.state_dict()
                    log_stats(logger_stats, "saving_model_checkpoint", epoch=epoch)
                _es.update(Util.dictize(best_val_metric=best_val_metric, best_epoch=best_epoch))

                epoch_stats.add(**_es)
                # Save state at the end of epoch
                torch.save(Util.dictize(
                    model=model.state_dict(), best_model=best_model_dict if best_model_dict is not None else None,
                    optimizer=optimizer.state_dict(), best_val_metric=best_val_metric,
                    epoch=epoch, best_epoch=best_epoch, training_time=training_time
                ), cfg.checkpoint_file_name())

        if best_model_dict is None:
            raise ValueError("Unexpected empty best_model_dict at the end of training.")
        best_model.load_state_dict(best_model_dict)

        return best_val_metric, best_model

    @classmethod
    def main(cls):
        cfg = Config()

        # Initialize loggers
        logger_stats = JsonDump(cfg.log_file_name())
        epoch_stats = JsonDump(cfg.stats_file_name())
        output_stats = JsonDump(cfg.output_file_name())

        if cfg.experiment_data in (TC_ExperimentData.TREC_transformer, TC_ExperimentData.sst2_transformer):
            _, best_model = cls.dev_point_wise(cfg, logger_stats, epoch_stats)
            train_file = "../data/" + cfg.experiment_data.name
            dataset = Dataset(cfg)
            dataset.load_data(train_file, cfg.experiment_data.name)
            test_acc = evaluate_model(best_model, dataset.test_iterator)
            print('Final Test Accuracy: {:.4f}'.format(test_acc))
            _es_test = {'test_acc': test_acc}
            if isinstance(best_model, Transformer_PE_reduce):
                _es_test['best_pe_variance'], _es_test['best_pe_norm'] = get_pe_variance(best_model.src_embed[1].pe.weight)
            output_stats.add(**_es_test)
        else:
            acc_flod = []
            for i in range(1, cfg.n_fold + 1):
                print("{} cross validation ".format(i))
                acc_flod.append(cls.dev_point_wise(cfg, logger_stats, epoch_stats)[0])
            print("the average acc {}".format(np.mean(acc_flod)))

if __name__ == '__main__':
    Train.main()