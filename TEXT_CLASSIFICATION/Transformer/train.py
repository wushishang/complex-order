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
from regularizers import JpRegularizer
from util.constants import TC_ExperimentData, MaxSenLen, PE_Type, TaskType
from utils import *
from config import Config
# from static_config import Static_Config
# from model_transformer.transformer_wo import Transformer_wo
from model_transformer.PE_reduce import Transformer_PE_real
from model_transformer.TPE_reduce import Transformer_TPE_reduce
from model_transformer.Complex_vanilla import Transformer_Complex_vanilla
from model_transformer.Complex_order import Transformer_Complex_order


namedclass = {'Transformer_PE_real': Transformer_PE_real,
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
    def get_data(cls, cfg, logger_stats):
        log_stats(logger_stats, 'Loading data...')
        train_file = "../data/" + cfg.experiment_data.name
        dataset = Dataset(cfg)
        dataset.load_data(train_file, cfg.experiment_data.name)
        log_stats(logger_stats, 'Loaded successfully!')

        return dataset

    @classmethod
    def get_model(cls, cfg, dataset, logger_stats):
        log_stats(logger_stats, "Building model...")

        #
        # Initialize the regularizer
        #
        regularizer = JpRegularizer(cfg.regularization, cfg.input_type, total_permutations=cfg.num_reg_permutations,
                                    strength=cfg.regularization_strength, eps=cfg.regularization_eps,
                                    representations=cfg.regularization_representation, normed=cfg.r_normed,
                                    power=cfg.r_power, tangent_prop=cfg.tangent_prop, tp_config=cfg.tp_config,
                                    fixed_edge=cfg.fixed_edge, random_segment=cfg.random_segment,
                                    permute_positional_encoding=cfg.permute_positional_encoding,
                                    task_type=None,  # if cfg.task_type == TaskType.node_classification
                                    agg_penalty_by_node=False)  # cfg.aggregate_penalty_by_node

        log_stats(logger_stats, "Regularization Info", regularizer=regularizer)
        # log_stats(logger_stats, "Scaling Info", scaling=cfg.scaling)

        model = namedclass[cfg.model_cfg.get_class_name()](cfg, len(dataset.vocab), MaxSenLen[cfg.experiment_data])
        n_all_param = sum([p.nelement() for p in model.parameters()])
        model.add_regularizer(regularizer)
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
            if state_dict['best_pe_variance'] is not None:
                best_pe_variance = state_dict['best_pe_variance']
            else:
                best_pe_variance = None
            if state_dict['best_pe_norm'] is not None:
                best_pe_norm = state_dict['best_pe_norm']
            else:
                best_pe_norm = None
            patience = state_dict['patience']
            patience_increase = state_dict['patience_increase']
            patience_reductions = state_dict['patience_reductions']
            epoch = state_dict['epoch']
            best_epoch = state_dict['best_epoch']
            training_time = state_dict['training_time']
            # if cfg.testing:
            #     assert patience - epoch <= 1, "Training must be finished before testing!"
            log_stats(logger_stats,  # "testing" if cfg.testing else "restarting_optimization",
                      best_val_metric=best_val_metric, patience=patience, patience_increase=patience_increase,
                      patience_reductions=patience_reductions, epoch=epoch, training_time=training_time)
        else:
            best_val_metric = 0.
            best_model_dict = None
            patience = cfg.num_epochs  # look as these many epochs
            patience_increase = cfg.patience_increase  # wait these many epochs longer once validation error stops reducing
            patience_reductions = 0
            best_pe_variance, best_pe_norm = None, None
            epoch = -1
            best_epoch = -1
            training_time = 0
            log_stats(logger_stats, "starting_optimization",
                      best_val_metric=best_val_metric, patience=patience, patience_increase=patience_increase,
                      patience_reductions=patience_reductions, epoch=epoch, training_time=0)
        return model, best_model_dict, optimizer, best_val_metric, patience, patience_increase, patience_reductions, \
               best_pe_variance, best_pe_norm, epoch, best_epoch, training_time

    @classmethod
    def dev_point_wise(cls, cfg, logger_stats, epoch_stats):
        # Set seeds
        cls.set_seed(cfg.seed_val, cfg, logger_stats)

        # Initialize data
        dataset = cls.get_data(cfg, logger_stats)

        # Initialize model
        model = cls.get_model(cfg, dataset, logger_stats)
        if cfg.original_mode:
            model.train()
        optimizer = optim.Adam(model.parameters(), lr=cfg.learning_rate)
        NLLLoss = nn.CrossEntropyLoss(reduction='none')
        # model.add_optimizer(optimizer)
        model.add_loss_op(NLLLoss)
        best_model = copy.deepcopy(model)  # will load the final state at the end of training for model evaluation

        model, best_model_dict, optimizer, best_val_metric, patience, patience_increase, patience_reductions, \
            best_pe_variance, best_pe_norm, epoch, best_epoch, training_time = cls.load_state(cfg, model, optimizer, logger_stats)

        # train_losses = []
        # max_score = 0.1

        # TODO: 1. Save checkpoint for continued training and evaluation (finished)
        #       2. Add the original mode (fixed PE, incorrect use of .train() and .eval()) (finished)
        #       3. Merge PE_reduce and wo (finished)
        #       4. Add regularization
        #       5. Add different orderings for testing
        #       6. Implement patience (finished)

        log_stats(logger_stats, "---------Training Model---------", model=model, optimizer=optimizer)
        while patience - epoch >= 1:
            # start from current value of epoch and run till patience
            # If during this period, patience increases, the outer loop takes care of it

            # If lr_reduction_limit is larger than 1, we will continue training with reduced learning rate when
            # patience runs up. We will train for 'cooldown' epochs first without monitoring change of evaluation metric,
            # then resume monitoring training with patience_increase.
            if patience - epoch == 1:
                if patience_reductions < cfg.lr_reduction_limit:
                    patience_reductions += 1
                    log_stats(logger_stats, "Reducing LR")
                    for g in optimizer.param_groups:
                        g['lr'] *= cfg.lr_reduction_factor
                    patience += (cfg.lr_reduction_cooldown + patience_increase)
                    continue
                else:
                    break

            # Update epoch here, or else the last value of epoch from the previous for loop will be duplicated.
            epoch = epoch + 1

            for epoch in tqdm(range(epoch, patience), desc="Epochs"):
                start_time = time.time()
                overall_losses, train_losses, breg_penalties = model.run_epoch(dataset.train_iterator, optimizer)
                epoch_time = time.time() - start_time
                training_time += epoch_time

                # Evaluate the model on (training and) validation
                train_acc = evaluate_model(model, dataset.train_iterator, not cfg.original_mode)
                val_acc = evaluate_model(model, dataset.val_iterator, not cfg.original_mode)

                _es = Util.dictize(epoch=epoch
                                   , overall_loss=float(np.mean(np.array(overall_losses)))
                                   , breg_penalties=float(np.mean(np.array(breg_penalties)))
                                   , loss_train=float(np.mean(np.array(train_losses)))
                                   , acc_train=train_acc
                                   , acc_val=val_acc)
                if cfg.model_cfg.trans_pe_type == PE_Type.ape:
                    _es['pe_variance'], _es['pe_norm'] = get_pe_variance(model.input_embed.pe,
                                                                         cfg.original_mode,
                                                                         MaxSenLen[cfg.experiment_data])

                if val_acc > best_val_metric:
                    patience = max(patience, epoch + patience_increase + 1)
                    best_val_metric = val_acc
                    best_epoch = epoch
                    if cfg.model_cfg.trans_pe_type == PE_Type.ape:
                        best_pe_variance, best_pe_norm = _es['pe_variance'], _es['pe_norm']
                    # Save Weights
                    best_model_dict = copy.deepcopy(model.state_dict())
                    log_stats(logger_stats, "saving_model_checkpoint", epoch=epoch)

                if cfg.model_cfg.trans_pe_type == PE_Type.ape:
                    _es['best_pe_variance'], _es['best_pe_norm'] = best_pe_variance, best_pe_norm

                _es.update(Util.dictize(best_val_metric=best_val_metric, best_epoch=best_epoch,
                                        patience_reductions=patience_reductions))

                epoch_stats.add(**_es)
                # Save state at the end of epoch
                torch.save(Util.dictize(
                    model=model.state_dict(), best_model=best_model_dict if best_model_dict is not None else None,
                    optimizer=optimizer.state_dict(), best_val_metric=best_val_metric,
                    patience=patience, patience_increase=patience_increase, epoch=epoch, best_epoch=best_epoch,
                    best_pe_variance=best_pe_variance, best_pe_norm=best_pe_norm, training_time=training_time,
                    patience_reductions=patience_reductions
                ), cfg.checkpoint_file_name())

        if best_model_dict is None:
            raise ValueError("Unexpected empty best_model_dict at the end of training.")
        best_model.load_state_dict(best_model_dict)

        # Double-check training is finished
        assert patience - epoch <= 1 and patience_reductions == cfg.lr_reduction_limit, \
            "Training must be finished before testing!"
        log_stats(logger_stats, 'Training finished!')

        return best_val_metric, best_model

    @classmethod
    def test_point_wise(cls, best_model, cfg, logger_stats, output_stats):
        log_stats(logger_stats, "---------Testing Model---------", testing_ordering=cfg.testing_ordering.name)
        dataset = cls.get_data(cfg, logger_stats)
        test_acc = evaluate_model(best_model, dataset.test_iterator, not cfg.original_mode)
        print(f'Final Test Accuracy: {test_acc}; Testing ordering: {cfg.testing_ordering.name}')
        _es_test = {'test_acc': test_acc, 'test_order': cfg.testing_ordering}
        if cfg.testing_ordering == SentenceOrdering.random:
            _es_test['test_shuffle_seed'] = cfg.testing_shuffle_seed
        if cfg.model_cfg.trans_pe_type == PE_Type.ape:
            _es_test['best_pe_variance'], _es_test['best_pe_norm'] = get_pe_variance(best_model.input_embed.pe,
                                                                                     cfg.original_mode,
                                                                                     MaxSenLen[cfg.experiment_data])
        log_stats(logger_stats, "Testing finished!")
        output_stats.add(**_es_test)

    @classmethod
    def main(cls):
        cfg = Config()

        # Initialize loggers
        logger_stats = JsonDump(cfg.log_file_name())
        epoch_stats = JsonDump(cfg.stats_file_name())
        output_stats = JsonDump(cfg.output_file_name())

        _, best_model = cls.dev_point_wise(cfg, logger_stats, epoch_stats)
        cls.test_point_wise(best_model, cfg, logger_stats, output_stats)

if __name__ == '__main__':
    Train.main()