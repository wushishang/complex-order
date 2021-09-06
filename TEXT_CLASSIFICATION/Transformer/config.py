import argparse
# import os
# import sys
# from collections import OrderedDict
# from warnings import warn
#
# from common.helper import PickleUtil, Util, print_stats
# from configs.data_config import DataArgParser
# from configs.graph_config import GraphDataArgParser
from common.helper import print_stats, Util, PickleUtil
from configs.model_config import ModelArgParser
# from configs.set_config import SetDataArgParser
from configs.transformer_config import TransformerArgParser
from my_common.my_helper import enum_sprint, is_positive_int
# from scalings import Scalings
from util.constants import TC_ExperimentData, TC_ModelType, Constants, MaxSenLen

RESULT_DIR = "./results/"
MODEL_DIR = "./model_save/"
PickleUtil.check_create_dir(RESULT_DIR)
PickleUtil.check_create_dir(MODEL_DIR)

COMMON_ARG_PARSERS = [ModelArgParser]  # DataArgParser,
# DATA_ARG_PARSERS = [GraphDataArgParser, SetDataArgParser]
MODEL_ARG_PARSERS = [TransformerArgParser]
#
#
class Config:
    def __init__(self, args=None):

        parser = argparse.ArgumentParser("Experiments with Regularization")
        # Gather all data and model arguments and add them to the parser
        for argparser in COMMON_ARG_PARSERS:
            self.add_common_args(parser, argparser)
        for argparser in MODEL_ARG_PARSERS:  # DATA_ARG_PARSERS +
            self.add_args(parser, argparser)

        # ##########
        # # PARAMETERS FOR DATA LOADING
        # ##########
        # parser.add_argument('-it', '--input_type', type=str, required=True,
        #                     help=f'Type of input. One of: {enum_sprint(InputType)}')
        # parser.add_argument('-ix', '--cv_fold', type=int, help='Which fold in cross-validation (e.g., 0 thru 5)',
        #                     required=True)
        # parser.add_argument('--train_all', default=False, action='store_true',
        #                     help='If set, model will be trained on both training and validation data')

        parser.add_argument('-data', '--experiment_data', required=True, type=str,
                            help=f'Data for experiment: {enum_sprint(TC_ExperimentData)}')
        parser.add_argument('-msl', '--max_sen_len', default=0, type=int,
                            help='Maximum length of sentences')

        ##########
        # PARAMETERS FOR MODEL
        ##########
        parser.add_argument('-mt', '--model_type', type=str, required=True,
                            help=f'Type of model. One of: {enum_sprint(TC_ModelType)}')

        ##########
        # PARAMETERS THAT CONTROL TRAINING AND EVALUATION
        ##########
        parser.add_argument('-ori', '--original_mode', action='store_true', help="Simulate (the bugs in) the original "
                            "code, including fixed PE, wrongly used .train() and .eval()")
        parser.add_argument('-lr', '--learning_rate', default=0.00001, type=float, help='Learning rate for Adam Optimizer')
        parser.add_argument('-ne', '--num_epochs', default=100, type=int,
                            help='Number of epochs for training at a maximum')
#         parser.add_argument('-pi', '--patience_increase', default=100, type=int,
#                             help='Number of epochs to increase patience')
#         parser.add_argument('-ni', '--num_inf_perm', default=5, type=int, help='Number of inference-time permutations')
#         parser.add_argument('-ei', '--eval_interval', default=50, type=int, help='Time interval of more-permutation inference')
#         parser.add_argument('-et', '--eval_train', default=False, action='store_true',
#                             help='Evaluate train loss/acc after backprop at each epoch')
        parser.add_argument('-sv', '--seed_val', default=-1, type=int,
                            help='Seed value, to get different random inits and variability')
#         parser.add_argument('-lt', '--loss_type', default=None, type=str,
#                             help=f'Optionally override default loss type for regression tasks. One of {enum_sprint(LossType)}')
#         parser.add_argument('-umb', '--use_mini_batching', default=False, action='store_true',
#                             help='Use mini-batching for training')
        parser.add_argument('-bs', '--batch_size', default=32, type=int,
                            help='Size of mini-batches when mini-batching is used')
#         parser.add_argument('-dl', '--drop_last', default=False, action='store_true',
#                             help='Drop the last incomplete batch, if the dataset size is not divisible by the batch size.')
#         parser.add_argument('-bl', '--batch_limit', default=1024, type=int,
#                             help='Maximum size of mini-batches or the full-batch.')
#         parser.add_argument('-dsb', '--dont_shuffle_batches', default=False, action='store_true',
#                             help="Don't shuffle batches for every epoch")
#         parser.add_argument('--dont_permute_for_pi_sgd', default=False, action='store_true',
#                             help='If set, pi-SGD training will NOT be used: data will not be permuted')
#         parser.add_argument('-ppe', '--permute_positional_encoding', action='store_true',
#                             help='If true, permute positional encodings (default: permute input data)')
#
#         ##########
#         # PARAMETERS THAT CONTROL PADDING
#         ##########
#         parser.add_argument('-ppd', '--pre_padding', action='store_true', help='If true, pre-pad the adjacency matrices '
#                             '(and node features, if applicable) in case of variable-size graphs.')
#         parser.add_argument('-pid', '--X_padding_idx', default=0, type=int, help='Padding index used to pad node/elem features.')
#
#         ##########
#         # PARAMETERS THAT CONTROL VARIANCE EVALUATION
#         ##########
#         # Test variability: every "eval_interval", assess the variability in farrow to permutations
#         # This will track it during training, as well as at the end of training. We can always load a pre-trained
#         # model and simply evaluate variability in the output file.
#         parser.add_argument('--test_variability', action='store_true',
#                             help="Run experiment to track variability of farrow through training every `eval_interval`")
#         parser.add_argument('--track_latent_norm', action='store_true',
#                             help="Track the norm of the latent representation through training every epoch")
#
#         ##########
#         # PARAMETERS FOR SCALING LATENT "EMBEDDINGS"
#         # Hypothesis: intermediate layers can arbitrarily scale and unscale their outputs, thwarting attempts to
#         #             regularize permutation sensitivity according to the distance between vectors.
#         ##########
#         parser.add_argument('-sc_p', '--scaling_penalize_embedding', action='store_true', default=False,
#                             help="Scaling Option to penalize the L2 norm of the latent embedding")
#         parser.add_argument('-sc_ep', '--scaling_embedding_penalty', default=None,
#                             help="Scaling Option: strength of L2 penalty. Must be set if --scaling_penalize_embedding")
#         parser.add_argument('-sc_q', '--scaling_quantize', action='store_true', default=False,
#                             help="Scaling Option: Add random normal noise to the latent representations")
#
#         ##########
#         # PARAMETERS FOR PRE-EMBEDDING
#         ##########
#         parser.add_argument('--input_embedding_type', type=str, default=None,
#                             help=f"Type of the input embedding, {enum_sprint(InputEmbedType)}")
#         parser.add_argument('--input_embedding_dim', type=int, default=None,
#                             help="Dimension of the input embedding.")
#         parser.add_argument('--input_embedding_learnable', action='store_true', default=False,
#                             help="Pre-embedding option to make embedding learnable")
#
#         ##########
#         # PARAMETERS FOR WEIGHT-DECAY
#         ##########
#         parser.add_argument('-wd', '--weight_decay', action='store_true',
#                             help='Apply Weight Decay to the positional embedding, if applicable (default: False)')
#         parser.add_argument('--wd_strength', nargs='+', default=[0.], type=float,
#                             help='The (list of) strength in the weight decay terms for positional embedding (default: 0.)')
#         parser.add_argument('--wd_power', nargs='+', default=[2], type=int,
#                             help='The (list of) powers in the weight decay terms for positional embedding (default: 2)')
#
#         ##########
#         # PARAMETERS THAT CONTROL REGULARIZATION
#         ##########
#         #
#         # General regularization options
#         #
#         parser.add_argument('-r', '--regularization', type=str,
#                             help='Type of regularization to apply', default='none')
#         parser.add_argument('-r_strength', '--regularization_strength', default=None,
#                             help='Regularization strength')
#         # User be aware! r_eps effectively becomes multiplied by the regularization
#         # penalty in finite difference penalties.  So it interacts with reg strength
#         parser.add_argument('-r_eps', '--regularization_epsilon', default=None, type=float,
#                             help='Size of finite difference for regularization')
#         parser.add_argument('--fixed_edge', action='store_true',
#                             help='Fix the edge of DS matrices for diff_step_edge regularization')
#         parser.add_argument('--random_segment', action='store_true',
#                             help='Randomly swap two rows when constructing the DS matrices for diff_step_edge regularization')
#         # TODO: implement more permutations later (argument: '-r_np')
#         parser.add_argument('-r_np', '--regularization_num_perms', default=None,
#                             help='Number of extra permutations for regularization')
#         parser.add_argument('-r_repr', '--regularization_representation', type=str, default='none', help=f"Penalize \
#                             permutation sensitivity in one of {enum_sprint(RegRepresentation)} (default: none)")
#         parser.add_argument('--r_normed', default=False, action='store_true',
#                             help='Compute norm, not only a power, of penalty by taking root (default:False)')
#         parser.add_argument('--r_power', default=2, type=int,
#                             help='The power in the penalty term (default: 2)')
#         parser.add_argument('-r_agg_by_node', '--aggregate_penalty_by_node', default=False, action='store_true',
#                             help="This argument only affects node classification tasks. If True, "
#                                  "the penalty will be mean aggregated by nodes; otherwise by graphs (default: by graphs)")
#         #
#         # TangentProp options
#         #
#         parser.add_argument('-tpp', '--tangent_prop', action='store_true',
#                             help='Use TangentProp regularization (default: False)')
#         parser.add_argument('-tp_ntv', '--tp_num_tangent_vectors', default=1, type=int,
#                             help='Number of tangent vectors for TangentProp (default:1)')
#         #
#         # Regularization Schedulers
#         #
#         # Extended from Delayed Strong Regularization: https://openreview.net/pdf?id=Bys_NzbC-
#         # TODO: refactor this option when implementing other schedulers
#         parser.add_argument('-tpr', '--two_phase_regularization', action='store_true',
#                             help='Train with two-phase regularization, namely, unreg phase and reg phase.')
#         parser.add_argument('--pull_down', action='store_true',
#                             help='Run reg phase first followed by unreg phase (default: pull-up)')
#         parser.add_argument('--milestone', type=int, help='Epoch of phase changing.')
#
#         ##########
#         # PARAMETERS THAT CONTROL TESTING
#         ##########
#         parser.add_argument('--testing', action='store_true', default=False,
#                             help='Do evaluation for the specified trained model on testing data')
#         parser.add_argument('-tchk', '--test_checkpoint_id', default=None, type=str,
#                             help='Checkpoint string of model to test (ie output of checkpoint_file_name()). Must finish training before testing.')
#         parser.add_argument('-tats', '--test_add_train_set', default=False, action='store_true',
#                             help='Add specifications of training set to model_id for distinguishable output files.')

        args = parser.parse_args(args)

        # ===================================================
        # Parameters for data and model types
        # ===================================================
        self.experiment_data = TC_ExperimentData[str(args.experiment_data)]
        print_stats("-" * 10)
        print_stats(f"Experiment data: {self.experiment_data.name}")
        print_stats("-" * 10)

        assert isinstance(args.max_sen_len, int)
        if args.max_sen_len > 0:
            self.max_sen_len = args.max_sen_len
        elif args.max_sen_len == 0:
            self.max_sen_len = MaxSenLen[self.experiment_data]
            print_stats(f"Specified max_sen_len is 0. Reset to max_sen_len of training samples of {self.experiment_data.name}.")

#         # ===================================================
#         # Parameters for data and model types
#         # ===================================================
#         self.input_type = InputType[args.input_type.lower()]
        self.model_type = TC_ModelType[args.model_type.lower()]
#
#         # assign parsed hyperparameters to attributes of config containers
#         if self.input_type == InputType.set:
#             assert self.model_type != ModelType.gin_like, "Set data cannot be handled by RpGin."
#             self.data_cfg = self.parsed_to_cfg(args, SetDataArgParser)
#         elif self.input_type == InputType.graph:
#             self.data_cfg = self.parsed_to_cfg(args, GraphDataArgParser)
#         else:
#             raise NotImplementedError(f"Unable to handle {self.input_type} yet.")

        self.original_mode = args.original_mode
#
        if self.model_type == TC_ModelType.transformer:
            self.model_cfg = self.parsed_to_cfg(args, TransformerArgParser)
#         elif self.model_type == ModelType.lstm:
#             raise NotImplementedError
#         elif self.model_type == ModelType.transformer:
#             self.model_cfg = self.parsed_to_cfg(args, TransformerArgParser)
        else:
            raise NotImplementedError(f"Haven't implemented {self.model_type} yet.")
#
#         # ===================================================
#         # Parameters for cross-validation
#         # ===================================================
#         self.cv_fold = args.cv_fold
#         if self.data_cfg.experiment_data not in (ExperimentData.br, ExperimentData.zinc):
#             assert 0 <= self.cv_fold
#         else:
#             assert -1 <= self.cv_fold  # -1 indicates the testing set
#         if self.data_cfg.experiment_data == ExperimentData.rp_paper:
#             assert self.cv_fold < 5  # While we load the rp_paper data, user may enter n_splits other than 5
#         else:
#             assert self.cv_fold < self.data_cfg.n_splits
#
#         self.train_all = args.train_all
#
#         # ===================================================
#         # Parameters for optimization and computation
#         # ===================================================
#
#         # ===================================================
#         # Parameters for experiment data
#         # ===================================================
#         if self.data_cfg.experiment_data in (ExperimentData.rp_paper, ExperimentData.customized):
#             self.task_type = TaskType.multi_classification
#             if self.data_cfg.experiment_data == ExperimentData.rp_paper:
#                 self.data_cfg.sparse = True
#                 self.data_id = ""
#                 self.sample_random_state = None
#             elif self.data_cfg.experiment_data == ExperimentData.customized:
#                 self.data_id = self.data_cfg.get_data_id_string()
#                 self.sample_random_state = self.data_cfg.sample_random_state
#         elif self.data_cfg.experiment_data in (ExperimentData.smp, ExperimentData.er_edges,
#                                                ExperimentData.brain, ExperimentData.zinc):
#             self.data_id = self.data_cfg.get_data_id_string()
#             self.sample_random_state = None
#         elif self.data_cfg.experiment_data in (ExperimentData.ba, ExperimentData.ger,
#                                                ExperimentData.scalar_int_arm, ExperimentData.scalar_cont_arm):
#             self.data_id = self.data_cfg.get_data_id_string()
#             self.sample_random_state = self.data_cfg.sample_random_state
#         else:
#             raise ValueError(f"Invalid ExperimentData given: {self.data_cfg.experiment_data}")
#
#         if self.input_type == InputType.graph:
#             self.load_sparse = self.data_cfg.sparse
#         else:
#             self.load_sparse = False
#
#         # ===================================================
#         # Parameters for pickle loading
#         # ===================================================
#         self.data_path = self.data_cfg.get_data_path()
#         # Check if data exists
#         if os.path.exists(self.data_path):
#             print(f"Will load data from {self.data_path}")
#         else:
#             raise FileNotFoundError(f"{self.data_path} does not exist! Please create data using \'create_data.py\'.")
#
#         # ===================================================
#         # Parameters that control training, validation and/or testing
#         # ===================================================
        self.learning_rate = args.learning_rate
        self.num_epochs = args.num_epochs
#         self.patience_increase = args.patience_increase
#         assert self.patience_increase >= 0, "Patience is a non-negative int"
#         self.num_inf_perm = args.num_inf_perm
#         self.eval_interval = args.eval_interval
#         assert self.eval_interval > 0, "eval_interval must be a positive integer"
#         self.eval_train = args.eval_train
        self.seed_val = args.seed_val
#         if args.loss_type is not None:
#             self.loss_type = LossType[args.loss_type.lower()]
#         else:
#             self.loss_type = None
#
#         self.use_mini_batching = args.use_mini_batching  # will be checked in Train
#         # If mini-batching options are specified, we must be using mini-batching
#         assert args.batch_size > 0 and args.batch_limit > 0, "batch_size and batch_limit must be strictly positive."
#         if args.batch_size != parser.get_default('batch_size') or args.drop_last != parser.get_default('drop_last'):
#             assert self.use_mini_batching, \
#                 "Non-default values of mini-batching options were specified, but --use_mini_batching is False"
        self.batch_size = args.batch_size  # will be checked in Train
#         self.drop_last = args.drop_last
#         self.batch_limit = args.batch_limit
#         self.shuffle_batches = not args.dont_shuffle_batches
#
#         self.pre_padding = args.pre_padding
#         self.X_padding_idx = args.X_padding_idx
#         if args.pre_padding:
#             assert self.input_type == InputType.graph, "pre_padding is only applicable to graphs for now."
#         else:
#             self.X_padding_idx = None
#             print_stats("Padding_idx is set to None as --pre_padding is False")
#
#         # ===================================================
#         # Parameters that control permuting and pi-SGD
#         # ===================================================
#         if args.dont_permute_for_pi_sgd and args.num_inf_perm != parser.get_default('num_inf_perm'):
#             raise ValueError("Cannot specify `--dont_permute_for_pi_sgd` and non-default num_inf_perm")
#         if not args.dont_permute_for_pi_sgd and self.model_type == ModelType.gin_like:
#             assert self.model_cfg.gin_model_type == GinModelType.rpGin, "Pi-SGD cannot be used with Gin models other than rpGin"
#         self.permute_for_pi_sgd = not args.dont_permute_for_pi_sgd
#
#         self.permute_positional_encoding = args.permute_positional_encoding
#
#         # ===================================================
#         # Parameters that control variance evaluation
#         # ===================================================
#         self.test_variability = args.test_variability
#         self.track_latent_norm = args.track_latent_norm
#
#         # ===================================================
#         # Parameters that control scaling
#         # (Scaling is relevant for both baseline and regularization, since we may want to
#         #  do a fair comparison)
#         # ===================================================
#         # Build a scaling struct, and let the class do the input validation
#         self.scaling = Scalings(penalize_embedding=args.scaling_penalize_embedding,
#                                 quantize=args.scaling_quantize,
#                                 embedding_penalty=args.scaling_embedding_penalty)
#
#         # ===================================================
#         # Parameters that control pre-embedding of set inputs
#         # ===================================================
#         # (checks are deferred to PreEmbedding class constructor)
#         if args.input_embedding_type is not None:
#             self.input_embedding_type = InputEmbedType[args.input_embedding_type]
#         else:
#             self.input_embedding_type = None
#         self.input_embedding_dim = args.input_embedding_dim
#         self.input_embedding_learnable = args.input_embedding_learnable
#
#         if self.input_embedding_learnable:
#             raise NotImplementedError("The embedding strategies currently implemented should not be learnable.")
#             # assert self.input_embedding_type is not None, \
#             #     "Asked for learnable embeddings but --input_embedding_type is None"
#
#         if self.input_embedding_type is not None:
#             if self.model_type==ModelType.gin_like and self.model_cfg.positional_encoding_type != GinPositionalEncodingType.none:
#                 assert self.model_cfg.positional_encoding_dim == self.input_embedding_dim, \
#                     "Input_embedding_dim must match positional_encoding_dim of GIN."
#
#         # ===================================================
#         # Parameters that control weight-decay of positional embedding
#         # ===================================================
#         self.weight_decay = args.weight_decay
#         self.wd_strength = args.wd_strength
#         self.wd_power = args.wd_power
#
#         # ===================================================
#         # Parameters that control regularization
#         # ===================================================
#         self.regularization = Regularization[args.regularization.lower()]
#         if self.regularization != Regularization.none:
#             if self.model_type == ModelType.gin_like:
#                 assert self.model_cfg.gin_model_type == GinModelType.rpGin, "Regularizers cannot be used with Gin models other than rpGin"
#         self.regularization_strength = args.regularization_strength
#         self.regularization_eps = args.regularization_epsilon
#         self.num_reg_permutations = args.regularization_num_perms
#         self.fixed_edge = args.fixed_edge
#         self.random_segment = args.random_segment
#
#         if self.regularization_strength is not None:
#             self.regularization_strength = float(self.regularization_strength)
#             assert self.regularization_strength >= 0.0  # Allow 0.0 for debug and easy hyperparameter search
#
#         if self.regularization != Regularization.none and self.regularization != Regularization.naive:
#             assert self.regularization_eps is not None
#             assert self.regularization_eps > 0.0
#
#         if self.num_reg_permutations is not None:
#             self.num_reg_permutations = int(self.num_reg_permutations)
#             assert self.num_reg_permutations > 0
#
#         self.r_normed = args.r_normed
#         self.r_power = args.r_power
#         assert self.r_power > 0
#
#         self.regularization_representation = RegRepresentation[args.regularization_representation.lower()]
#         self.tangent_prop = args.tangent_prop
#
#         if args.r_normed or args.r_power != parser.get_default('r_power'):
#             assert self.regularization != Regularization.none, \
#                 "Non-default values of r_normed or r_power were specified, but self.regularization is none"
#
#         if args.aggregate_penalty_by_node != parser.get_default('aggregate_penalty_by_node'):
#             assert self.regularization != Regularization.none and self.input_type == InputType.graph, "Non-default value \
#             of aggregate_penalty_by_node was specified, but regularization is none or the input is not graphs."
#         self.aggregate_penalty_by_node = args.aggregate_penalty_by_node
#
#         # If tangent prop options are specified, we must be using tangent prop
#         if args.tp_num_tangent_vectors != parser.get_default('tp_num_tangent_vectors'):
#             assert self.tangent_prop, \
#                 "Non-default value of tp_num_tangent_vectors was specified, but --tangent_prop is False"
#
#         assert args.tp_num_tangent_vectors > 0
#         self.tp_config = OrderedDict()
#         self.tp_config['tp_num_tangent_vectors'] = args.tp_num_tangent_vectors
#
#         self.two_phase_regularization = args.two_phase_regularization
#         self.pull_down = args.pull_down
#         self.milestone = args.milestone
#         if args.pull_down != parser.get_default('pull_down') or args.milestone is not None:
#             assert self.two_phase_regularization, \
#                 "Non-default values of pull_down or milestone were specified, but --two_phase_regularization is False"
#         if self.two_phase_regularization:
#             assert self.regularization != Regularization.none, \
#                 "Two-phase regularization is selected but regularization is inactive."
#             assert self.milestone > 0, "Milestone epoch of two-phase regularization must be a positive integer."
#             assert self.milestone < self.num_epochs, "Milestone epoch should be smaller than the total number of training epochs."
#
#         # ===================================================
#         # Parameters that control testing
#         # ===================================================
#         self.testing = args.testing
#         self.test_checkpoint_id = args.test_checkpoint_id
#         self.test_add_train_set = args.test_add_train_set
#         if self.testing:
#             assert self.test_checkpoint_id is not None
#             # Delete some strings if present, so that user has more flexibility in what they enter.
#             # Train.py prints the checkpoint, which the user can copy-and-paste, but it has extra strings.
#             #   When appended, those strings could cause confusion
#             self.test_checkpoint_id = self.test_checkpoint_id.replace(Constants.CHECKPOINT_TAIL, "")
#             self.test_checkpoint_id = self.test_checkpoint_id.replace(MODEL_DIR, "")
#             assert os.path.exists(self.checkpoint_file_name()), "Checkpoint file of the testing model does not exist!"
#             print(f"Will load trained model from {self.checkpoint_file_name()}")
#
#         # ===========================================================================
#         # Assertions and Warnings for some (data, model, training) configurations:
#         # ===========================================================================
#         #
#         # Transformers: (1) warn if using with pi-SGD; pi-SGD must be used with positional encoding
#         #               (2) regularization => positional encoding
#         if self.model_type == ModelType.transformer:
#             if self.permute_for_pi_sgd:
#                 warn("\n\n\nWarning: Using pi-SGD with transformers?\n\n")
#                 assert self.model_cfg.trans_positional_encoding, \
#                     "It only makes sense to use pi-SGD w/ transformers if PE is active."
#
#             if self.regularization != Regularization.none:
#                 assert self.model_cfg.trans_positional_encoding
#
#         #
#         # GIN: (1) pi-SGD must be used with positional encoding
#         #      (2) regularization => positional encoding
#         #
#         if self.model_type == ModelType.gin_like:
#             if self.permute_for_pi_sgd:
#                 assert self.model_cfg.positional_encoding_type != GinPositionalEncodingType.none, \
#                     "It only makes sense to use pi-SGD w/ GIN if PE is active."
#
#             if self.regularization != Regularization.none:
#                 assert self.model_cfg.positional_encoding_type != GinPositionalEncodingType.none
#
#         #
#         # Checks related to embeddings
#         #
#         # Make sure we don't use word embedding with non-integer input
#         if self.data_cfg.experiment_data not in (ExperimentData.scalar_int_arm, ExperimentData.zinc):  # These experiments use int input
#             assert self.input_embedding_type != InputEmbedType.word_embedding, \
#                     f"Word embedding not applicable for {self.data_cfg.experiment_data} (it expects int inputs)"
#
#         # Scalar arithmetic => usually use pre-embedding.
#         if self.data_cfg.experiment_data in (ExperimentData.scalar_int_arm, ExperimentData.scalar_cont_arm):
#             if self.input_embedding_type is None:
#                 warn(f"\n\nWarning: No input embedding for {self.data_cfg.experiment_data}.\n\n")
#         #
#         # Checks specific for set and graph tasks.
#         #
#         if self.input_type == InputType.set:
#             # - We should use L1 loss for max-regression task of sets.
#             if self.data_cfg.set_function == SetFunction.max and self.loss_type != LossType.mae:
#                 warn("\n\nWarning: L1 loss is not in use for the max-regression task of sets.\n\n")
#         elif self.input_type == InputType.graph:
#             if self.input_embedding_type is not None:
#                 warn("\n\nWarning: Using Pre-embedding for graphs?\n\n")
#
#         #
#         # Checks specific for data and training/testing.
#         #
#         if self.train_all:
#             assert self.data_cfg.experiment_data in (ExperimentData.br, ExperimentData.zinc), \
#                 "Only support train_all for real-world datasets"
#             assert 0 <= self.cv_fold, "Expected non-negative cv_fold which contains indices of training and validation"
#             assert self.patience_increase == 0, "patience_increase should be zero since there is no validation data"
#             assert not self.testing, "train_all can only be set for training"
#
#         # Check that pre_padding only works with edge-step regularizer and no SGD
#         if self.pre_padding:
#             if self.regularization != Regularization.none:
#                 assert self.regularization == Regularization.edge, "Pre-padding only works with edge-step / random-segment" \
#                                                                    "regularization for now."
#             assert not self.permute_for_pi_sgd, "Pre-padding does NOT work with pi-SGD for now."
#
#             if not self.permute_positional_encoding:
#                 warn("\n\nWarning: Using pre-padding while permuting data instead of PE?\n\n")

        # ===========================================================================
        # Initialize model id according to data, model and training configurations.
        # Hashed by md5 if the length exceeds linux system limit.
        # ===========================================================================
        self.init_model_id()

    def add_common_args(self, parser, subparser):
        """
        Wrapper for adding common arguments from other config files
        """
        return subparser.add_common_args(parser)

    def add_args(self, parser, subparser):
        """
        Wrapper for adding arguments from other config files
        """
        return subparser.add_args(parser)

    def parsed_to_cfg(self, parsed, subparser):
        """
        Wrapper for initializing active data/model config containers
        """
        # parsed arguments must not be None
        assert parsed is not None
        return subparser(None, parsed)

    @staticmethod
    def params_to_str(params):
        params = map(lambda x: x if not isinstance(x, bool) else ("T" if x else "F"), params)
        return "_".join(map(str, params))

    def _data_params(self):
        """
        Return Data Information
        """
        data_params = [self.experiment_data.name, self.max_sen_len]  # self.cv_fold

        # data_params += [self.data_id, self.data_cfg.n_splits, self.data_cfg.use_default_train_val_split,
        #                 self.data_cfg.scale_target, self.data_cfg.balance_target]
        # if self.data_cfg.shuffle :
        #     data_params += [self.data_cfg.shuffle_random_state]

        return data_params

    def _model_params(self):
        """
        Return Model Information
        """
        model_params = [self.model_type.name]
        model_params += self.model_cfg.get_model_id_list()
        # if hasattr(self.model_cfg, 'use_batchnorm') and self.model_cfg.use_batchnorm:
        #     model_params += [self.batch_limit]  # batch limit impacts batch sizes used by batchnorm

        return model_params

    def _train_params(self):
        """
        Return Training Information
        """
        train_params = [self.original_mode, self.learning_rate, self.num_epochs, self.seed_val, self.batch_size]
#         train_params = [self.permute_for_pi_sgd, self.permute_positional_encoding, self.learning_rate,
#                         self.patience_increase, self.seed_val, self.num_epochs, self.num_inf_perm, self.eval_interval,
#                         self.eval_train, self.shuffle_batches]
#         if self.loss_type is not None:
#             train_params += [self.loss_type.name]
#         #
#         # Add Padding information
#         #
#         if self.pre_padding:
#             train_params += ["pre_pad", self.X_padding_idx]
#         #
#         # Add Batching information
#         #
#         train_params += ["mini_batches", self.batch_size, self.drop_last] if self.use_mini_batching else ["full_batch"]
#         #
#         # Add Scaling information
#         #
#         if self.scaling.scaling_active:
#             train_params += ["s"]
#             train_params += [self.scaling.penalize_embedding]
#             if self.scaling.penalize_embedding:
#                 train_params += [self.scaling.embedding_penalty]
#             train_params += [self.scaling.quantize]
#         #
#         # Add Pre-embedding information
#         #
#         if self.input_embedding_type is not None:
#             train_params += [self.input_embedding_type.name, self.input_embedding_dim, self.input_embedding_learnable]
#         #
#         # Add WeightDecay Information
#         #
#         train_params += [self.weight_decay]
#         if self.weight_decay:
#             train_params += self.wd_strength
#             train_params += self.wd_power
#         #
#         # Add Regularization Information
#         #
#         if self.regularization != Regularization.none:
#             train_params += [self.regularization.name]
#             if self.regularization in (Regularization.diff_step_edge, Regularization.diff_step_basis):
#                 train_params += [self.fixed_edge, self.random_segment]
#             train_params += [self.num_reg_permutations, self.regularization_strength, self.regularization_eps,
#                              self.regularization_representation.name, self.r_normed, self.r_power, self.tangent_prop]
#             if self.tangent_prop:
#                 train_params += list(self.tp_config.values())
#             if self.input_type == InputType.graph:
#                 train_params += [self.aggregate_penalty_by_node]
#             if self.two_phase_regularization:
#                 train_params += [self.pull_down, self.milestone]
#         #
#         # Add General Information
#         #
#
        return train_params

    def init_model_id(self):
        #
        # Add Dataset Information
        #
        data_params = self._data_params()
        # if self.testing and "DAT" in self.test_checkpoint_id:
        #     # test_checkpoint_id is not hashed; do sanity check of data specifications - expects different data
        #     test_data_string = self.params_to_str(data_params)
        #     train_data_string = self.test_checkpoint_id.split("DAT_")[1].split("_MOD")[0]
        #     if test_data_string == train_data_string:
        #         warn_msg = "\n" + "!" * 30
        #         warn_msg += "\nTesting data has identical parameters to that of training data"
        #         warn_msg += f"\ndata_string: {test_data_string}\n"
        #         warn_msg += "!" * 30
        #         warn(warn_msg)
        #     elif self.test_add_train_set:
        #         data_params += [train_data_string]
        data_params = ["DAT"] + data_params
        #
        # Add Model Information
        #
        model_params = self._model_params()
        # if self.testing and "MOD" in self.test_checkpoint_id:
        #     # test_checkpoint_id is not hashed; do sanity check of model specifications - requires same model
        #     test_model_string = self.params_to_str(model_params)
        #     train_model_string = self.test_checkpoint_id.split("MOD_")[1].split("_TRA")[0]
        #     assert test_model_string == train_model_string, \
        #         f"\ntest_model_string: {test_model_string}\n" \
        #         f"train_model_string: {train_model_string}\n" \
        #         f"Model configurations must match those in the entered test_checkpoint_id!"
        model_params = ["MOD"] + model_params
        #
        # Add Training/testing Information
        #
        train_params = self._train_params()
        # if self.testing and "TRA" in self.test_checkpoint_id:
        #     # test_checkpoint_id is not hashed; do sanity check of training/testing specifications - expects same spec
        #     test_spec_string = self.params_to_str(train_params)
        #     train_spec_string = self.test_checkpoint_id.split("TRA_")[1]
        #     if test_spec_string != train_spec_string:
        #         warn_msg = "\n" + "!" * 30
        #         warn_msg += "\nTesting specifications are different from training specifications"
        #         warn_msg += f"\ntest_spec_string: {test_spec_string}"
        #         warn_msg += f"\ntrain_spec_string: {train_spec_string}\n"
        #         warn_msg += "!" * 30
        #         warn(warn_msg)
        train_params = ["TRA"] + train_params  # ["TRA"] if not self.testing else ["TES"]

        self.model_id = self.params_to_str(data_params + model_params + train_params)
        # Last resort: hash the filenames, at the cost of human readability
        if len(self.model_id) > Constants.MODEL_ID_MAX:
            print_stats(f"Model ID too long! MAX: {Constants.MODEL_ID_MAX}; Length of '{self.model_id}': {len(self.model_id)}.")
            print_stats("Filename will be hashed by md5.")
            self.model_id = Util.md5(self.model_id)
            print_stats("New ID", hashed_id=self.model_id)

    def output_file_name(self):
        return f"{RESULT_DIR}{self.model_id}.output.txt"

    def stats_file_name(self):
        return f"{RESULT_DIR}{self.model_id}.stats.txt"

    def log_file_name(self):
        return f"{RESULT_DIR}{self.model_id}.log"

    def checkpoint_file_name(self):
        # if self.testing:
        #     return f"{MODEL_DIR}{self.test_checkpoint_id}{Constants.CHECKPOINT_TAIL}"
        # else:
        return f"{MODEL_DIR}{self.model_id}{Constants.CHECKPOINT_TAIL}"
#
#     @property
#     def track_var_pe_weights(self):
#         if self.model_type == ModelType.transformer:
#             return self.model_cfg.trans_positional_encoding
#         elif self.model_type == ModelType.gin_like:
#             return self.model_cfg.positional_encoding_type == GinPositionalEncodingType.random
#         else:
#             return False
#

if __name__ == "__main__":
    cfg = Config()
    print("\n", cfg.stats_file_name(), "\n")
    print(cfg.scaling)
