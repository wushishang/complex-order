from abc import abstractmethod
from argparse import ArgumentParser, Namespace

from common.helper import print_stats
from configs.config_argparser import ArgParser
from data.Create_CSL import SYNTHETIC_DIR
from data.canonical_orderings import Canonical_Order, Canonical_Orderings
from data.random_noise import Random_Noise
from my_common.my_helper import enum_sprint, args_list_to_dict
from util.constants import DEFAULT_RANDOM_GENERATION_SEED, ExperimentData, SwappingScheme, PickleBy, NoiseType, Constants

PROCESSED_DATA_DIR = f'{SYNTHETIC_DIR}processed/'


class DataArgParser(ArgParser):

    @classmethod
    def add_common_args(cls, parser):
        assert isinstance(parser, ArgumentParser)

        # Common configurations for data generation
        parser.add_argument('-data', '--experiment_data', required=True, type=str,
                            help=f'Data for experiment: {enum_sprint(ExperimentData)}')
        parser.add_argument('--sample_random_state', default=DEFAULT_RANDOM_GENERATION_SEED, type=int,
                            help='random state for sampling data.')
        parser.add_argument('--balance_target', default=False, action="store_const", const=True,
                            help='Create data with balanced targets (by sampling data until targets are balanced).')

        # Common configurations for data ordering
        # Pre-orderings
        parser.add_argument("--pre_canonical", default=None, type=str, help=f"Order hidden data by the specified \
                            pre-canonical order. One of {enum_sprint(Canonical_Order)}")
        parser.add_argument("--pre_canonical_kwargs", nargs="+",
                           help=f"Arguments for the pre-canonical ordering function, e.g., 'seed' for randomly_shuffle")
        # Random permuting
        parser.add_argument("--noise_type", default="none",
                            help=f"How to add random (uniform) noise to the hidden data. One of: {enum_sprint(NoiseType)}")
        parser.add_argument('--noise_probability', default=0., type=float, help='Probability of randomly permuting \
                            the hidden data. Only applicable when the noise type is bernoulli.')
        parser.add_argument('--noise_random_state', default=Constants.NOISE_RANDOM_STATE, type=int,
                            help='Random state of the random noise.')

        # Post-orderings
        parser.add_argument("--swapping_scheme", default=None,
                            help=f"How to swap elements in validation data. One of: {enum_sprint(SwappingScheme)}")
        parser.add_argument("--do_swapping", default=False, action='store_true',
                            help=f"Perform a swapping of elements in the non-training data")
        parser.add_argument("--canonical", default=None, nargs="+", help=f"Order elements by the specified canonical order. \
                            One argument for both train and valid (keeping test identical), two arguments for train and \
                            valid respectively, three arguments for train, valid and test respectively. \
                            Available orderings: {enum_sprint(Canonical_Order)}")
        parser.add_argument("--canonical_kwargs", nargs="+",
                            help=f"Arguments for the canonical ordering functions, e.g., 'seed' for randomly_shuffle")

        # Number of CV splits: currently, set to 2 when swapping
        parser.add_argument('--n_splits', default=5, type=int, help='Number of folds')
        parser.add_argument('--use_default_train_val_split', action='store_true',
                            help='Use default train/val split for the real-world dataset.')
        parser.add_argument('--hold_out_default_test', action='store_true',
                            help='Hold out default testing data for the real-world dataset.')
        parser.add_argument('--print_splits', action='store_true',
                            help='Print the indices of cv splits')
        parser.add_argument('--recreate_data', action='store_true',
                            help='If data files already exist, setting this flag will recreate them')
        parser.add_argument('--dont_shuffle', default=False, action="store_const", const=True,
                            help="Turn off shuffling each class' samples before splitting")
        parser.add_argument('--shuffle_random_state', default=1, type=int,
                            help='random_state affects the ordering of the indices if shuffle is True. Turned off if --dont_shuffle is passed')

        parser.add_argument('--scale_target', action='store_true', help='If true, targets are scaled to mean 0 var 1')

        # Common configurations for data saving
        parser.add_argument("--pickle_by", type=str, default=PickleBy.pickle.name,
                            help=f"Serialize data as pickle files using one of: {enum_sprint(PickleBy)}")

    def assign_common_parsed(self, parsed_args):

        assert isinstance(parsed_args, Namespace)

        ##########
        # PARAMETERS THAT CONTROL DATA GENERATION
        ##########
        self.data_dir = PROCESSED_DATA_DIR
        self.experiment_data = ExperimentData[str(parsed_args.experiment_data).lower()]
        print_stats("-" * 10)
        print_stats(f"Experiment data: {self.experiment_data.name}")
        print_stats("-" * 10)
        self.sample_random_state = parsed_args.sample_random_state

        self.recreate_data = parsed_args.recreate_data
        if self.recreate_data:
            print_stats("We will re-create data (if applicable)")

        ##########
        # PARAMETERS THAT CONTROL DATA ORDERING
        ##########
        self.pre_ordering = Canonical_Orderings(parsed_args.pre_canonical,
                                                self.experiment_data,
                                                args_list_to_dict(parsed_args.pre_canonical_kwargs))
        if self.pre_ordering.do_ordering:
            assert self.experiment_data in (ExperimentData.scarim, ExperimentData.scarom), \
                "Pre-ordering only applies to sets at this time!"

        self.noise = Random_Noise(parsed_args.noise_type, parsed_args.noise_random_state, parsed_args.noise_probability)

        self.do_swapping = parsed_args.do_swapping
        if self.do_swapping:
            assert parsed_args.swapping_scheme != self.parser.get_default("swapping_scheme")
            assert parsed_args.n_splits == 2, "Only swap when n_splits == 2 (train/test), o.w. logic swaps elems in all data"

        if parsed_args.swapping_scheme != self.parser.get_default("swapping_scheme"):
            self.swapping_scheme = SwappingScheme[parsed_args.swapping_scheme]
        else:  # Set none: we need to reference this var whether or not swapping is in effect to build the file names.
            self.swapping_scheme = None

        self.ordering = Canonical_Orderings(parsed_args.canonical,
                                            self.experiment_data,
                                            args_list_to_dict(parsed_args.canonical_kwargs))
        self._set_default_canonical_function_seeds()

        ##########
        # PARAMETERS THAT CONTROL CV SPLIT
        ##########
        self.n_splits = parsed_args.n_splits
        assert self.n_splits > 1
        self.print_splits = parsed_args.print_splits

        self.use_default_train_val_split = parsed_args.use_default_train_val_split
        self.hold_out_default_test = parsed_args.hold_out_default_test

        if self.use_default_train_val_split:
            assert self.n_splits == 2

        # Check if the default data splitting is available for the dataset
        if self.hold_out_default_test:
            assert self.experiment_data in (ExperimentData.zinc,), \
                f"Default testing data is NOT available for {self.experiment_data.name}."

        self.shuffle = not parsed_args.dont_shuffle
        self.shuffle_random_state = parsed_args.shuffle_random_state
        if self.shuffle:
            assert self.shuffle_random_state > 0
        else:
            self.shuffle_random_state = None
            print_stats("Indices will not be shuffled before cross-validation split.")

        ##########
        # PARAMETERS ABOUT THE TARGET
        ##########
        self.balance_target = parsed_args.balance_target
        if self.balance_target:
            assert self.experiment_data in (ExperimentData.scarim, ExperimentData.scarom), \
                "Target balancing only applies to sets right now."
            assert self.shuffle, "Indices must be shuffled when --balance_target is True"
        self.scale_target = parsed_args.scale_target

        ##########
        # PARAMETERS THAT CONTROL DATA PICKLING
        ##########
        self.pickle_by = PickleBy[parsed_args.pickle_by.lower()]
        if (self.pickle_by == PickleBy.torch and self.experiment_data not in (ExperimentData.scarim, ExperimentData.scarom)) \
            or self.pickle_by == PickleBy.numpy:
            raise NotImplementedError(f"Haven't implemented pickling by {self.pickle_by.name} for {self.experiment_data.name} yet.")

    def _set_default_canonical_function_seeds(self):
        for ordering, default_rs in zip((self.pre_ordering, self.ordering),(Constants.PRE_CANONICAL_RANDOM_STATE, Constants.ORDERING_RANDOM_STATE)):
            if 'seed' not in ordering.canon_kwargs:
                if any((ordering.canon[ss] == Canonical_Order.random for ss in ('train', 'val', 'test'))):
                    ordering.canon_kwargs['seed'] = default_rs
            else:
                assert any((ordering.canon[ss] == Canonical_Order.random for ss in ('train', 'val', 'test'))), \
                    "Random state for random permuting is specified, but no ordering is random."

    #
    # Functions to help build a full data path
    #  > Relies on having a "data ID string" for the specific experiment (see below)
    #  > Adds other details like CV splits
    def get_data_path(self):
        """
        Get data path from an instantiated object
        :return: The path of .pkl / .pt file saving data and targets
        """
        assert isinstance(self.experiment_data, ExperimentData)

        # prefix with "experimentName_data" for the CSL tasks, else just experimentName
        if self.experiment_data in [ExperimentData.rp_paper, ExperimentData.customized]:
            data_filename = "_".join([self.experiment_data.name, "data"])
        else:
            data_filename = self.experiment_data.name

        # Add info about dataset splitting
        #   * Note that self.hold_out_default_test is not a must for distinguishable dataset filenames
        #   * The default testing data (and/or indices) will be added to the dataset if self.hold_out_default_test
        #   * is True, which has no impact on the training and validation sets. If we need to evaluate models on the
        #   * default testing set and the existed dataset file does not contains the default testing data/indices,
        #   * just run create_data.py again with --hold_out_default_test.
        data_id_string = self.get_data_id_string()
        n_splits_string = str(self.n_splits)
        if self.use_default_train_val_split:
            n_splits_string += "_default"
        if self.shuffle:
            n_splits_string += "_" + str(self.shuffle_random_state)

        # Add info on whether y was rebalanced or scaled.
        balance_y_string = "bal" if self.balance_target else ""

        #     * Irrelevant for non-regression tasks, but we don't have that info at this stage of the code
        #     * Ignoring this would involve updating the ID string **after** data creation which is NOT elegant
        #     * Let's just ignore
        scale_y_string = "T" if self.scale_target else "F"

        # Each time a new feature is added to data_filename, it should also be added to _data_params() in Config
        if self.experiment_data != ExperimentData.rp_paper:
            data_filename = "_".join([data_filename, data_id_string, n_splits_string, scale_y_string, balance_y_string])

        # Save the file with extension corresponding to the pickling methods
        if self.pickle_by == PickleBy.pickle:
            file_extension = ".pkl"
        elif self.pickle_by == PickleBy.torch:
            file_extension  = ".pt"
        else:
            raise NotImplementedError(f"Haven't implemented pickling by {self.pickle_by.name} yet.")

        return self.data_dir + data_filename + file_extension

    @staticmethod
    def add_additional_id_info(data_id_string, data_function, swapping_scheme, swapped, ordering, sparse=None,
                               pre_ordering=None, noise=None, function_args=None):
        if sparse is not None:
            assert isinstance(sparse, bool)
            data_id_string += "_sparse" if sparse else "_dense"

        if data_function is not None:
            data_id_string += f"_{data_function.name}"

            if function_args is not None:
                assert isinstance(function_args, dict)
                for kk, vv in function_args.items():
                    data_id_string += f"_{vv}"

        if swapping_scheme is not None:
            data_id_string += f"_{swapping_scheme.name}"

        if swapped:
            data_id_string += "_swapped"

        if pre_ordering is not None:
            data_id_string += Canonical_Orderings.get_ordering_info(pre_ordering, False)

        if noise is not None:
            data_id_string += Random_Noise.get_noise_info(noise)

        data_id_string += Canonical_Orderings.get_ordering_info(ordering, True)

        return data_id_string

    @abstractmethod
    def get_data_id_string(self):
        """
        Get data id string
        :return: The data id string representing the configurations of data
        """
        pass


if __name__ == '__main__':
    args = DataArgParser(args=['hello'])  # Should get not instantiatiable error