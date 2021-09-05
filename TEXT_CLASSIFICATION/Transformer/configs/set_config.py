from argparse import ArgumentParser
from collections import OrderedDict

from common import print_stats
from configs.data_config import DataArgParser
from data.Arithmetic_data import ArmData
from data.set_task_functions import SetFunction
from my_common.my_helper import enum_sprint, args_list_to_dict
from util.constants import ExperimentData, ArmSampleScheme


class SetDataArgParser(DataArgParser):

    @classmethod
    def add_args(cls, parser):

        assert isinstance(parser, ArgumentParser)

        group = parser.add_argument_group('Set Data', 'HyperParameters for Set Data')

        # Arbitrary functions: implemented for newer tasks
        group.add_argument('-sf', '--set_function', default=None, type=str,
                           help=f'Set function to compute: {enum_sprint(SetFunction)}')
        group.add_argument('--set_function_args',  nargs="+",
                           help='Arguments for set function, if applicable')
        group.add_argument('--vocab_size', type=int, default=0,
                           help="Size of vocabulary for given task, if applicable (integer arithmetic, language, ..)")
        group.add_argument('--n_sets', type=int, default=0,
                           help="Number of sets in dataset, if applicable (e.g. most synthetic tasks)")
        group.add_argument('--len_sets', type=int, default=0,
                           help="Cardinality of sets, if applicable")
        #
        # Arguments for arithmetic tasks
        #
        group.add_argument('--arm_s_type', type=str, default=ArmSampleScheme.unif.name,
                           help="Sampling scheme for arithmetic task")
        group.add_argument("--arm_s_args", nargs="+",
                           help=f"Arguments for the sample functions in ArmData\n {ArmData.helper()}")

        #
        # General arguments
        #

    def assign_parsed(self, args):
        ##########
        # PARAMETERS THAT CONTROL DATA GENERATION
        ##########
        if args.set_function != self.parser.get_default("set_function"):
            self.set_function = SetFunction[args.set_function]
            self.set_function_args = args_list_to_dict(args.set_function_args)
        else:
            self.set_function = args.set_function  # i.e., None
            self.set_function_args = OrderedDict()
        self.vocab_size = args.vocab_size

        ##########
        # PARAMETERS THAT DEPEND ON THE EXPERIMENT
        ########
        if self.experiment_data in (ExperimentData.scalar_int_arm, ExperimentData.scalar_cont_arm):
            if self.experiment_data == ExperimentData.scalar_int_arm:
                print_stats("Using scalar integer arithmetic tasks.")
            else:
                print_stats("Using scalar continuous arithmetic tasks.")
            assert args.set_function is not None
            if self.balance_target:
                assert args.n_sets % 2 == 0, "Number of sets must be even when --balance_target is True."
            self.arm_config = self.load_arm_args(args)

        else:
            raise ValueError("Unknown experiment data. Please check if it is graph data. If not, then it hasn't been implemented.")

    @staticmethod
    def load_arm_args(args):
        assert args.n_sets > 1, "Num sets should be > 1 (for cv splits)"
        assert args.len_sets > 1, "Len sets should be > 1"
        assert args.vocab_size >= 0, "Vocab size should be >= 0"
        s_kwargs = args_list_to_dict(args.arm_s_args)

        arm_config = OrderedDict()
        arm_config['n_sets'] = args.n_sets
        arm_config['len_sets'] = args.len_sets
        arm_config['vocab_size'] = args.vocab_size
        arm_config['i_type'] = ExperimentData[args.experiment_data]
        arm_config['s_type'] = ArmSampleScheme[args.arm_s_type.lower()]
        arm_config['s_kwargs'] = s_kwargs

        return arm_config

    def get_data_id_string(self):
        """
        Get set id string from an instantiated object
        :return: The set id string representing the configurations of sets
        """
        if self.experiment_data in (ExperimentData.scalar_int_arm, ExperimentData.scalar_cont_arm):
            data_id_string = self.get_arm_style_full_data_id(set_len=self.arm_config['len_sets'],
                                                             n_sets=self.arm_config['n_sets'],
                                                             vocab_size=self.arm_config['vocab_size'],
                                                             i_type=self.arm_config['i_type'],
                                                             s_type=self.arm_config['s_type'],
                                                             s_kwargs=self.arm_config['s_kwargs'],
                                                             random_state=self.sample_random_state,
                                                             set_function=self.set_function,
                                                             swapping_scheme=self.swapping_scheme,
                                                             swapped=self.do_swapping,
                                                             ordering=self.ordering,
                                                             pre_ordering=self.pre_ordering,
                                                             noise=self.noise,
                                                             function_args=self.set_function_args)

        else:
            raise NotImplementedError("Don't know how to get data id string for ExperimentData entered.")

        return data_id_string

    @staticmethod
    def get_arm_style_full_data_id(set_len, n_sets, vocab_size, i_type, s_type, s_kwargs,
                                   random_state, set_function, swapping_scheme, swapped, ordering, pre_ordering, noise,
                                   function_args):
        data_id_string = ArmData.arm_style_data_id(set_len=set_len,
                                                   n_sets=n_sets,
                                                   i_type=i_type,
                                                   s_type=s_type,
                                                   s_kwargs=s_kwargs,
                                                   vocab_size=vocab_size,
                                                   random_state=random_state
                                                   )
        data_id_string = SetDataArgParser.add_additional_id_info(data_id_string,
                                                                 set_function, swapping_scheme, swapped, ordering,
                                                                 pre_ordering=pre_ordering,
                                                                 noise=noise,
                                                                 function_args=function_args)

        return data_id_string


if __name__ == "__main__":
    data_cfg = SetDataArgParser()
    print(data_cfg.get_data_path())
