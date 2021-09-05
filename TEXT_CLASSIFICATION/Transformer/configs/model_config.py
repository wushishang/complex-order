from abc import abstractmethod
from argparse import ArgumentParser, Namespace

from common.helper import Util
from configs.config_argparser import ArgParser
from util.constants import Activation
from my_common.my_helper import enum_sprint


class ModelArgParser(ArgParser):

    @classmethod
    def add_common_args(cls, parser):
        assert isinstance(parser, ArgumentParser)

        parser.add_argument('-act', '--activation_function', default=Activation.ReLU.name, type=str,
                            help=f'Activation functions used in models. One of {enum_sprint(Activation)}')

    def assign_common_parsed(self, parsed_args):
        assert isinstance(parsed_args, Namespace)

        self.act = Activation[parsed_args.activation_function]

    def get_common_args_info(self, format='list'):
        """
        Get info of common model arguments. Two formats are available: list or dict.
        """
        assert isinstance(format, str)

        if format == 'list':
            return [self.act.name]
        elif format == 'dict':
            return Util.dictize(activation = self.act.name)
        else:
            raise ValueError(f"Unrecognized format: {format}.")

    def add_additional_id_info(self, model_id_list):
        """
        Add additional info to specific model id list
        """
        assert isinstance(model_id_list, list)

        # Add common arguments
        return model_id_list + self.get_common_args_info()

    @abstractmethod
    def get_model_id_list(self):
        """
        Get model id list
        :return: The model id list representing the configurations of models
        """
        pass

if __name__ == '__main__':
    args = ModelArgParser(args=['hello'])  # Should get not instantiatiable error