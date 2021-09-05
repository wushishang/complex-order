import argparse
from abc import ABC, abstractmethod

class ArgParser(ABC):
    """
    Parent class that implements a general argument parser of data/model configurations.

    Children ArgParsers should inherit and implement class methods of adding common arguments shared by different
    data/models (in "add_common_args") as well as that of adding arguments unique to a specific data/model (in "add_args").
    These class methods will be called by Config during training to gather all arguments.

    Children ArgParsers should also implement instance methods "assign_common_parsed" and "assign_parsed"
    that assign parsed values of common arguments and unique arguments respectively to attributes of ArgParsers,
    which will be passed as data/model configs to Config during training or DataCreator during data creation.
    """
    def __init__(self, args=None, parsed_args=None):
        super().__init__()
        self.parser = argparse.ArgumentParser("Argument parser for data/model configurations")
        self.add_common_args(self.parser)
        self.add_args(self.parser)
        if parsed_args is None:
            parsed_args = self.parser.parse_args(args)
        # Assign parsed args to object
        try:
            self.assign_common_parsed(parsed_args)
            self.assign_parsed(parsed_args)
        except AttributeError:
            print("-"*10)
            print("Did you forget a flag?\nDid you add new parser to MODEL_ARG_PARSERS list in config.py?")
            print("-" * 10)
        finally:
            pass  # Automatically prints usual warning message and quits if there's an error.

    @classmethod
    @abstractmethod
    def add_common_args(cls, parser):
        """
        Add common arguments shared among different kinds of data and models (e.g., n_splits, activations) to a parser.
        It is separate from `add_args` to prevent collision of duplicate common arguments.

        :param parser: argparse.ArgumentParser
        """
        pass

    @classmethod
    @abstractmethod
    def add_args(cls, parser):
        """
        Add arguments unique to a specific data/model (e.g., graph_function, num_gnn_layers) to a parser.

        :param parser: argparse.ArgumentParser
        """
        pass

    @abstractmethod
    def assign_common_parsed(self, parsed_args):
        """
        Assign parsed common arguments shared among different kinds of data and models (e.g., n_splits, activations)
        to the attributes of the ArgParser.

        :param parsed_args: argparse.NameSpace
        """
        pass

    @abstractmethod
    def assign_parsed(self, parsed_args):
        """
        Assign parsed arguments unique to the specific data/model (e.g., graph_function, num_gnn_layers)
        to the attributes of the ArgParser.

        :param parsed_args: argparse.NameSpace
        """
        pass


if __name__ == '__main__':
    args = ArgParser(args=['hello'])  # Should get not instantiatiable error