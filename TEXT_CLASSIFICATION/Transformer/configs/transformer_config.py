from argparse import ArgumentParser

from common.helper import print_stats
from my_common.my_helper import enum_sprint, is_positive_int
from configs.model_config import ModelArgParser
from util.constants import PE_Type, Pooling


class TransformerArgParser(ModelArgParser):

    @classmethod
    def add_args(cls, parser):
        ##########
        # PARAMETERS THAT CONTROL GIN MODELS
        ##########
        assert isinstance(parser, ArgumentParser)

        group = parser.add_argument_group('Transformer', 'HyperParameters for Transformer')

        # Positional encoding
        group.add_argument('-trans_pt', '--trans_pe_type', type=str, default='ape',
                           help=f"Type of PE used by Transformer. One of {enum_sprint(PE_Type)}")
        group.add_argument('--trans_small_pe', default=False, action='store_true',
                           help="Use a relatively small PE (compared to word embeddings) by removing the scaling factor.")
        # Pooling function
        group.add_argument('-trans_pl', '--trans_pooling', type=str, default='last_dim',
                           help=f"Type of pooling function. One of {enum_sprint(Pooling)}")
        # Positional encoding (using default from the 'complex order' paper; 6 in Transformer Paper)
        group.add_argument('-trans_nl', '--trans_num_layers', type=int, default=1,
                           help="Number of encoder layers.")
        # Hidden model dimension (using default from the 'complex order' paper; 512 in Transformer Paper)
        group.add_argument('-trans_dm', '--trans_dim_model', type=int,
                           default=256,
                           help="Dimension of the hidden layer in transformer")
        # Hidden dimension of feedforward layer (using default from the 'complex order' paper; 2048 in Transformer Paper)
        group.add_argument('-trans_dff', '--trans_dim_ff', type=int,
                           default=512,
                           help="Dimension of the hidden layer in transformer")
        # Attention heads (using default from the 'complex order' paper)
        group.add_argument('-trans_nhd', '--trans_num_heads', type=int,
                           default=8,
                           help="Number of attention heads in transformer")
        # Dropout (using default from the 'complex order' paper)
        group.add_argument('-trans_dp', '--trans_dropout', type=float,
                           default=0.1,
                           help="Dropout probability")
        group.add_argument('--trans_dont_dropout_input', default=False, action='store_true',
                           help="Don't dropout input, i.e., word embedding + PE.")
        group.add_argument('--trans_dont_mask_padding', default=False, action='store_true',
                           help="Don't mask padding.")
        # Layer Normalization (using default from the 'complex order' paper)
        # FIXME: this argument takes no effect right now
        group.add_argument('--trans_dont_use_layer_norm', default=False, action='store_true',
                           help="Do not use layer normalization")

        # # Small architecture
        # group.add_argument('--use_small_transformer', default=False, action='store_true',
        #                    help='Use small transformer')
        # # Use default hyperparameters for the small transformer (from Set Transformer's code)
        # group.add_argument('--use_default_small_transformer', default=False, action='store_true',
        #                    help='Use small transformer with default dim_hidden and num_heads')

    def assign_parsed(self, args):
        """ Set params that control model """

        self.trans_pe_type = PE_Type[args.trans_pe_type.lower()]
        self.trans_small_pe = args.trans_small_pe
        self.trans_pooling = Pooling[args.trans_pooling.lower()]
        self.trans_num_layers = args.trans_num_layers
        self.trans_dim_model = args.trans_dim_model
        self.trans_dim_ff = args.trans_dim_ff
        self.trans_num_heads = args.trans_num_heads
        self.trans_dropout = args.trans_dropout
        self.trans_dropout_input = not args.trans_dont_dropout_input
        self.trans_mask_padding = not args.trans_dont_mask_padding
        self.trans_layer_norm = not args.trans_dont_use_layer_norm
        if not self.trans_layer_norm:
            raise NotImplementedError("We haven't implemented Transformer w/o LayerNorm.")
        if self.trans_dropout == 0.:
            assert self.trans_dropout_input == True, "Non-default value of args.trans_dont_dropout_input is specified, " \
                                                     "but dropout is zero."
        for _var in [self.trans_num_layers, self.trans_dim_model, self.trans_dim_ff, self.trans_num_heads]:
            assert is_positive_int(_var)
        assert isinstance(self.trans_dropout, float) and self.trans_dropout >= 0
        for _var in [self.trans_dropout_input, self.trans_layer_norm, self.trans_mask_padding]:
            assert isinstance(_var, bool)
        if self.trans_small_pe:
            assert self.trans_pe_type != PE_Type.none


    def get_model_id_list(self):
        _lst = [self.trans_pe_type.name, self.trans_small_pe, self.trans_pooling.name, self.trans_num_layers,
                self.trans_dim_model, self.trans_dim_ff, self.trans_num_heads, self.trans_dropout]
        if self.trans_dropout > 0.:
            _lst += [self.trans_dropout_input]
        _lst += [self.trans_mask_padding, self.trans_layer_norm]

        return self.add_additional_id_info(_lst)

    def get_class_name(self):
        if self.trans_pe_type in (PE_Type.none, PE_Type.ape):
            return 'Transformer_PE_real'
        else:
            raise ValueError(f"Unrecognized PE type {self.trans_pe_type.name}.")

if __name__ == '__main__':
    pass
    # args = TransformerArgParser('-trans_nl 3 '.strip().split())  # Should get not instantiatiable error
    # print(args.__dict__)
