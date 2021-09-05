from argparse import ArgumentParser
from collections import OrderedDict

from Create_Brain_Network_Data import NUM_CONNECTOMES, BR_FILE_NAME
from data.ZINC import NUM_UNIQUE_NODE_TYPES_COMPLETE, NUM_UNIQUE_NODE_TYPES_SUBSET
from data.BA_data import BaData
from data.er_data import ErData
from data.graph_task_functions import GraphFunction
from util.constants import ExperimentData, LType, ReplicatePermType, DEFAULT_RANDOM_GENERATION_SEED, BaM, ErP
from common import PickleUtil, print_stats
from my_common.my_helper import enum_sprint, args_list_to_dict
from configs.data_config import DataArgParser, PROCESSED_DATA_DIR
from data.Create_CSL import Synthetic_Graphs, SYNTHETIC_DIR


SMP_DATA_DIR = f'{SYNTHETIC_DIR}raw_smp/'

PickleUtil.check_create_dir(PROCESSED_DATA_DIR)
PickleUtil.check_create_dir(SMP_DATA_DIR)

SYNTHETIC_GRAPHS = PROCESSED_DATA_DIR + 'graphs_Kary_Deterministic_Graphs.pkl'
SYNTHETIC_Y = PROCESSED_DATA_DIR + "y_Kary_Deterministic_Graphs.pt"
SYNTHETIC_X_UNITY = PROCESSED_DATA_DIR + "X_unity_list_Kary_Deterministic_Graphs.pkl"
SYNTHETIC_X_EYE = PROCESSED_DATA_DIR + "X_eye_list_Kary_Deterministic_Graphs.pkl"
RP_PAPER_DATA = PROCESSED_DATA_DIR + "rp_paper_data.pkl"
BR_DATA = PROCESSED_DATA_DIR + BR_FILE_NAME

SMP_SINGLE_GRAPH_SIZE = {72: 64}  # For each SMP graph with n = KEY, take only graphs of size VALUE


class GraphDataArgParser(DataArgParser):

    @classmethod
    def add_args(cls, parser):

        assert isinstance(parser, ArgumentParser)

        group = parser.add_argument_group('Graph Data', 'HyperParameters for Graph Data')

        # Arbitrary functions: implemented for newer tasks
        group.add_argument('-gf', '--graph_function', default=None,
                            help=f'Graph function to compute: {enum_sprint(GraphFunction)}')

        #
        # Arguments for CSL tasks
        #
        group.add_argument('-N', '--num_vertices', default=41, type=int, help='Number of vertices')
        group.add_argument('-L', '--num_skip_lengths', default=10, type=int, help='Number of distinct skip lengths')
        group.add_argument('-np', '--num_permutations', default=15, type=int,
                            help='Number of permutations for each skip length')
        group.add_argument('-lty', "--l_type", default="", type=str,
                            help=f'Type of L creation: {enum_sprint(LType)}')
        group.add_argument('-ps', '--permutation_strategy', default="",
                            help=f'Scheme for creating more graphs within same class via permuting adjmat: {enum_sprint(ReplicatePermType)}')
        group.add_argument('--relax_N_prime', action='store_true', help='Allows graphs of non-prime size')
        #
        # Arguments for SMP tasks
        #
        group.add_argument('-smp_k', '--smp_cycle_size', default=0, type=int, help='Size of cycles for smp task')
        group.add_argument('-smp_n', '--smp_n_vert', default=0, type=int, help='Average size of graph `n` for smp')
        group.add_argument('-smp_s', '--smp_set', default="", type=str,
                            help='Whether to use Train or Test set for smp task')
        group.add_argument('-smp_p', '--smp_prop', default=0.0, type=float,
                            help='Proportion of data to take for smp task')
        group.add_argument('--smp_var_size', action='store_true',
                            help='Use varying sizes for the SMP dataset')
        #
        # Arguments for ZINC tasks
        #
        group.add_argument('--zinc_complete', action='store_true', help='Use complete ZINC dataset (default: ZINC-12k)')
        group.add_argument('--zinc_multigraph', action='store_true', help='Use multiple adjacency matrices corresponding \
                           to different bond types (i.e., categorical edge features).')
        #
        # Arguments for Erdos-Renyi edges
        # (( This should be deprecated, use Generalized below ))
        # (( Keeping for legacy ))
        group.add_argument('-er_n', '--er_edges_n_vert', default=750, type=int,
                            help='Number of vertices in erdos-renyi-edges task.')
        group.add_argument('-er_N', '--er_edges_n_graphs', default=500, type=int,
                            help='Number of (same-sized) graphs for erdos-renyi-edges task.')
        group.add_argument('-er_b1', '--er_edges_beta_1', default=1.0, type=float,
                            help='First beta-distribution parameter for erdos-renyi-edges task.')
        group.add_argument('-er_b2', '--er_edges_beta_2', default=1.0, type=float,
                            help='Second beta-distribution parameter for erdos-renyi-edges task.')
        group.add_argument('--er_edges_dont_scale_targets', default=False, action='store_true',
                            help='If set, we do NOT center-and-scale the targets of the erdos-renyi-edges task.')
        group.add_argument('--er_edges_seed', default=DEFAULT_RANDOM_GENERATION_SEED, type=int,
                            help='Seed for erdos-renyi-edges. Can override w/ sample_random_state')  # bw compatible
        #
        # Barabasi-Albert
        #
        group.add_argument('-ba_n', '--ba_n_vert', type=int, default=-1,
                            help="Number of vertices in BA task, set > 2 if using BA.")
        group.add_argument('-ba_N', '--ba_n_graphs', type=int, default=-1,
                            help="Number of graphs in BA task")
        group.add_argument('--ba_m_type', default=None,
                            help=f"General strategy for determining the `m` argument in BA: {enum_sprint(BaM)}")
        group.add_argument("--ba_m_args", nargs="+",
                            help=f"Arguments for the m_list functions in BaData\n {BaData.helper()}")
        group.add_argument("--ba_ba_args", nargs="+",
                            help=f"Additional KW Arguments for Barabasi Albert generation beyond n and m")

        #
        # General Erdos-Renyi
        #
        group.add_argument('-ger_n', '--ger_n_vert', type=int, default=-1,
                            help="Number of vertices in BA task, set > 2 if using BA.")
        group.add_argument('-ger_N', '--ger_n_graphs', type=int, default=-1,
                            help="Number of graphs in BA task")
        group.add_argument('--ger_p_type', default=None,
                            help=f"General strategy for determining the prob argument in ER: {enum_sprint(ErP)}")
        group.add_argument("--ger_p_args", nargs="+",
                            help=f"Arguments for the probability functions in ErData\n {ErData.helper()}")

        #
        # 100-Brain-networks
        #
        group.add_argument('-br_type', '--br_processing_type', default='spanningtree', help=f"Processing strategy for "
                           f"determining the adjacency matrices of Brain Networks: spanningtree (default) / threshold")
        group.add_argument('-br_n_test', '--br_n_test', type=int, default=50,
                           help=f"Number of samples to hold out for testing. Min: 0, Max: {NUM_CONNECTOMES}, default: 50.")

        #
        # General arguments
        #
        group.add_argument('-sps', "--sparse", action='store_true',
                            help='Use coo sparse adjacency matrices (default: dense ndarrays)')

    def assign_parsed(self, args):
        ##########
        # PARAMETERS THAT CONTROL DATA GENERATION
        ##########
        if args.graph_function != self.parser.get_default("graph_function"):
            self.graph_function = GraphFunction[args.graph_function]
        else:
            self.graph_function = None

        ##########
        # PARAMETERS THAT DEPEND ON THE EXPERIMENT
        ########
        self.sparse = args.sparse
        if self.experiment_data in (ExperimentData.rp_paper, ExperimentData.customized):
            self.num_vertices = args.num_vertices
            self.num_skip_lengths = args.num_skip_lengths
            self.num_permutations = args.num_permutations
            self.permutation_strategy = args.permutation_strategy
            self.force_N_prime = not args.relax_N_prime

            assert self.num_vertices > 9, "N must be an integer greater than or equal to 10"
            assert 1 < self.num_skip_lengths < self.num_vertices - 1, "L must be between 2 and N-1"
            assert 1 < self.num_permutations <= 10000, \
                   "Number of permutations for the same isomorphism class must be between 2 and 10000"

            if self.experiment_data == ExperimentData.customized:
                assert args.l_type in LType.__members__
                self.l_type = LType[args.l_type]

                assert self.permutation_strategy in ReplicatePermType.__members__
                self.permutation_strategy = ReplicatePermType[self.permutation_strategy]

                if self.permutation_strategy == ReplicatePermType.sampled:
                    assert self.sample_random_state > 0
                    sample_permutations = True
                    print_stats("Adjacency matrices will be permuted with random permutations.")
                elif self.permutation_strategy == ReplicatePermType.deterministic:
                    self.sample_random_state = None
                    sample_permutations = False
                    print_stats("Adjacency matrices will be permuted with deterministic permutations.")

            elif self.experiment_data == ExperimentData.rp_paper:  # For legacy
                # Make sure the graphs are exactly as they were in the RP-Paper

                assert self.num_vertices == self.parser.get_default('num_vertices') \
                       and self.num_skip_lengths == self.parser.get_default('num_skip_lengths') \
                       and self.num_permutations == self.parser.get_default('num_permutations') \
                       and self.n_splits == self.parser.get_default('n_splits'), \
                       "Inconsistent specification with --experiment_data = rp_paper"

                print_stats("Forcing L to brute force L and sparse adjacencies for legacy compatibility")
                self.l_type = LType.brute
                self.sparse = True
                sample_permutations = False

            self.graph_generator = Synthetic_Graphs(self.num_vertices, self.num_skip_lengths, self.num_permutations,
                                                    self.l_type, self.sparse, sample_permutations,
                                                    self.sample_random_state, self.force_N_prime)

        elif self.experiment_data == ExperimentData.smp:
            self.smp_config = self.load_smp_args(args)

        elif self.experiment_data == ExperimentData.er_edges:
            raise ValueError("Don't use er_edges, use generalized ER (ger) scheme instead.  TODO: delete later.")
            # self.er_edges_config = self.load_er_edges_args(args)

        elif self.experiment_data == ExperimentData.ba:
            assert args.graph_function is not None
            self.ba_config = self.load_ba_args(args)

        elif self.experiment_data == ExperimentData.ger:
            print_stats("Using generalized Erdos-Renyi (ger) tasks.")
            assert args.graph_function is not None
            self.ger_config = self.load_ger_args(args)

        elif self.experiment_data == ExperimentData.br:
            print_stats("Using 100-Brain_networks Dataset for node classification task.")
            self.br_config = self.load_br_args(args)

        elif self.experiment_data == ExperimentData.zinc:
            print_stats("Using ZINC for graph regression task.")
            self.zinc_config = self.load_zinc_args(args)
            # Vocab_size is the number of node features, which could be used as the default word embedding dim
            # Plus 1 in case padding is used for variable-size graphs.
            if self.zinc_config['subset']:
                self.vocab_size = NUM_UNIQUE_NODE_TYPES_SUBSET + 1
            else:
                self.vocab_size = NUM_UNIQUE_NODE_TYPES_COMPLETE + 1

        else:
            raise ValueError("Unknown experiment data. Please check if it is set data. If not, then it hasn't been implemented.")

    @staticmethod
    def load_smp_args(args):
        assert args.smp_cycle_size > 0 and args.smp_n_vert > 0, "Invalid int values for smp experiment"
        assert args.smp_set.lower() in ('train', 'test')

        # Store smp params as a dictionary
        # Order is important so the filenames are always consistent.
        smp_config = OrderedDict()
        smp_config['cycle_size'] = args.smp_cycle_size
        smp_config['n_vert'] = args.smp_n_vert
        smp_config['set'] = args.smp_set.lower()
        smp_config['var_size'] = args.smp_var_size

        if smp_config['var_size']:
            print_stats("Taking SMP graphs of variable sizes")
            assert 0 < args.smp_prop <= 1
            smp_config['prop'] = args.smp_prop
        else:
            print_stats("Taking SMP graphs of one fixed size, --smp_prop is ignored.")
            print_stats(f"Fixed size is determined by SMP_SINGLE_GRAPH_SIZE, {SMP_SINGLE_GRAPH_SIZE}")
            smp_config['prop'] = 1.0  # Not a dummy value, used later.

        return smp_config

    @staticmethod
    def load_er_edges_args(args):
        assert args.er_edges_n_vert > 1, "er_edges_n_vert should be an integer greater than 1"
        assert args.er_edges_n_graphs > 1, "er_edges_n_graphs should be an integer greater than 1"
        assert args.er_edges_beta_1 > 0., "er_edges_beta_1 parameter should be strictly positive."
        assert args.er_edges_beta_2 > 0., "er_edges_beta_2 parameter should be strictly positive."
        assert isinstance(args.er_edges_dont_scale_targets, bool)

        # Store smp params as a dictionary
        # Order is important so the filenames are always consistent.
        er_edges_config = OrderedDict()
        er_edges_config['n_vert'] = args.er_edges_n_vert
        er_edges_config['n_graphs'] = args.er_edges_n_graphs
        er_edges_config['beta_1'] = args.er_edges_beta_1
        er_edges_config['beta_2'] = args.er_edges_beta_2
        er_edges_config['scale_targets'] = not args.er_edges_dont_scale_targets

        # Set seed, allow args.sample_random_state to override er_edges seed if available
        # All this is done for legacy purposes as the code evolved...
        if args.sample_random_state != DEFAULT_RANDOM_GENERATION_SEED:
            er_edges_config['seed'] = args.sample_random_state
        else:
            er_edges_config['seed'] = args.er_edges_seed

        return er_edges_config

    @staticmethod
    def load_ba_args(args):
        assert args.ba_m_type is not None
        assert args.ba_n_vert > 2
        assert args.ba_n_graphs > 2

        m_kwargs = args_list_to_dict(args.ba_m_args)
        ba_kwargs = args_list_to_dict(args.ba_ba_args)

        ba_config = OrderedDict()
        ba_config['n_vert'] = args.ba_n_vert
        ba_config['n_graphs'] = args.ba_n_graphs
        ba_config['m_type'] = BaM[args.ba_m_type]
        ba_config['m_kwargs'] = m_kwargs
        ba_config['ba_kwargs'] = ba_kwargs

        return ba_config

    @staticmethod
    def load_ger_args(args):
        assert args.ger_n_vert > 2, "Num vertices should be > 2"
        assert args.ger_n_graphs > 2, "Num vertices should be > 2"
        assert args.ger_p_type in ErP.__members__

        p_kwargs = args_list_to_dict(args.ger_p_args)

        ger_config = OrderedDict()
        ger_config['n_vert'] = args.ger_n_vert
        ger_config['n_graphs'] = args.ger_n_graphs
        ger_config['p_type'] = ErP[args.ger_p_type]
        ger_config['p_kwargs'] = p_kwargs

        return ger_config

    @staticmethod
    def load_br_args(args):
        assert 0 <= args.br_n_test <= NUM_CONNECTOMES, "Num of hold-out connectomes should fall in [0, 400]"
        assert args.br_processing_type in ("spanningtree", "threshold"), \
            "Processing type must be one of ('spanningtree', 'threshold')"

        br_config = OrderedDict()
        br_config['n_test'] = args.br_n_test
        br_config['processing_type'] = args.br_processing_type

        return br_config

    @staticmethod
    def load_zinc_args(args):
        zinc_config = OrderedDict()
        zinc_config['subset'] = not args.zinc_complete
        zinc_config['multigraph'] = args.zinc_multigraph

        return zinc_config

    def get_data_id_string(self):
        """
        Get graph id string from an instantiated object
        :return: The graph id string representing the configurations of graphs
        """
        if self.experiment_data in (ExperimentData.rp_paper, ExperimentData.customized):
            graph_id_string = self.graph_generator.get_graph_id()
        elif self.experiment_data == ExperimentData.smp:
            graph_id_string = self.get_data_id_from_dict(self.smp_config, self.sparse)
        elif self.experiment_data == ExperimentData.er_edges:
            graph_id_string = self.get_data_id_from_dict(self.er_edges_config, self.sparse)
        elif self.experiment_data == ExperimentData.ba:
            graph_id_string = self.get_ba_style_full_graph_id(
                                                              graph_size=self.ba_config['n_vert'],
                                                              n_graphs=self.ba_config['n_graphs'],
                                                              m_type=self.ba_config['m_type'],
                                                              m_kwargs=self.ba_config['m_kwargs'],
                                                              ba_kwargs=self.ba_config['ba_kwargs'],
                                                              random_state=self.sample_random_state,
                                                              sparse=self.sparse,
                                                              graph_function=self.graph_function,
                                                              swapping_scheme=self.swapping_scheme,
                                                              swapped=self.do_swapping,
                                                              ordering=self.ordering
                                                        )
        elif self.experiment_data == ExperimentData.ger:
            graph_id_string = self.get_ger_style_full_graph_id(
                                                              graph_size=self.ger_config['n_vert'],
                                                              n_graphs=self.ger_config['n_graphs'],
                                                              p_type=self.ger_config['p_type'],
                                                              p_kwargs=self.ger_config['p_kwargs'],
                                                              random_state=self.sample_random_state,
                                                              sparse=self.sparse,
                                                              graph_function=self.graph_function,
                                                              swapping_scheme=self.swapping_scheme,
                                                              swapped=self.do_swapping,
                                                              ordering=self.ordering
                                                        )
        elif self.experiment_data == ExperimentData.br:
            graph_id_string = DataArgParser.add_additional_id_info(self.get_data_id_from_dict(self.br_config, self.sparse),
                                                                   data_function=self.graph_function,
                                                                   swapping_scheme=self.swapping_scheme,
                                                                   swapped=self.do_swapping,
                                                                   ordering=self.ordering)
        elif self.experiment_data == ExperimentData.zinc:
            graph_id_string = DataArgParser.add_additional_id_info(self.get_data_id_from_dict(self.zinc_config, self.sparse),
                                                                   data_function=self.graph_function,
                                                                   swapping_scheme=self.swapping_scheme,
                                                                   swapped=self.do_swapping,
                                                                   ordering=self.ordering)

        else:
            raise NotImplementedError("Don't know how to get graph id string for ExperimentData entered.")

        return graph_id_string

    @staticmethod
    def get_ba_style_full_graph_id(graph_size, n_graphs, m_type, m_kwargs, ba_kwargs, random_state,
                                   sparse, graph_function, swapping_scheme, swapped, ordering):
        graph_id_string = BaData.ba_style_graph_id(
            graph_size=graph_size,
            n_graphs=n_graphs,
            m_type=m_type,
            m_kwargs=m_kwargs,
            ba_kwargs=ba_kwargs,
            random_state=random_state
        )
        graph_id_string = GraphDataArgParser.add_additional_id_info(graph_id_string, graph_function,
                                                                    swapping_scheme, swapped, ordering, sparse)

        return graph_id_string

    @staticmethod
    def get_ger_style_full_graph_id(graph_size, n_graphs, p_type, p_kwargs, random_state,
                                   sparse, graph_function, swapping_scheme, swapped, ordering):
        graph_id_string = ErData.er_style_graph_id(
            graph_size=graph_size,
            n_graphs=n_graphs,
            p_type=p_type,
            p_kwargs=p_kwargs,
            random_state=random_state
        )
        graph_id_string = GraphDataArgParser.add_additional_id_info(graph_id_string, graph_function,
                                                                    swapping_scheme, swapped, ordering, sparse)

        return graph_id_string

    #
    # Functions to build an experiment-specific string that describes the experiment,
    # but not other details like number of CV splits
    @staticmethod
    def get_data_id_from_dict(config_dict, sparse):
        """
        Base method for creating a Graph ID from a dictionary, by using keys and values
        """
        assert isinstance(config_dict, OrderedDict)
        assert isinstance(sparse, bool)
        out_id = ""
        for kk, vv in config_dict.items():
            out_id += f"{vv}_"

        out_id += "sparse" if sparse else "dense"
        return out_id


if __name__ == "__main__":
    data_cfg = GraphDataArgParser()
    print(data_cfg.get_data_path())
