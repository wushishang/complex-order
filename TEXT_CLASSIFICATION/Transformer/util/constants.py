from enum import Enum

import numpy as np

# Seed used to generate graphs. Generation scheme depends on task.
DEFAULT_RANDOM_GENERATION_SEED = 1  # Do NOT change for this project!  If default, we don't add to filepath.
SMP_NUM_SAMPLES = 10000  # Both train and test have same nsamples
MAX_BATCH = 300  # moved from config; outdated?


class Pooling(Enum):
    last_dim = 1
    sum = 2
    max = 3


class PE_Type(Enum):
    none = 1
    ape = 2


class TC_ModelType(Enum):
    transformer = 1
    fasttext = 2


class TC_ExperimentData(Enum):
    TREC_transformer = 1
    sst2_transformer = 2
    cr = 3
    mpqa = 4
    mr = 5
    subj = 6


TC_OutputSize = {TC_ExperimentData.TREC_transformer: 6, TC_ExperimentData.sst2_transformer: 2,
                 TC_ExperimentData.cr: 2, TC_ExperimentData.mpqa: 2,
                 TC_ExperimentData.mr: 2, TC_ExperimentData.subj: 2}

# Maximum length of sentences in the training set of TREC and SST-2 (following the original code)
# Maximum length of sentences in the whole dataset of CR, MPQA, MR, SUBJ
MaxSenLen = {TC_ExperimentData.TREC_transformer: 53, TC_ExperimentData.sst2_transformer: 37}
                 # TC_ExperimentData.cr: 2, TC_ExperimentData.mpqa: 2,
                 # TC_ExperimentData.mr: 2, TC_ExperimentData.subj: 2}


class Activation(Enum):
    ReLU = 1
    Tanh = 2
    Sigmoid = 3


class LossType(Enum):
    mse = 1
    mae = 2


class GinModelType(Enum):
    regularGin = 1
    dataAugGin = 2
    rpGin = 3


class GinPositionalEncodingType(Enum):
    none = 1
    onehot = 2
    random = 3
    random_embedding = 3


# Some aliases have been added to the Enums for shorter filenames
class ExperimentData(Enum):
    rp_paper = 1
    customized = 2
    smp = 3  # From structural message passing paper
    er_edges = 4  # Erdos Renyi Edges
    ba = 5  # Barabasi-Albert
    ger = 6  # General Erdos Renyi (replaces er_edges but leaving for compatibility)

    scarim = 7  # scalar integer arithmetic
    scalar_int_arm = 7
    scarom = 8  # scalar real continuous arithmetic
    scalar_cont_arm = 8

    br = 9  # 100-brain-networks dataset
    brain = 9

    zinc = 10


class LType(Enum):
    # meaningless: analytic_prime_l = 1
    analytic_int_l = 2
    brute = 3


class ReplicatePermType(Enum):
    sampled = 1
    deterministic = 2


class TaskType(Enum):
    binary_classification = 1
    multi_classification = 2
    regression = 3
    node_classification = 4


class GraphFunction(Enum):
    first_degree = 1
    max_degree = 2
    det_adj = 3


class SetFunction(Enum):
    max = 1
    sum = 2
    first_large = 3


# MType for Bernoulli Data
class BaM(Enum):
    const = 1
    constant = 1
    bern = 2
    bernoulli = 2  # The number of new edges from (1+Bernoulli) after some vertices added: prevent tree, more realistic


# Probability for Erdos-Renyi
class ErP(Enum):
    const = 1
    constant = 1

    rand = 2
    random = 2


class ArmSampleScheme(Enum):
    unif = 1
    uniform = 1


class NoiseType(Enum):
    none = 1
    bern = 2
    bernoulli = 2


class SwappingScheme(Enum):
    first_two = 1


class PickleBy(Enum):
    pickle = 1
    torch = 2
    numpy = 3


class C:
    """Class of plotting constants"""
    search_complete = "search_complete"
    error_type = "error_type"
    phase = "phase"
    total_samples = "total_samples"
    average_samples = "average_samples"
    method = "method"
    order = "order"
    sl = "sl"
    sp = "sp"
    isamp = "isamp"
    rmse = "rmse"
    Tr = "Tr"
    Vl = "Vl"
    train = "train"
    test = "test"
    validation = "validation"
    dt = "dt"
    Te = "Te"
    lr = "lr"
    min = "min"
    mean = "mean"
    black = "black"
    purple = "purple"
    epoch = "epoch"
    total_training_time = "total_training_time"
    ix = "ix"
    mae = "mae"
    nrmse = "nrmse"
    batches = "batches"
    mse = "mse"
    accuracy = "accuracy"
    maccuracy = "maccuracy"
    mawce = "mawce"


class Result(Enum):
    """ For analysis, which type of file? """
    stats = 0
    output = 1


class InputType(Enum):
    set = 1
    graph = 2


class ModelType(Enum):
    gin_like = 1
    lstm = 2
    transformer = 3


class InputEmbedType(Enum):
    web = 1
    word_embedding = 1
    leb = 2
    linear_embedding = 2


class Regularization(Enum):
    """ Type of regularization to use"""
    none = 1
    pw_diff = 2
    perm_grad = 3
    center = 4
    diff_step_center = 4
    edge = 5
    diff_step_edge = 5
    basis = 6
    diff_step_basis = 6
    naive = 7
    diff_step_naive = 7


class RegRepresentation(Enum):
    none = 1
    latent = 2
    pred = 3
    prediction = 3
    pe = 4
    positional_embedding = 4


class Task(Enum):
    sum = 1
    max_xor = 2
    median = 3
    prod_median = 4
    longest_seq = 5
    k_ary_distance = 6
    var = 7

    def vector_task(self):
        return self in {self.k_ary_distance}


class ImportanceSample(Enum):
    none = 0
    is_sequence = 1
    is_permutation = 2
    r_permutation = 3
    os_permutation = 4
    m_permutation = 5
    f_permutation = 6
    hf_permutation = 7


class Model(Enum):
    lstm = 1
    gru = 2
    mlp = 3


class Aggregation(Enum):
    last = 1
    attention = 2
    summation = 3


class Constants:
    # Word embedding and positional embedding standard dev
    # From https://github.com/pytorch/fairseq/blob/master/fairseq/models/fconv.py
    WP_EMBEDDING_STD = 0.1

    PRE_CANONICAL_RANDOM_STATE = 101
    NOISE_RANDOM_STATE = 201
    ORDERING_RANDOM_STATE = 301

    FOLD_COUNT = 6
    GS_INTERVAL = 50  # frequency of dumping a snapshot of the model to disk
    GS_SEQUENCES = 10000  # sequences sampled to compute norm distributions
    GS_PERMUTATION_BASE = 10  # sequences sampled to compute norm distributions across permutations
    GS_PERMUTATION_COUNT = 1000  # permutations sampled of the above sequences

    MNIST_SEQ_LENS = (5 + 5 * np.arange(10)).tolist()
    MNIST_TASKS = [Task.sum, Task.var, Task.max_xor, Task.median, Task.prod_median, Task.longest_seq]

    VECTOR_SEQ_LENS = [100, 200]
    VECTOR_TASKS = [Task.k_ary_distance]
    VECTOR_K = [2, 3]
    VECTOR_D = [2, 5, 8, 10]

    INFERENCE_PERMUTATIONS = 20
    LSTM_HIDDEN = 50
    GRU_HIDDEN = 80
    MLP_HIDDEN = 30

    NORM_TYPE = 2

    NAME_MAX = 255  # filename length limit of Linux systems
    CHECKPOINT_TAIL = ".checkpoint.pth.tar"  # Making it a "constant" will make it easier to shorten this in the future.
    MODEL_ID_MAX = NAME_MAX - len(CHECKPOINT_TAIL)

    # When creating data, print warnings if the targets exceed
    WARNING_MEAN_THRESH = 200.
    WARNING_STD_THRESH = 60.

    # Default hyperparameters for small transformers (from Set Transformer's code)
    SMALL_ST_NUM_INDS = 0  # ISAB is not used in the small transformer
    SMALL_ST_DIM_HIDDEN = 64
    SMALL_ST_NUM_HEADS = 4

    # Data splits used in data loading
    DATA_SPLITS = ['train', 'val', 'test']


    @staticmethod
    def get_batch_limit(nograd=False):
        if nograd:
            return 5000
        else:
            return 30

    @staticmethod
    def get_dataset_size(task: Task):
        # returns n_tr, n_te
        if task.vector_task():
            return 12000, 2000
        else:
            # Supports sequences of less than 50 length
            # Each fold will be 20000 sequences maximum and test_examples will be an additional 20000 sequences
            # return 120000, 20000
            return 1200, 200


class DT(Enum):
    train = 0
    validation = 1
    test = 2
