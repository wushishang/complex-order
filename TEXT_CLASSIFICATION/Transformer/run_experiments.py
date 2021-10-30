from itertools import chain, combinations
import numpy

TRANSFORMER_TWO_TASKS = False
PI_TRANSFORMER_HP_TUNING = False
PS_TRANSFORMER_HP_TUNING = False
PS_TRANSFORMER_SMALL_PE = False
UNREG_TRANSFORMER_PATIENCE = False
REG_BEST_TRANSFORMER_PATIENCE = False
REG_BEST_TRANSFORMER_PATIENCE_TESTING = False
REG_PS_TRANSFORMER_PATIENCE = False
REG_PS_TRANSFORMER_PATIENCE_TESTING = False
REG_PS_TRANSFORMER_PATIENCE_TRAIN_ON_SORT_UP_RANDOM = False
REG_PS_TRANSFORMER_PATIENCE_TRAIN_ON_SORT_UP_RANDOM_TESTING = False

REG_PS_TRANSFORMER_PATIENCE_OTHER_FOUR = True
REG_PS_TRANSFORMER_PATIENCE_OTHER_FOUR_TESTING = False

SYNTAX_CHECK = False
DRYRUN = True

EXP_DATE = "_oct_16_2021" #   "_sept_19_2021"
CLUSTER = "_ml00"  #   "_gilbreth"
QUEUE_NAME = "RP_REG_wu1396" + CLUSTER + EXP_DATE
SRC_FOLDER = "complex-order" + EXP_DATE + "/TEXT_CLASSIFICATION/Transformer" # "rpgin_reg_mar_27"
PYTHON_MAINFILE = "train.py"
USER = "wu1396"


def powerset(iterable):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))

def _get_command_tuple(base_args, arg_dict, binary_flags=()):
    args = " ".join([f"-{x} {y}" for x, y in arg_dict.items()])
    if len(binary_flags) > 0:
        args += " " + " ".join([f"--{bf}" for bf in binary_flags])
    src_dir = f"/##SCRATCH##/{USER}/{SRC_FOLDER}"
    cmd = f"./##SCRIPT## ##GPU## {PYTHON_MAINFILE} {base_args} {args}"
    op_file = "nohup." + ".".join(map(str, arg_dict.values())).replace(" ", ".")
    if len(binary_flags) > 0:
        op_file += "." + ".".join(binary_flags).replace(" ", ".")
    op_file += ".out"
    if len(binary_flags) > 0:
        arg_dict.update(dict(zip(binary_flags, (True,)*len(binary_flags))))
    return src_dir, cmd, op_file, arg_dict


if TRANSFORMER_TWO_TASKS:

    # data config
    data = ""

    # model_config
    model = "-mt Transformer "

    # training config
    train = "-ne 100 "

    # binary_flags = ["use_batchnorm"]  # "use_mini_batching -bs 32"

    TASKS = [
        _get_command_tuple(data + model + train, {"data": _data, "trans_pt": _trans_pt, "trans_pl": _trans_pl, "sv":_sv})
        for _data in ['TREC_transformer', 'sst2_transformer']  #  'cr', 'mpqa', 'mr', 'subj'
        for _trans_pt in ['none', 'ape']
        for _trans_pl in ['sum', 'max']  # 'last_dim',
        for _sv in range(133337, 133347)

        # for _bf in powerset(binary_flags)
    ]


elif PI_TRANSFORMER_HP_TUNING:

    # data config
    data = ""

    # model_config
    model = "-mt Transformer "

    # training config
    train = ""

    # binary_flags = ["use_batchnorm"]  # "use_mini_batching -bs 32"

    TASKS = [
        _get_command_tuple(data + model + train, {"data": _data, "ne":_ne, "lr": _lr, "bs": _bs, "trans_pt": _trans_pt,
                                                  "trans_pl": _trans_pl, "trans_dp": _trans_dp,  "sv":_sv})
        for _data in ['TREC_transformer', 'sst2_transformer']  # , , 'cr', 'mpqa', 'mr', 'subj'
        for _ne in [100, 200, 400]  #  800
        for _lr in [0.001, 0.0001, 0.00001]
        for _bs in [32, 64, 128]
        for _trans_pt in ['none']
        for _trans_pl in ['last_dim']
        for _trans_dp in [0., 0.1, 0.5]
        for _sv in range(133337, 133347)

        # for _bf in powerset(binary_flags)
    ]


elif PS_TRANSFORMER_HP_TUNING:

    # data config
    data = ""

    # model_config
    model = "-mt Transformer "

    # training config
    train = ""

    binary_flags = ["original_mode"]  # "use_mini_batching -bs 32"

    TASKS = [
        _get_command_tuple(data + model + train, {"data": _data, "ne":_ne, "lr": _lr, "bs": _bs, "trans_pt": _trans_pt,
                                                  "trans_pl": _trans_pl, "trans_dp": _trans_dp,  "sv":_sv}, _bf)
        for _data in ['TREC_transformer', 'sst2_transformer']  # 'TREC_transformer', 'sst2_transformer', , 'cr', 'mpqa', 'mr', 'subj'
        for _ne in [100, 200, 400]  #  800
        for _lr in [0.001, 0.0001, 0.00001]
        for _bs in [32, 64, 128]
        for _trans_pt in ['ape']
        for _trans_pl in ['last_dim']
        for _trans_dp in [0., 0.1, 0.5]
        for _sv in range(133337, 133347)

        for _bf in powerset(binary_flags)
    ]


elif SYNTAX_CHECK:

    # data config
    data = "-it set "

    # model_config
    model = "-mt Transformer "

    # training config
    train = "-pi 0 -lrrl 1 -lrrc 2 "

    dummy = 'it set '  # Do not include the minus! (-)cd

    reg_base = "-r_repr pred -tpp -ppe "

    binary_flags = ["random_segment"]  # "use_mini_batching -bs 32"

    TASKS = [
        _get_command_tuple(data + model + train, {"data": _data, "ne":_ne, "lr": _lr, "bs": _bs, "trans_pt": _trans_pt,
                                                  "trans_pl": _trans_pl, "trans_dp": _trans_dp,  "sv":_sv, dummy: _other_string,
                                                  "r": _r, "r_eps": _r_eps, "r_strength": _r_strength})
        for _data in ['TREC_transformer']  # , 'sst2_transformer', 'cr', 'mpqa', 'mr', 'subj'
        for _ne in [2]  # 100, 200, 800
        for _lr in [0.001]  #0.001, 0.00001
        for _bs in [64]  # 32, 128
        for _trans_pt in ['none', 'ape']
        for _trans_pl in ['mean']  # , 'sum', 'max'
        for _trans_dp in [0.1]  # 0., 0.5
        for _r in ["none", "edge"]
        for _r_eps in [0., 0.1]
        for _r_strength in [0., 0.1]
        for _other_string in ["", reg_base]
        for _sv in [790]
        for _bf in powerset(binary_flags)

        if not (_r != "edge" and "random_segment" in _bf)
           and not (_r == "edge" and "random_segment" not in _bf)
           and not (_r == "none" and (_r_strength != 0. or _r_eps != 0.))
           and not (_r != "none" and (_r_strength == 0. or _r_eps == 0.))
           and not (_r == "none" and _other_string != "")
           and not (_r != "none" and _other_string != reg_base)
           and not (_r == "none" and "random_segment" in _bf)
           and not (_r == "none" and "tangent_prop" in _bf)
           and not (_trans_pt == 'none' and _r != "none")
    ]


elif PS_TRANSFORMER_SMALL_PE:

    # data config
    data = ""

    # model_config
    model = "-mt Transformer --trans_small_pe "

    # training config
    train = ""

    # binary_flags = ["original_mode"]  # "use_mini_batching -bs 32"

    TASKS = [
        _get_command_tuple(data + model + train, {"data": _data, "ne":_ne, "lr": _lr, "bs": _bs, "trans_pt": _trans_pt,
                                                  "trans_pl": _trans_pl, "trans_dp": _trans_dp,  "sv":_sv})  #,_bf
        for _data in ['TREC_transformer', 'sst2_transformer']  # 'TREC_transformer', 'sst2_transformer', , 'cr', 'mpqa', 'mr', 'subj'
        for _ne in [100, 200, 400]  #  800
        for _lr in [0.001, 0.0001, 0.00001]
        for _bs in [32, 64, 128]
        for _trans_pt in ['ape']
        for _trans_pl in ['last_dim']
        for _trans_dp in [0., 0.1, 0.5]
        for _sv in range(133337, 133347)

        # for _bf in powerset(binary_flags)
    ]

elif UNREG_TRANSFORMER_PATIENCE:

    # data config
    data = ""

    # model_config
    model = "-mt Transformer "

    # training config
    train = "-pi 200 -lrrc 20 "

    # binary_flags = ["original_mode"]  # "use_mini_batching -bs 32"

    TASKS = [
        _get_command_tuple(data + model + train, {"data": _data, "ne":_ne, "lr": _lr, "bs": _bs, "trans_pt": _trans_pt,
                                                  "trans_nl": _trans_nl, "trans_pl": _trans_pl, "trans_dp": _trans_dp,  "sv":_sv})  #,_bf
        for _data in ['TREC_transformer', 'sst2_transformer']  # 'TREC_transformer', 'sst2_transformer', , 'cr', 'mpqa', 'mr', 'subj'
        for _ne in [200]  #  800
        for _lr in [0.001, 0.0001, 0.00001]
        for _bs in [32, 64, 128]
        for _trans_pt in ['none', 'ape']
        for _trans_nl in [1]  # 2
        for _trans_pl in ['last_dim']
        for _trans_dp in [0., 0.1, 0.5]
        for _sv in range(133347, 133352)

        # for _bf in powerset(binary_flags)
    ]

elif REG_BEST_TRANSFORMER_PATIENCE:

    # data config
    data = "-it set "

    # model_config
    model = "-mt Transformer "

    # training config
    train = "-pi 200 -lrrc 20 "

    reg_base = "-r_repr pred -tpp -random_segment -ppe "

    binary_flags = ["random_segment"]

    dummy = 'it set '  # Do not include the minus! (-)cd

    TASK_1 = [
        _get_command_tuple(data + model + train, {"data": _data, "ne": _ne, "lr": _lr, "bs": _bs,
                                                  "trans_pt": _trans_pt, "trans_nl": _trans_nl,
                                                  "trans_pl": _trans_pl, "trans_dp": _trans_dp,
                                                  dummy: _other_string,
                                                  "r": _r, "r_eps": _r_eps, "r_strength": _r_strength,
                                                  "sv": _sv}, _bf)
        for _data in ['TREC_transformer']
        for _ne in [200]
        for _lr in [0.001]
        for _bs in [64]
        for _trans_pt in ['ape']
        for _trans_nl in [1]
        for _trans_pl in ['last_dim']
        for _trans_dp in [0.5]
        for _other_string in ["", reg_base]
        for _r in ["edge"]  # , "none"
        for _r_eps in [0., 0.1]
        for _r_strength in [0., 0.0078125, 0.015625, 0.03125, 0.0625]  # 0.125, 0.25, 0.5, 1., 2., 4., 8., 16.
        for _sv in range(133347, 133352)
        for _bf in powerset(binary_flags)
        if not (_r != "edge" and "random_segment" in _bf)
           and not (_r == "edge" and "random_segment" not in _bf)
           and not (_r == "none" and (_r_strength != 0. or _r_eps != 0.))
           and not (_r != "none" and (_r_strength == 0. or _r_eps == 0.))
           and not (_r == "none" and _other_string != "")
           and not (_r != "none" and _other_string != reg_base)
           and not (_r == "none" and "random_segment" in _bf)
           and not (_r == "none" and "tangent_prop" in _bf)
           and not (_trans_pt == 'none' and _r != "none")
    ]

    TASK_2 = [
        _get_command_tuple(data + model + train, {"data": _data, "ne": _ne, "lr": _lr, "bs": _bs,
                                                  "trans_pt": _trans_pt, "trans_nl": _trans_nl,
                                                  "trans_pl": _trans_pl, "trans_dp": _trans_dp,
                                                  dummy: _other_string,
                                                  "r": _r, "r_eps": _r_eps, "r_strength": _r_strength,
                                                  "sv": _sv}, _bf)
        for _data in ['sst2_transformer']  # ,
        for _ne in [200]
        for _lr in [0.001]
        for _bs in [128]
        for _trans_pt in ['ape']
        for _trans_nl in [1]
        for _trans_pl in ['last_dim']
        for _trans_dp in [0.5]
        for _other_string in ["", reg_base]
        for _r in ["edge"]  # , "none"
        for _r_eps in [0., 0.1]
        for _r_strength in [0., 0.125, 0.25, 0.5, 1., 2., 4., 8., 16.]
        for _sv in range(133347, 133352)
        for _bf in powerset(binary_flags)
        if not (_r != "edge" and "random_segment" in _bf)
           and not (_r == "edge" and "random_segment" not in _bf)
           and not (_r == "none" and (_r_strength != 0. or _r_eps != 0.))
           and not (_r != "none" and (_r_strength == 0. or _r_eps == 0.))
           and not (_r == "none" and _other_string != "")
           and not (_r != "none" and _other_string != reg_base)
           and not (_r == "none" and "random_segment" in _bf)
           and not (_r == "none" and "tangent_prop" in _bf)
           and not (_trans_pt == 'none' and _r != "none")
    ]

    TASKS = TASK_1 # + TASK_2

elif REG_BEST_TRANSFORMER_PATIENCE_TESTING:

    # data config
    data = "-it set "

    # model_config
    model = "-mt Transformer "

    # training config
    train = "-pi 200 -lrrc 20 "

    reg_base = "-r_repr pred -tpp -random_segment -ppe "

    binary_flags = ["random_segment"]

    dummy = 'it set '  # Do not include the minus! (-)cd

    TASK_1 = [
        _get_command_tuple(data + model + train, {"data": _data, "ne": _ne, "lr": _lr, "bs": _bs,
                                                  "trans_pt": _trans_pt, "trans_nl": _trans_nl,
                                                  "trans_pl": _trans_pl, "trans_dp": _trans_dp,
                                                  dummy: _other_string,
                                                  "r": _r, "r_eps": _r_eps, "r_strength": _r_strength,
                                                  "to": _to, "sv": _sv}, _bf)
        for _data in ['TREC_transformer']
        for _ne in [200]
        for _lr in [0.001]
        for _bs in [64]
        for _trans_pt in ['ape']
        for _trans_nl in [1]
        for _trans_pl in ['last_dim']
        for _trans_dp in [0.5]
        for _other_string in ["", reg_base]
        for _r in ["edge"]  # , "none"
        for _r_eps in [0., 0.1]
        for _r_strength in [0., 0.0078125, 0.015625, 0.03125, 0.0625]  # , 0.125, 0.25, 0.5, 1., 2., 4., 8., 16.
        for _to in ["random", "sort_up", "sort_down", "reverse", "pad_first"]
        for _sv in range(133347, 133352)
        for _bf in powerset(binary_flags)
        if not (_r != "edge" and "random_segment" in _bf)
           and not (_r == "edge" and "random_segment" not in _bf)
           and not (_r == "none" and (_r_strength != 0. or _r_eps != 0.))
           and not (_r != "none" and (_r_strength == 0. or _r_eps == 0.))
           and not (_r == "none" and _other_string != "")
           and not (_r != "none" and _other_string != reg_base)
           and not (_r == "none" and "random_segment" in _bf)
           and not (_r == "none" and "tangent_prop" in _bf)
           and not (_trans_pt == 'none' and _r != "none")
    ]

    TASK_2 = [
        _get_command_tuple(data + model + train, {"data": _data, "ne": _ne, "lr": _lr, "bs": _bs,
                                                  "trans_pt": _trans_pt, "trans_nl": _trans_nl,
                                                  "trans_pl": _trans_pl, "trans_dp": _trans_dp,
                                                  dummy: _other_string,
                                                  "r": _r, "r_eps": _r_eps, "r_strength": _r_strength,
                                                  "to": _to, "sv": _sv}, _bf)
        for _data in ['sst2_transformer']  # ,
        for _ne in [200]
        for _lr in [0.001]
        for _bs in [128]
        for _trans_pt in ['ape']
        for _trans_nl in [1]
        for _trans_pl in ['last_dim']
        for _trans_dp in [0.5]
        for _other_string in ["", reg_base]
        for _r in ["edge"]  # , "none"
        for _r_eps in [0., 0.1]
        for _r_strength in [0., 0.125, 0.25, 0.5, 1., 2., 4., 8., 16.]
        for _to in ["random", "sort_up", "sort_down", "reverse", "pad_first"]
        for _sv in range(133347, 133352)
        for _bf in powerset(binary_flags)
        if not (_r != "edge" and "random_segment" in _bf)
           and not (_r == "edge" and "random_segment" not in _bf)
           and not (_r == "none" and (_r_strength != 0. or _r_eps != 0.))
           and not (_r != "none" and (_r_strength == 0. or _r_eps == 0.))
           and not (_r == "none" and _other_string != "")
           and not (_r != "none" and _other_string != reg_base)
           and not (_r == "none" and "random_segment" in _bf)
           and not (_r == "none" and "tangent_prop" in _bf)
           and not (_trans_pt == 'none' and _r != "none")
    ]

    TASKS = TASK_1 # + TASK_2

elif REG_PS_TRANSFORMER_PATIENCE:

    # data config
    data = "-it set "

    # model_config
    model = "-mt Transformer "  # --trans_dont_dropout_input

    # training config
    train = "-pi 200 -lrrc 20 "

    reg_base = "-r_repr pred -tpp -random_segment -ppe "

    binary_flags = ["random_segment"]

    dummy = 'it set '  # Do not include the minus! (-)cd

    TASKS = [
        _get_command_tuple(data + model + train, {"data": _data, "ne": _ne, "lr": _lr, "bs": _bs,
                                                             "trans_pt": _trans_pt, "trans_nl": _trans_nl,
                                                             "trans_pl": _trans_pl, "trans_dp": _trans_dp,
                                                             dummy: _other_string,
                                                             "r": _r, "r_eps": _r_eps, "r_strength": _r_strength,
                                                             "sv": _sv} , _bf)
        for _data in ['TREC_transformer', 'sst2_transformer']
        for _ne in [200]
        for _lr in [0.001]  # , 0.0001, 0.00001
        for _bs in [32]  #
        for _trans_pt in ['none', 'ape']
        for _trans_nl in [1]
        for _trans_pl in ['mean']  # ' last_dim', , 'max'
        for _trans_dp in [0., 0.5]  # 0.1
        for _other_string in ["", reg_base]
        for _r in ["edge"]  # , "none"
        for _r_eps in [0., 0.1]
        for _r_strength in [0., 0.0078125, 0.015625, 0.03125, 0.0625, 0.125, 0.25, 0.5, 1., 2., 4., 8., 16.]  #
        for _sv in range(133347, 133352)
        for _bf in powerset(binary_flags)
        if not (_r != "edge" and "random_segment" in _bf)
           and not (_r == "edge" and "random_segment" not in _bf)
           and not (_r == "none" and (_r_strength != 0. or _r_eps != 0.))
           and not (_r != "none" and (_r_strength == 0. or _r_eps == 0.))
           and not (_r == "none" and _other_string != "")
           and not (_r != "none" and _other_string != reg_base)
           and not (_r == "none" and "random_segment" in _bf)
           and not (_r == "none" and "tangent_prop" in _bf)
           and not (_trans_pt == 'none' and _r != "none")
           and not (_data == 'TREC_transformer' and _r_strength > 1.)
           and not (_data == 'sst2_transformer' and 0. < _r_strength < 0.125)
    ]

elif REG_PS_TRANSFORMER_PATIENCE_TESTING:

    # data config
    data = "-it set "

    # model_config
    model = "-mt Transformer "

    # training config
    train = "-pi 200 -lrrc 20 "

    reg_base = "-r_repr pred -tpp -random_segment -ppe "

    binary_flags = ["random_segment"]

    dummy = 'it set '  # Do not include the minus! (-)cd

    TASKS = [
        _get_command_tuple(data + model + train, {"data": _data, "ne": _ne, "lr": _lr, "bs": _bs,
                                                             "trans_pt": _trans_pt, "trans_nl": _trans_nl,
                                                             "trans_pl": _trans_pl, "trans_dp": _trans_dp,
                                                             dummy: _other_string,
                                                             "r": _r, "r_eps": _r_eps, "r_strength": _r_strength,
                                                             "teo": _teo, "tess": _tess, "sv": _sv} , _bf)
        for _teo in ["random", "sort_up", "sort_down", "reverse"]  # , "pad_first"
        for _data in ['sst2_transformer', 'TREC_transformer']
        for _sv, _tess in zip(range(133347, 133352), range(100, 105))
        for _ne in [200]
        for _lr in [0.001]  # , 0.0001, 0.00001
        for _bs in [32, 64, 128]  #
        for _trans_pt in ['none', 'ape']
        for _trans_nl in [1]
        for _trans_pl in ['mean']  # , 'max'
        for _trans_dp in [0., 0.5]  # 0.1
        for _other_string in ["", reg_base]
        for _r in ["none"]  # "none",
        for _r_eps in [0., 0.1]
        for _r_strength in [0., 0.125, 0.25, 0.5, 1., 2., 4., 8., 16.]  #   0.0078125, 0.015625, 0.03125, 0.0625,
        for _bf in powerset(binary_flags)
        if not (_r != "edge" and "random_segment" in _bf)
           and not (_r == "edge" and "random_segment" not in _bf)
           and not (_r == "none" and (_r_strength != 0. or _r_eps != 0.))
           and not (_r != "none" and (_r_strength == 0. or _r_eps == 0.))
           and not (_r == "none" and _other_string != "")
           and not (_r != "none" and _other_string != reg_base)
           and not (_r == "none" and "random_segment" in _bf)
           and not (_r == "none" and "tangent_prop" in _bf)
           and not (_trans_pt == 'none' and _r != "none")
    ]

elif REG_PS_TRANSFORMER_PATIENCE_TRAIN_ON_SORT_UP_RANDOM:

    # data config
    data = "-it set "

    # model_config
    model = "-mt Transformer --trans_dont_dropout_input "

    # training config
    train = "-pi 200 -lrrc 20 "

    reg_base = "-r_repr pred -tpp -random_segment -ppe "

    binary_flags = ["random_segment"]

    dummy = 'it set '  # Do not include the minus! (-)cd

    TASKS = [
        _get_command_tuple(data + model + train, {"data": _data, "tro": _tro, "ne": _ne, "lr": _lr, "bs": _bs,
                                                  "trans_pt": _trans_pt, "trans_nl": _trans_nl,
                                                  "trans_pl": _trans_pl, "trans_dp": _trans_dp,
                                                  dummy: _other_string,
                                                  "r": _r, "r_eps": _r_eps, "r_strength": _r_strength,
                                                  "sv": _sv} , _bf)
        for _data in ['TREC_transformer', 'sst2_transformer']
        for _tro in ["sort_up", "random"]
        for _ne in [200]
        for _lr in [0.001]  # , 0.0001, 0.00001
        for _bs in [32, 64]  # , 128
        for _trans_pt in ['none', 'ape']
        for _trans_nl in [1]
        for _trans_pl in ['mean']
        for _trans_dp in [0., 0.5]  # 0.1
        for _other_string in ["", reg_base]
        for _r in ["none"]  # , "edge"
        for _r_eps in [0., 0.1]
        for _r_strength in [0., 0.0078125, 0.015625, 0.03125, 0.0625, 0.125, 0.25, 0.5, 1., 2., 4., 8., 16.]
        for _sv in range(133347, 133352)
        for _bf in powerset(binary_flags)
        if not (_r != "edge" and "random_segment" in _bf)
           and not (_r == "edge" and "random_segment" not in _bf)
           and not (_r == "none" and (_r_strength != 0. or _r_eps != 0.))
           and not (_r != "none" and (_r_strength == 0. or _r_eps == 0.))
           and not (_r == "none" and _other_string != "")
           and not (_r != "none" and _other_string != reg_base)
           and not (_r == "none" and "random_segment" in _bf)
           and not (_r == "none" and "tangent_prop" in _bf)
           and not (_trans_pt == 'none' and _r != "none")
           and not (_data == 'TREC_transformer' and _r_strength > 1.)
           and not (_data == 'sst2_transformer' and 0. < _r_strength < 0.125)
    ]

elif REG_PS_TRANSFORMER_PATIENCE_TRAIN_ON_SORT_UP_RANDOM_TESTING:

    # data config
    data = "-it set "

    # model_config
    model = "-mt Transformer "

    # training config
    train = "-pi 200 -lrrc 20 "

    reg_base = "-r_repr pred -tpp -random_segment -ppe "

    binary_flags = ["random_segment"]

    dummy = 'it set '  # Do not include the minus! (-)cd

    TASKS = [
        _get_command_tuple(data + model + train, {"data": _data, "tro": _tro, "ne": _ne, "lr": _lr, "bs": _bs,
                                                  "trans_pt": _trans_pt, "trans_nl": _trans_nl,
                                                  "trans_pl": _trans_pl, "trans_dp": _trans_dp,
                                                  dummy: _other_string,
                                                  "r": _r, "r_eps": _r_eps, "r_strength": _r_strength,
                                                  "teo": _teo, "sv": _sv} , _bf)
        for _data in ['TREC_transformer', 'sst2_transformer']
        for _tro in ["sort_up", "random"]
        for _ne in [200]
        for _lr in [0.001]  # , , 0.0001, 0.00001
        for _bs in [32, 64]  # , 128
        for _trans_pt in ['none', 'ape']
        for _trans_nl in [1]
        for _trans_pl in ['mean']
        for _trans_dp in [0., 0.5]  # 0.1
        for _other_string in ["", reg_base]
        for _r in ["edge"]  # , "none"
        for _r_eps in [0., 0.1]
        for _r_strength in [0., 0.0078125, 0.015625, 0.03125, 0.0625, 0.125, 0.25, 0.5, 1., 2., 4., 8., 16.]
        for _teo in ["random", "sort_up", "sort_down", "reverse", "pad_first"]
        for _sv in range(133347, 133352)
        for _bf in powerset(binary_flags)
        if not (_r != "edge" and "random_segment" in _bf)
           and not (_r == "edge" and "random_segment" not in _bf)
           and not (_r == "none" and (_r_strength != 0. or _r_eps != 0.))
           and not (_r != "none" and (_r_strength == 0. or _r_eps == 0.))
           and not (_r == "none" and _other_string != "")
           and not (_r != "none" and _other_string != reg_base)
           and not (_r == "none" and "random_segment" in _bf)
           and not (_r == "none" and "tangent_prop" in _bf)
           and not (_trans_pt == 'none' and _r != "none")
           and not (_data == 'TREC_transformer' and _r_strength > 1.)
           and not (_data == 'sst2_transformer' and 0. < _r_strength < 0.125)
    ]

elif REG_PS_TRANSFORMER_PATIENCE_OTHER_FOUR:

    # data config
    data = "-it set --n_fold 10 "

    # model_config
    model = "-mt Transformer "

    # training config
    train = "-pi 200 -lrrc 20 "

    reg_base = "-r_repr pred -tpp -random_segment -ppe "

    binary_flags = ["random_segment"]

    dummy = 'it set '  # Do not include the minus! (-)cd

    TASKS = [
        _get_command_tuple(data + model + train, {"data": _data, "ix": _ix, "ne": _ne, "lr": _lr, "bs": _bs,
                                                  "trans_pt": _trans_pt, "trans_nl": _trans_nl,
                                                  "trans_pl": _trans_pl, "trans_dp": _trans_dp,
                                                  dummy: _other_string,
                                                  "r": _r, "r_eps": _r_eps, "r_strength": _r_strength,
                                                  "sv": _sv} , _bf)
        for _data in ['cr']  # , 'mpqa', 'mr', 'subj'
        for _ix, _sv in zip(range(10), range(133352, 133362))
        for _ne in [200]
        for _lr in [0.001]  # , 0.0001 , 0.00001
        for _bs in [32]  # , 64, 128
        for _trans_pt in ['none', 'ape']
        for _trans_nl in [1]
        for _trans_pl in ['mean']
        for _trans_dp in [0., 0.5]  # 0.1
        for _other_string in ["", reg_base]
        for _r in ["edge"]  # , "none"
        for _r_eps in [0., 0.1]
        for _r_strength in [0., 0.125, 0.25, 0.5, 1., 2., 4., 8., 16.]  # 0.0078125, 0.015625, 0.03125, 0.0625,
        for _bf in powerset(binary_flags)
        if not (_r != "edge" and "random_segment" in _bf)
           and not (_r == "edge" and "random_segment" not in _bf)
           and not (_r == "none" and (_r_strength != 0. or _r_eps != 0.))
           and not (_r != "none" and (_r_strength == 0. or _r_eps == 0.))
           and not (_r == "none" and _other_string != "")
           and not (_r != "none" and _other_string != reg_base)
           and not (_r == "none" and "random_segment" in _bf)
           and not (_r == "none" and "tangent_prop" in _bf)
           and not (_trans_pt == 'none' and _r != "none")
           # and not (_data == 'TREC_transformer' and _r_strength > 1.)
           # and not (_data == 'sst2_transformer' and 0. < _r_strength < 0.125)
    ]

elif REG_PS_TRANSFORMER_PATIENCE_OTHER_FOUR_TESTING:

    # data config
    data = "-it set --n_fold 10 "

    # model_config
    model = "-mt Transformer "

    # training config
    train = "-pi 200 -lrrc 20 "

    reg_base = "-r_repr pred -tpp -random_segment -ppe "

    binary_flags = ["random_segment"]

    dummy = 'it set '  # Do not include the minus! (-)cd

    TASKS = [
        _get_command_tuple(data + model + train, {"data": _data, "ix": _ix, "ne": _ne, "lr": _lr, "bs": _bs,
                                                  "trans_pt": _trans_pt, "trans_nl": _trans_nl,
                                                  "trans_pl": _trans_pl, "trans_dp": _trans_dp,
                                                  dummy: _other_string,
                                                  "r": _r, "r_eps": _r_eps, "r_strength": _r_strength,
                                                  "teo": _teo, "sv": _sv} , _bf)
        for _teo in ["random", "sort_up", "sort_down", "reverse"]  #
        for _data in ['cr']  # , 'mpqa', 'mr', 'subj'
        for _ix, _sv, _tess in zip(range(10), range(133352, 133362), range(100, 110))
        for _ne in [200]
        for _lr in [0.001, 0.0001]  # , 0.00001
        for _bs in [32, 64, 128]  #
        for _trans_pt in ['none', 'ape']
        for _trans_nl in [1]
        for _trans_pl in ['mean']
        for _trans_dp in [0., 0.5]  # 0.1
        for _other_string in ["", reg_base]
        for _r in ["none"]  # , "edge"
        for _r_eps in [0., 0.1]
        for _r_strength in [0., 0.125, 0.25, 0.5, 1., 2., 4., 8., 16.]  # 0.0078125, 0.015625, 0.03125, 0.0625,
        for _bf in powerset(binary_flags)
        if not (_r != "edge" and "random_segment" in _bf)
           and not (_r == "edge" and "random_segment" not in _bf)
           and not (_r == "none" and (_r_strength != 0. or _r_eps != 0.))
           and not (_r != "none" and (_r_strength == 0. or _r_eps == 0.))
           and not (_r == "none" and _other_string != "")
           and not (_r != "none" and _other_string != reg_base)
           and not (_r == "none" and "random_segment" in _bf)
           and not (_r == "none" and "tangent_prop" in _bf)
           and not (_trans_pt == 'none' and _r != "none")
           # and not (_data == 'TREC_transformer' and _r_strength > 1.)
           # and not (_data == 'sst2_transformer' and 0. < _r_strength < 0.125)
    ]


if __name__ == '__main__':
    print(TASKS)
    if DRYRUN:
        for t in TASKS:
            print(t)
        print(len(TASKS))
    else:
        from run.worker import Worker
        Worker.main(TASKS, QUEUE_NAME)
