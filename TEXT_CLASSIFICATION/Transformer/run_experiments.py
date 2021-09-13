from itertools import chain, combinations
import numpy

TRANSFORMER_TWO_TASKS = False
PI_TRANSFORMER_HP_TUNING = False
PS_TRANSFORMER_HP_TUNING = False
PS_TRANSFORMER_SMALL_PE = False
UNREG_TRANSFORMER_PATIENCE = False

SYNTAX_CHECK = True
DRYRUN = True

EXP_DATE = "_sept_11_2021" # "_apr_10"
CLUSTER = "_gilbreth"  #    "_ml00"
QUEUE_NAME = "RP_REG_wu1396" + CLUSTER + EXP_DATE # "RP_REG_wu1396_bell"
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
    train = "-pi 0 -lrrc 2 "

    # binary_flags = ["use_batchnorm"]  # "use_mini_batching -bs 32"

    TASKS = [
        _get_command_tuple(data + model + train, {"data": _data, "ne":_ne, "lr": _lr, "bs": _bs, "trans_pt": _trans_pt,
                                                  "trans_pl": _trans_pl, "trans_dp": _trans_dp,  "sv":_sv})
        for _data in ['TREC_transformer']  # , 'sst2_transformer', 'cr', 'mpqa', 'mr', 'subj'
        for _ne in [2]  # 100, 200, 800
        for _lr in [0.001]  #0.001, 0.00001
        for _bs in [64]  # 32, 128
        for _trans_pt in ['none', 'ape']
        for _trans_pl in ['last_dim']  # , 'sum', 'max'
        for _trans_dp in [0.1]  # 0., 0.5
        for _sv in [783]

        # for _bf in powerset(binary_flags)
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


if __name__ == '__main__':
    print(TASKS)
    if DRYRUN:
        for t in TASKS:
            print(t)
        print(len(TASKS))
    else:
        from run.worker import Worker
        Worker.main(TASKS, QUEUE_NAME)
