from itertools import chain, combinations
import numpy

PS_TRANSFORMER_SIX_TASKS = True

SYNTAX_CHECK = True
DRYRUN = False

EXP_DATE = "_sept_1_2021" # "_apr_10"
QUEUE_NAME = "RP_REG_wu1396_ml00" + EXP_DATE # "RP_REG_wu1396_bell"
SRC_FOLDER = "complex-order" + EXP_DATE + "/TEXT_CLASSIFICATION/Transformer" # "rpgin_reg_mar_27"
PYTHON_MAINFILE = "train.py"
USER= "wu1396"


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


if PS_TRANSFORMER_SIX_TASKS:

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

if __name__ == '__main__':
    print(TASKS)
    if DRYRUN:
        for t in TASKS:
            print(t)
        print(len(TASKS))
    else:
        from run.worker import Worker
        Worker.main(TASKS, QUEUE_NAME)
