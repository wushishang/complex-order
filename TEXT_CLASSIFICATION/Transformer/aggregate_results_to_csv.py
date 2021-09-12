import os

DEBUG = int(os.environ.get("pydebug", 0)) > 0
if DEBUG:
    import pydevd_pycharm
    pydevd_pycharm.settrace('localhost', port=27189, stdoutToServer=True, stderrToServer=True, suspend=False)

import pandas as pd
import time
from tqdm import tqdm

from common.helper import Util
from common.json_dump import JsonDump
from run_experiments import TASKS
from config import Config
from functools import partial


def fetch_df(c, stats_or_output):
    t, td = c
    t = list(filter(None, t))  # Remove empty strings
    cfg = Config(t)
    if stats_or_output == "output":
        fp = cfg.output_file_name()
    else:
        fp = cfg.stats_file_name()
    if os.path.exists(fp):
        # @team, change this to aggregate a different file
        print(fp)
        df = JsonDump.read(fp)
        if df.shape[0] == 0:
            return None
        for k, v in td.items():
            df.loc[:, k] = v
        if df.shape[0] > 1 and stats_or_output == 'output':
            # Trim to the first computed output
            df = df.iloc[:1]
        return df
    else:
        return None


class AggregateResults:

    @classmethod
    def read_data(cls, stats_or_output='output'):
        variants = list(TASKS[0][3].keys())
        tasks = [(t[1].split("train.py ")[1].split(" "), t[3]) for t in TASKS]
        fetch_function = partial(fetch_df, stats_or_output=stats_or_output)
        dfl = list(map(fetch_function, tqdm(tasks)))
        dfl = Util.not_none(dfl)
        df = pd.concat(dfl, sort=True).reindex()
        return df, variants

    @classmethod
    def stat_files(cls):
        tasks = [(t[1].split("train.py ")[1].split(" "), t[3]) for t in TASKS]

        with open('my_statfiles.txt', 'w') as f:
            for c in tasks:
                t, td = c
                t = list(filter(None, t))  # Remove empty strings
                cfg = Config(t)
                f.write(cfg.stats_file_name() + "\n")

    @classmethod
    def main(cls, stats_or_output='output'):
        df, bl_variants = cls.read_data(stats_or_output=stats_or_output)
        # Sanity check
        # if stats_or_output == "output":
        #     check = (df.epoch == 999)
        #     if not np.all(check):
        #         print("\n", "-"*10, "\nWarning: not all epoch in range, will be skipped\n", "-"*10)
        #     df = df.loc[check]

        prefix = time.strftime("%Y_%m_%d_%H_%M_")
        df.to_csv(f"../csv/{prefix}_{stats_or_output}.csv")
        print("Done")
        # BROKEN: Variants does not handle boolean flags.
        # gb_keys = bl_variants.copy()
        # # gb_keys.remove('lr')
        # # val_df = df.loc[df.phase == 'validation']
        # bl_stats = df.groupby(gb_keys).agg({'best_val_metric': 'min'}).reset_index().merge(df).loc[:, gb_keys ].merge(
        #     df)
        # @team bl_stats are hyperparameter tuned and df are the raw combined CSVs
        # bl_stats.to_csv(f"../csv/{prefix}hp_tuned.csv"

        # @team This is example code to make pivot tables
        # for _a in bl_stats['a'].unique():
        #     print(f"Aggregation = {_a}")
        #     bl_pt = bl_stats.loc[bl_stats.a == _a].pivot_table(['rmse'], ['sl'], ['phase'], ['mean', 'std', 'count'], dropna=False)
        #     bl_pt = Util.process_pivot_table(bl_pt, count=False)
        #     print(bl_pt.sort_values(['sl']).to_string())


if __name__ == '__main__':
    AggregateResults.main('stats')
    # AggregateResults.stat_files()
