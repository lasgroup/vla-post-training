import glob
import hashlib
import itertools
import json
import multiprocessing
import os
import sys
from typing import Dict, Optional, Any, List

import numpy as np
import pandas as pd



""" Async executor """


class AsyncExecutor:

    def __init__(self, n_jobs=1):
        self.num_workers = n_jobs if n_jobs > 0 else multiprocessing.cpu_count()
        self._pool = []
        self._populate_pool()

    def run(self, target, *args_iter, verbose=False):
        workers_idle = [False] * self.num_workers
        tasks = list(zip(*args_iter))
        n_tasks = len(tasks)

        while not all(workers_idle):
            for i in range(self.num_workers):
                if not self._pool[i].is_alive():
                    self._pool[i].terminate()
                    if len(tasks) > 0:
                        if verbose:
                            print(n_tasks - len(tasks))
                        next_task = tasks.pop(0)
                        self._pool[i] = _start_process(target, next_task)
                    else:
                        workers_idle[i] = True

    def _populate_pool(self):
        self._pool = [_start_process(_dummy_fun) for _ in range(self.num_workers)]


def _start_process(target, args=None):
    if args:
        p = multiprocessing.Process(target=target, args=args)
    else:
        p = multiprocessing.Process(target=target)
    p.start()
    return p


def _dummy_fun():
    pass


""" Command generators """


def generate_base_command(module, no_flag_option: str = None, flags: Optional[Dict[str, Any]] = None, unbuffered: bool = True) -> str:
    """ Generates the command to execute python module with provided flags

    Args:
        module: python module / file to run
        flags: dictionary of flag names and the values to assign to them.
               assumes that boolean flags are encoded as store_true flags with False as default.
        unbuffered: whether to invoke an unbuffered python output stream

    Returns: (str) command which can be executed via bash

    """

    """ Module is a python file to execute """
    #interpreter_script = sys.executable
    interpreter_script = 'uv run'
    base_exp_script = os.path.abspath(module.__file__)
    base_cmd = interpreter_script + ' ' + base_exp_script
    if no_flag_option:
        base_cmd += ' ' + no_flag_option
    if flags is not None:
        assert isinstance(flags, dict), "Flags must be provided as dict"
        for flag, setting in flags.items():
            if type(setting) == bool or type(setting) == np.bool_:
                if setting:
                    base_cmd += f" --{flag}"
            else:
                base_cmd += f" --{flag}={setting}"
    return base_cmd

def generate_run_commands_clariden(command_list: List[str], output_file_list: Optional[List[str]] = None, dry: bool = False, duration: str = '3:59:00', account='a143', env='vla-post-training', prompt: bool = True,):
    cluster_cmds = []
    bsub_cmd = 'sbatch ' + \
                f'--time={duration} ' + \
                f'--nodes=1 ' + \
                f'--environment {env} ' \
                + f'--account={account} '
    if output_file_list is None:
        for cmd in command_list:
            cluster_cmds.append(bsub_cmd + f'--wrap="{cmd}";')
    
    if dry:
        for cmd in cluster_cmds:
            print(cmd)
    else:
        if prompt:
            answer = input(f"about to launch {len(command_list)} jobs. proceed? [yes/no]")
        else:
            answer = 'yes'
        if answer == 'yes':
            for cmd in cluster_cmds:
                os.system(cmd)

def dict_permutations(d: dict) -> List[dict]:
    keys = d.keys()
    values = d.values()
    perms = []

    # Calculate the Cartesian product of all values in the dictionary
    for value_combo in itertools.product(*values):
        perms.append(dict(zip(keys, value_combo)))

    return perms