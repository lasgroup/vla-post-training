from scripts.launcher_utils import generate_run_commands_clariden, generate_base_command, dict_permutations
import argparse
import os
import scripts.train_online as exp
from time import gmtime, strftime

USER = os.environ['USER']

# General Configurations
base_applicable_configs = {
    #'group_name': ["test"], #dreamer_val
    'config_name': ['pi05_libero_online'],
    'collect.env_num': [4,8,16,32,64,128],
    'collect.num_rollouts': [250],
    'checkpoint_base_dir': [f'/capstor/scratch/cscs/{USER}/checkpoints'],
    'weight-loader.params-path': ['gs://openpi-assets/checkpoints/pi05_libero/params'],
    'num_train_steps': [1],
}

libero_applicable_configs = {
    'collect.tasks': ["libero_90_59"] # Space seperated string of tasks
}

collect_applicable_configs = {}

env_applicable_config_list = [libero_applicable_configs]
algorithm_applicable_config_list = [collect_applicable_configs]
applicable_config_list = [base_applicable_configs | env_con | alg_con for env_con in env_applicable_config_list for alg_con in algorithm_applicable_config_list]

def generate_experiment_name(flags, applicable_config):
    exp_name = f"{flags.get('group_name', 'default').replace('/', '_')}_{flags.get('alg', 'exp').replace('/', '_')}"
    varied_keys = sorted(
        k for k, v in applicable_config.items() 
        if len(v) > 1 and k not in ['task', 'config_class', 'group_name', 'alg', 'seed']
    )
    
    if varied_keys:
        sweep_parts = [f"{k}_{flags[k]}" for k in varied_keys]
        exp_name += "__" + "__".join(sweep_parts)
    return exp_name

def main(args):
    command_list = []
    output_file_list = []

    for applicable_config in applicable_config_list:
        for flags in dict_permutations(applicable_config):
            flags['exp-name'] = generate_experiment_name(flags, applicable_config)
            config_name = flags.pop("config_name")
            cmd = generate_base_command(exp, no_flag_option=config_name, flags=flags)
            command_list.append(cmd)
            #output_dir = '...'
            #output_file_list.append(output_dir + flags['exp_name'] + "_" + strftime("%Y-%m-%d_%H:%M:%S", gmtime()) + "_" + str(flags['seed']) + ".out")

    num_hours = 10
    generate_run_commands_clariden(command_list, duration=f'{num_hours}:59:00', prompt=True, dry=args.print) #, output_file_list=output_file_list


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--print', default=False, action="store_true")

    args = parser.parse_args()
    main(args)