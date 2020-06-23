import argparse
import datetime
import multiprocessing
import subprocess

parser = argparse.ArgumentParser()
parser.add_argument('--yaml_file', type=str, default='./args_yml/main_exp/halfcheetah-rp1.yml')   ## only works properly for HalfCheetah and Ant
parser.add_argument('--seeds5to9', dest='seeds5to9', action='store_true')
parser.set_defaults(seeds5to9=False)

args = parser.parse_args()
params = vars(args)

experiment_id = datetime.datetime.now().strftime('%Y_%m_%d_%H-%M-%S')
num_experiments = 5
seeds_5to9 = params['seeds5to9']
lower = 0
upper = num_experiments

main_experiment = ["python", "train.py", "--yaml_file", params['yaml_file'], "--uuid", experiment_id, "--seed"]

if seeds_5to9:
    lower += 5
    upper += 5

all_experiments = [main_experiment + [str(i)] for i in range(lower, upper)]

def run_experiment(spec):
    subprocess.run(spec, check=True)

def run_all_experiments(specs):
    pool = multiprocessing.Pool()
    pool.map(run_experiment, specs)

run_all_experiments(all_experiments)
