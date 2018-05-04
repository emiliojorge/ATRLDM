import json
import os
import sys
import test
from shutil import copyfile
from base_algorithm import BaseAlgorithm
from util import EpsilonGreedy
import numpy as np
import pandas as pd
from speedyQ import Speedy_Qlearning
from mean_agent import MeanAgent
from bayesian_qlearning import Bayesian_Qlearning
from dynaq_agent import DynaQAgent
from q_agent import QAgent
from random_agent import RandomAgent

repetitions = 1

single_model = True
path_to_experiment_configs = "experiments/"
default_config = 'config.json'


#algorithm = BaseAlgorithm(exploration=True, explorer=EpsilonGreedy(start=0.5, end=0.05, steps=1000),
#                          use_database=True, action_selection = "epsilon greedy")

#algorithm = BaseAlgorithm(exploration=True, explorer=EpsilonGreedy(start=1, end=0.05, steps=5000),
#                          use_database=True, action_selection = "majority vote")
explorer = EpsilonGreedy(start=1., end=0.05, steps=10000)

# algorithm = QAgent(exploration=True, explorer=explorer)
# algorithm = DynaQAgent(exploration=True, explorer=explorer)
# algorithm = Bayesian_Qlearning()
#algorithm = Speedy_Qlearning(exploration=True, explorer=explorer)
# algorithm = MeanAgent(exploration=True, explorer=explorer)
# algorithm = RandomAgent()


def generate_experiments():
    for lamda in np.linspace(0, 0.9, num=3, endpoint=True):
        for seed in range(4):
            for double in (True, False):
                with open(os.path.join(path_to_experiment_configs, default_config)) as config_file:
                    config = json.load(config_file)

                config["start_agents"]["q"]["lamda"] = lamda
                config["start_agents"]["dynaq"]["lamda"] = lamda

                config["start_agents"]["q"]["double"] = double
                config["seed"] = seed

                f_name = path_to_experiment_configs + "config_lamda{}_double{}_seed{}.json".format(int(lamda * 100),
                                                                                                   double, seed)
                with open(f_name, 'w') as f:
                    json.dump(config, f)


# generate_experiments()

with open("results.out", 'w') as out:
    sys.stdout = out
    if single_model:
        configs = [""]
    else:
        configs =  os.listdir(path_to_experiment_configs)
    full_results = pd.DataFrame()
    for i in range(repetitions):
        for f in configs:
            print(f)
            if not single_model:
                copyfile(os.path.join(path_to_experiment_configs, f), "config.json")
            environments, scores = test.main(algorithm)

            df = pd.DataFrame(scores)
            df['config'] = f
            df['env'] = environments
            df['rep'] = i
            df['mean'] = np.mean(scores, axis=1)
            df['std'] = np.std(scores, axis=1)
            full_results = full_results.append(df)

    full_results.to_csv(f'full_results_{algorithm.algorithm}.csv', index=False)
