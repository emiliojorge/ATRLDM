import json
import os
import sys
import test
from shutil import copyfile

import numpy as np
import pandas as pd

repetitions = 1

path_to_experiment_configs = "experiments/"
default_config = 'config.json'


algorithm = BaseAlgorithm(exploration=True, explorer=EpsilonGreedy(start=0.5, end=0.05, steps=1000),
                          use_database=True, action_selection = "epsilon greedy")

#algorithm = BaseAlgorithm(exploration=True, explorer=EpsilonGreedy(start=1, end=0.05, steps=5000),
#                          use_database=True, action_selection = "majority vote")

# algorithm = QAgent(exploration=True, explorer=EpsilonGreedy(start=0.95, end=0.05, steps=1000))#eps_start=0.95, eps_end=0.05, eps_num=1000, learning_rate=lambda n: 1/n)
# algorithm = DynaQAgent()#planning_steps=50, eps_start=0.95, eps_end=0.05, eps_num=1000, learning_rate=lambda n: 1/n)
# algorithm = Bayesian_Qlearning()#action_selection="q-sampling", update_method="mom")
# algorithm = Speedy_Qlearning()
# algorithm = MeanAgent()


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

    full_results = pd.DataFrame()
    for i in range(repetitions):
        for f in os.listdir(path_to_experiment_configs):
            print(f)
            copyfile(os.path.join(path_to_experiment_configs, f), "config.json")
            environments, scores = test.main(algorithm)

            df = pd.DataFrame(scores)
            df['config'] = f
            df['env'] = environments
            df['rep'] = i
            df['mean'] = np.mean(scores, axis=1)
            df['std'] = np.std(scores, axis=1)
            full_results = full_results.append(df)

    full_results.to_csv('full_results.csv', index=False)
