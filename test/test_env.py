import datetime
from torch.utils.data import DataLoader
from env import Env
from data_loader import MADataset
from utils import arg_parser
import random
import os

TEST_DIR = "/home/rzuo02/data/mas_dqn/test_env/" + str(datetime.date.today())
if not os.path.isdir(TEST_DIR):
    os.mkdir(TEST_DIR)
TEST_TIME = 10
args = arg_parser()

dataset = MADataset(args.len_dataset, args.connectivity_path, args.task_path,
                    args.city_path, args.reward_path, args.destination_path)

data_loader = DataLoader(dataset)

env = None
for data in data_loader:
    env = Env(n_agents=args.n_agents, n_cities=args.n_cities, steps=args.steps,
              conn=data["conn"], tasks=data["tasks"], cities=data["cities"],
              rewards=data["rewards"], destinations=data["destinations"],
              budget=80)

with open(TEST_DIR + "/" + "res1.log", "w+") as logger:
    for time in range(TEST_TIME):
        logger.write("\nTIME : {}\n".format(time))
        for step in range(args.steps):
            logger.write("\n    STEP : {}\n".format(step))
            actions = []
            for i in range(args.n_agents):
                cand_city = []
                for city in env.world.cities:
                    if (env.world.agents[i].task[0][city.ID] != -1) and (env.world.agents[i].budget >= env.world.agents[i].task[0][city.ID]):
                        cand_city.append(city.ID)
                if len(cand_city) == 0:
                    actions.append([0, env.world.agents[i].at_city.ID])
                else:
                    actions.append([random.choice([0, 1])] + [random.choice(cand_city)])
            froms = [env.world.agents[j].at_city.ID
                     if env.world.agents[j].at_city is not None else -1
                     for j in range(args.n_agents)]
            rewards = env.step(actions)
            for j in range(args.n_agents):
                logger.write("\n          ID: {0}  FROM: {1}  ACTION: {2}  REWARD: {3}\n"
                             .format(j, froms[j], actions[j], rewards[j]))
        env.reset()





