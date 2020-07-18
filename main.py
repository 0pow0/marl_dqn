import torch
from utils import arg_parser
from data_loader import MADataset
from torch.utils.data import DataLoader
from trainer import Trainer
from evaluator import Evaluator
from tqdm import tqdm
from memory import ReplayMemory
from torch.utils.tensorboard import SummaryWriter
import datetime
import numpy as np
import os

SEED = 0
DEVICE = 0
MEMORY_CAPACITY = 30000
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

torch.manual_seed(SEED)
args = arg_parser()


def save_model(checkpoint_dir, model_checkpoint_name, model):
    if not os.path.isdir(checkpoint_dir):
        os.mkdir(checkpoint_dir)
    model_save_path = '{}/{}'.format(checkpoint_dir, model_checkpoint_name)
    print('\nsave model to: \n{}'.format(model_save_path))
    torch.save(model.state_dict(), model_save_path)


def evl():
    dataset = MADataset(args.len_test_data, args.connectivity_path, args.task_path,
                        args.city_path, args.reward_path, args.destination_path, start_idx=args.test_start_idx)
    test_data_loader = DataLoader(dataset)
    evaluator = Evaluator(args, args.n_agents, args.n_cities, DEVICE, test_data_loader)
    evaluator.gen_env()

    for i in range(evaluator.n_envs):
        with open(evaluator.log_dir + "/env_" + str(args.test_start_idx+i) + ".log", "w+") as logger:
            logger.write("\nEnv {0}\n".format(args.test_start_idx+i))
            sum_rewards = 0.0
            for step in range(args.steps):
                logger.write("\n   Step {0}\n".format(step))
                rewards, actions, froms = evaluator.step(i)
                for j in range(evaluator.n_agents):
                    logger.write("\n      ID: {0}  FROM: {1}  ACTION: {2}  REWARD: {3}\n"
                                 .format(j, froms[j], actions[j], rewards[j]))
                sum_rewards += sum(rewards)
            logger.write("\n sum reward: {0}\n".format(sum_rewards))


def train():
    dataset = MADataset(args.len_dataset, args.connectivity_path, args.task_path,
                        args.city_path, args.reward_path, args.destination_path)
    memory = ReplayMemory(MEMORY_CAPACITY)

    train_data_loader = DataLoader(dataset)

    trainer = Trainer(args=args, n_agents=args.n_agents, n_cities=args.n_cities,
                      device=DEVICE, data_loader=train_data_loader)

    save_model(checkpoint_dir=args.checkpoint_dir + "/" + str(datetime.date.today()),
               model_checkpoint_name="encoder",
               model=trainer.Encoder)
    min_loss = np.inf
    best_reward = 0.0
    trainer.gen_env()
    for epoch in tqdm(range(args.epochs)):
        trainer.optimizer.zero_grad()
        rewards = 0.0
        for step in range(args.steps):
            transitions = trainer.step()
            rewards += sum([t.reward for t in transitions])
            memory.push_all(transitions)
        # passed all time-steps and reset all envs
        trainer.reset()
        samples = memory.sample(args.batch_size)
        # calc maxQ(s_t+1) then calc loss
        loss = trainer.calc_loss(samples)
        if loss < min_loss and epoch > 10:
            save_model(checkpoint_dir=args.checkpoint_dir + "/" + str(datetime.date.today()),
                       model_checkpoint_name="best_" + str(float(loss)),
                       model=trainer.DQN)
            min_loss = float(loss)
        if rewards > best_reward:
            save_model(checkpoint_dir=args.checkpoint_dir + "/" + str(datetime.date.today()),
                       model_checkpoint_name="best_reward_" + str(float(loss)),
                       model=trainer.DQN)
            best_reward = rewards
        loss.backward()
        trainer.optimizer.step()
        # memory.clear(MEMORY_CAPACITY)
        print("\n Epoch: {0} reward: {1} loss: {2}\n".format(epoch, rewards, loss.float()))

        if epoch % 10 == 0 and epoch != 0:
            trainer.target_DQN.load_state_dict(trainer.DQN.state_dict())


if __name__ == '__main__':
    train()

