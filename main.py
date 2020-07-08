import torch
from utils import arg_parser
from data_loader import MADataset
from torch.utils.data import DataLoader
from trainer import Trainer
from tqdm import tqdm
from memory import ReplayMemory
from torch.utils.tensorboard import SummaryWriter
import datetime
import numpy as np
import os

SEED = 0
DEVICE = 0
MEMORY_CAPACITY = 10000
LOG_DIR = "/home/rzuo02/data/mas_dqn/log/"
TENSOR_BOARD_PATH = '/home/rzuo02/PycharmProjects/mas_dqn/runs/Jul6/'

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
    trainer = Trainer(args=args, n_agents=args.n_agents, n_cities=args.n_cities,
                      device=DEVICE, data_loader=test_data_loader)
    trainer.DQN.load_state_dict(torch.load(args.load_from_main_checkpoint, map_location=torch.device(0)))

    for i in range(args.len_test_data):
        # os.mkdir("./runs/env_" + str(i))
        # writer = SummaryWriter(log_dir="./runs/env_"+str(i))
        trainer.gen_env()
        reward = 0.0
        log_dir = LOG_DIR + str(datetime.date.today())
        if not os.path.isdir(log_dir):
            os.mkdir(log_dir)
        with open(log_dir + "/env_" + str(i) + ".log", "w+") as logger:
            logger.write("\nEnv {0}\n".format(i))
            for step in range(args.steps):
                transitions = trainer.step(need_eps=False)
                if transitions == "done":
                    break
                logger.write("\n   Step {0}\n".format(step))
                for j in range(len(transitions)):
                    logger.write("\n      ID: {0}  FROM: {1}  ACTION: {2}  REWARD: {3}\n"
                                 .format(j, transitions[j].from_, transitions[j].action, transitions[j].reward))
                reward += sum([t.reward for t in transitions])
            logger.write("\n sum reward: {0}\n".format(reward))


def main():
    dataset = MADataset(args.len_dataset, args.connectivity_path, args.task_path,
                        args.city_path, args.reward_path, args.destination_path)
    memory = ReplayMemory(MEMORY_CAPACITY)

    test_data_loader = None
    if args.need_eval:
        trainSize = int(args.split_ratio * len(dataset))
        testSize = len(dataset) - trainSize
        trainDataset, testDataset = torch.utils.data.random_split(dataset, [trainSize, testSize])
        train_data_loader = DataLoader(trainDataset)
        test_data_loader = DataLoader(testDataset)
    else:
        train_data_loader = DataLoader(dataset)

    trainer = Trainer(args=args, n_agents=args.n_agents, n_cities=args.n_cities,
                      device=DEVICE, data_loader=train_data_loader)

    min_loss = np.inf
    for i in range(args.n_envs):
        writer = SummaryWriter()
        trainer.gen_env()
        print("+----------------------------------------------+\n"
              "|               Environment: {0}                 |\n"
              "+----------------------------------------------+\n".format(trainer.idx_env))
        ten_epoch_reward = 0.0
        max_reward = 0.0
        # with open(LOG_DIR, "w+") as logger:
        for epoch in tqdm(range(args.epochs)):
            trainer.optimizer.zero_grad()
            rewards = 0.0
            # logger.write("\nEpoch {0}\n".format(epoch))
            for step in range(args.steps):
                transitions = trainer.step()
                if transitions == "done":
                    break
                # logger.write("\n   Step {0}\n".format(step))
                # for j in range(len(transitions)):
                #     logger.write("\n      ID: {0}  FROM: {1}  ACTION: {2}  REWARD: {3}\n"
                #                  .format(j, transitions[j].from_, transitions[j].action, transitions[j].reward))
                rewards += sum([t.reward for t in transitions])
                memory.push_all(transitions)
                # if step == int(args.steps/2):
                #     writer.add_scalar('Q', transitions[0].Q[transitions[0].action[1]], epoch)
            ten_epoch_reward += rewards
            max_reward = max(max_reward, rewards)
            # passed all time-steps and reset
            trainer.env.reset()
            if epoch % 10 == 0:
                samples = memory.sample(args.batch_size)
                # calc maxQ(s_t+1) then calc loss
                loss = trainer.calc_loss(samples)
                if loss < min_loss and epoch >= 20:
                    save_model(checkpoint_dir=args.checkpoint_dir + "/" + str(datetime.date.today()),
                               model_checkpoint_name="best",
                               model=trainer.DQN)
                    min_loss = loss
                loss.backward()
                trainer.optimizer.step()
                memory.clear(MEMORY_CAPACITY)
                print("\n Epoch: {0} reward: {1} loss: {2}\n".format(epoch, ten_epoch_reward, loss.float()))
                # writer.add_scalar('Loss', loss, epoch)
                # writer.add_scalar('Sum of ten epoch reward', ten_epoch_reward, epoch)
                # writer.add_scalar('max reward in ten epoch', max_reward, epoch)
                ten_epoch_reward = 0.0
                max_reward = 0.0


if __name__ == '__main__':
    evl()
