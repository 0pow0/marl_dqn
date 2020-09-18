import torch
from utils import arg_parser
from data_loader import MADataset
from torch.utils.data import DataLoader
from trainer import Trainer
from eval.evaluator import Evaluator
from tqdm import tqdm
from memory import ReplayMemory
import datetime
import numpy as np
import os
import sys
from torch.utils.tensorboard import SummaryWriter

SEED = 0
DEVICE = 0
MEMORY_CAPACITY = 30000
TARGET_DQN_UPDATE_PERIOD = 10
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

torch.manual_seed(SEED)
args = arg_parser()
writer = SummaryWriter()


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


def train_detail_log(log_output_path):
    dataset = MADataset(args.len_dataset, args.connectivity_path, args.task_path,
                        args.city_path, args.reward_path, args.destination_path)
    memory = ReplayMemory(MEMORY_CAPACITY)
    train_data_loader = DataLoader(dataset)
    trainer = Trainer(args=args, n_agents=args.n_agents, n_cities=args.n_cities,
                      device=DEVICE, data_loader=train_data_loader)
    trainer.gen_env()
    min_loss = np.inf
    example_state = trainer.envs[0].input().reshape(1, -1)
    with open(log_output_path, "w+") as logger:
        for epoch in tqdm(range(args.epochs)):
            trainer.adam.zero_grad()
            epoch_reward = 0.0
            for step in range(args.steps):
                transitions = trainer.step()
                for agent in range(args.n_agents):
                    Q = trainer.DQN(transitions[agent].state.to(DEVICE))\
                        .detach().cpu().gather(0, transitions[agent].action[0][1])
                    if transitions[agent].reward[0][0] != 1:
                        Q_target = 0.0
                    else:
                        Q_target = trainer.DQN(transitions[agent].next_state.to(DEVICE)).detach().cpu().max() \
                                   * args.gamma + transitions[agent].reward[0][1]
                    logger.write("\n({0}/{1}/{2}): ACTION: {3} REWARD: {4} Q: {5} Q*: {6}"
                                 .format(epoch, step, agent, transitions[agent].action, transitions[agent].reward, Q, Q_target))
                    logger.write("\n Budget: {0} "
                                 "Distance: {1} "
                                 "Task: {2} "
                                 "Reward: {3} "
                                 "History: {4}".format(transitions[agent].state[0][0],
                                                       transitions[agent].state[0][1],
                                                       transitions[agent].state[0][2: 2+args.n_cities],
                                                       transitions[agent].state[0][2 + args.n_cities:
                                                                                   2 + args.n_cities * 2],
                                                       transitions[agent].state[0][2 + args.n_cities * 2:
                                                                                   2 + args.n_cities * 2 +
                                                                                   2 * args.n_cities]))
                    if step == 5 and agent == 0:
                        writer.add_scalars("Q and Q*", {"Q": Q, "Q*": Q_target}, epoch)
                epoch_reward += sum([t.reward[0][1] for t in transitions])
                memory.push_all(transitions)
            trainer.reset()
            if len(memory.memory) < args.batch_size:
                continue
            samples = memory.sample(args.batch_size)
            loss = trainer.calc_loss(samples)
            if loss < min_loss and epoch > 10:
                save_model(checkpoint_dir=args.checkpoint_dir + "/" + str(datetime.date.today()),
                           model_checkpoint_name="best_" + str(float(torch.mean(loss))),
                           model=trainer.DQN)
                min_loss = float(loss)
            writer.add_scalar("Loss", loss, epoch)
            writer.add_scalar("Epoch Reward", epoch_reward, epoch)
            loss.backward()
            trainer.adam.step()
            print("\n Epoch: {0} reward: {1} loss: {2}\n".format(epoch, epoch_reward, loss))


def train():
    dataset = MADataset(args.len_dataset, args.connectivity_path, args.task_path,
                        args.city_path, args.reward_path, args.destination_path)
    memory = ReplayMemory(MEMORY_CAPACITY)

    train_data_loader = DataLoader(dataset)

    trainer = Trainer(args=args, n_agents=args.n_agents, n_cities=args.n_cities,
                      device=DEVICE, data_loader=train_data_loader)

    min_loss = np.inf
    trainer.gen_env()
    example_state = trainer.envs[0].input().reshape(1, -1)
    for epoch in tqdm(range(args.epochs)):
        trainer.adam.zero_grad()
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
                       model_checkpoint_name="best_" + str(float(torch.mean(loss))),
                       model=trainer.DQN)
            min_loss = float(loss)

        loss.backward()
        # Adam
        trainer.adam.step()
        # RMSProp
        # trainer.optimizer.step()
        writer.add_scalar("Loss", loss, epoch)
        writer.add_scalar("Q", trainer.DQN(example_state.to(DEVICE)).max().detach().cpu(), epoch)
        if epoch % TARGET_DQN_UPDATE_PERIOD == 0:
            trainer.target_DQN.load_state_dict(trainer.DQN.state_dict())

        print("\n Epoch: {0} reward: {1} loss: {2}\n".format(epoch, rewards, loss))


if __name__ == '__main__':
    train_detail_log(sys.argv[1])
    # evl()


