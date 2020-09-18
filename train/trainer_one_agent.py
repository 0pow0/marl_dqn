import random
import math
import sys
import os
from tqdm import tqdm
from models import Encoder, DQN, testDQN2
from env import Env
import torch
import torch.nn as nn
import torch.optim
from utils import arg_parser
from data_loader import MADataset
from torch.utils.data import DataLoader
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from memory import ExtendTransition
from memory import ReplayMemoryWithSet

MEMORY_CAPACITY = 1000
DEVICE = 0


def testing_ground(training_detail_log_path):
    best_loss = np.inf
    writer = SummaryWriter()
    args = arg_parser()
    dataset = MADataset(args.len_dataset, args.connectivity_path, args.task_path,
                        args.city_path, args.reward_path, args.destination_path)
    dataloader = DataLoader(dataset)
    trainer = OneEnvTrainer(dataloader, args, device=0)
    EPOCHs = args.epochs
    memory = ReplayMemoryWithSet(MEMORY_CAPACITY)
    with open(training_detail_log_path, "w+") as logger:
        for epoch in tqdm(range(EPOCHs)):
            transitions, epoch_states, epoch_next_state, epoch_action, epoch_reward, epoch_values, eps_threshold \
                = trainer.agent0_forward_one_epoch()
            memory.push_all(transitions)

            for step in range(args.steps):
                if epoch >= args.epochs * 0.9:
                    logger.write("\n({0}/{1}/{2}): ACTION: {3} REWARD: {4} Q: {5}"
                                 .format(epoch, step, 0, int(epoch_action[step].cpu().detach()), epoch_reward[step],
                                         epoch_values[step].cpu().detach()))
                    logger.write("\n Budget: {0} "
                                 "Distance: {1} "
                                 "\nTask: {2} "
                                 "\nReward: {3} "
                                 "\nHistory: {4}".format(epoch_states[step][0],
                                                         epoch_states[step][1],
                                                         epoch_states[step][2: 2 + args.n_cities],
                                                         epoch_states[step][2 + args.n_cities:
                                                                            2 + args.n_cities * 2],
                                                         epoch_states[step][2 + args.n_cities * 2:
                                                                            2 + args.n_cities * 2 + 2 * args.n_cities]))

            if epoch % 10 == 0:
                # tqdm.write("\n Epoch: {0} \nreward: {1}\n".format(epoch, epoch_reward), end="")
                if len(memory) > args.batch_size:
                    samples = memory.sample(args.batch_size)
                    loss = trainer.calc_loss_and_optimize(samples)
                    if loss < best_loss:
                        save_model(checkpoint_dir=args.checkpoint_dir + "/",
                                   model_checkpoint_name=str(float(loss)),
                                   model=trainer.DQN)
                        best_loss = float(loss)
                    writer.add_scalar("Loss", loss, epoch)
            writer.add_scalar("EPS", eps_threshold, epoch)
            trainer.reset()


def evl(evaluation_log_path):
    args = arg_parser()
    dataset = MADataset(args.len_dataset, args.connectivity_path, args.task_path,
                        args.city_path, args.reward_path, args.destination_path)
    dataloader = list(DataLoader(dataset))
    env = Env(n_agents=args.n_agents, n_cities=args.n_cities, steps=args.steps,
              conn=dataloader[0]["conn"], tasks=dataloader[0]["tasks"],
              cities=dataloader[0]["cities"], rewards=dataloader[0]["rewards"],
              destinations=dataloader[0]["destinations"], budget=args.budget)

    DQN = testDQN2(4).to(DEVICE)
    DQN.load_state_dict(torch.load(args.load_from_main_checkpoint, map_location=torch.device(DEVICE)))
    with open(evaluation_log_path, "w+") as logger:
        for step in range(args.steps):
            states_all_agents = env.input().reshape(args.n_agents, -1)
            state_agent_0 = states_all_agents[0]
            conns = state_agent_0[2: 2 + args.n_cities]
            rewards = state_agent_0[2 + args.n_cities: 2 + args.n_cities * 2]
            history = state_agent_0[2 + args.n_cities * 2:]
            state_for_each_city = torch.cat(
                [torch.stack([state_agent_0[0], rewards[i], conns[i], history[i + args.n_cities]])
                 for i in range(args.n_cities)]).reshape(args.n_cities, -1)
            value = DQN(state_for_each_city.to(DEVICE)).reshape(-1)
            action_city = value.max(0)[1]
            reward = env.step_one_agent([1, action_city], 0)
            logger.write("\nStep:{0}    Action:{1}    Reward:{2}".format(step, int(action_city), reward))
            logger.write("\nQ:{}".format(value))
            logger.write("\n Budget: {0} "
                         "Distance: {1} "
                         "\nTask: {2} "
                         "\nReward: {3} "
                         "\nHistory: {4}".format(state_agent_0[0],
                                                 state_agent_0[1],
                                                 conns,
                                                 rewards,
                                                 history))


def save_model(checkpoint_dir, model_checkpoint_name, model):
    if not os.path.isdir(checkpoint_dir):
        os.mkdir(checkpoint_dir)
    model_save_path = '{}/{}'.format(checkpoint_dir, model_checkpoint_name)
    print('\nsave model to: \n{}'.format(model_save_path))
    torch.save(model.state_dict(), model_save_path)


class OneEnvTrainer(object):
    def __init__(self, dataloader, args, device):
        self.args = args
        self.n_agents = args.n_agents
        self.n_cities = args.n_cities
        self.dataloader = list(dataloader)
        self.env = Env(n_agents=self.n_agents, n_cities=self.n_cities, steps=self.args.steps,
                       conn=self.dataloader[0]["conn"], tasks=self.dataloader[0]["tasks"],
                       cities=self.dataloader[0]["cities"], rewards=self.dataloader[0]["rewards"],
                       destinations=self.dataloader[0]["destinations"], budget=self.args.budget)
        self.device = device
        self.DQN = testDQN2(4).to(self.device)

        self.EPS_START = self.args.eps_start
        self.EPS_END = self.args.eps_end
        self.EPS_DECAY = self.args.eps_decay
        self.GAMMA = self.args.gamma

        self.mse = nn.MSELoss()
        self.adam = torch.optim.Adam(self.DQN.parameters(), lr=args.lr)

        self.epoch_finished = 0

    def calc_loss_and_optimize(self, samples):
        batch = ExtendTransition(*zip(*samples))
        batch_states = torch.cat(batch.state).reshape(self.args.batch_size, -1)
        batch_actions = torch.cat(batch.action).reshape(-1, 1, 1)
        batch_rewards = batch.reward
        batch_next_states = torch.cat(batch.next_state).reshape(self.args.batch_size, 1, self.n_cities, -1)

        batch_states = batch_states.to(self.device)
        batch_next_states = batch_next_states.to(self.device)

        values = self.DQN(batch_states)
        batch_next_states = batch_next_states.squeeze().reshape(-1, 4)
        next_state_values = self.DQN(batch_next_states).reshape(self.args.batch_size, self.n_cities, -1).max(1)[0]
        expected_values = [next_state_values[i] * self.GAMMA + batch_rewards[i][1]
                           if batch_rewards[i][0] != -1
                           else torch.tensor(0.0).reshape(1).to(self.device)
                           for i in range(self.args.batch_size)]
        expected_values = torch.cat(expected_values).reshape(-1, 1)
        # expected_values = torch.tensor([batch_rewards[i][1] for i in range(len(batch_rewards))]).to(self.device)
        # expected_values = expected_values.reshape(-1, 1)

        loss = self.mse(values, expected_values)
        loss.backward()
        self.adam.step()
        return loss

    def reset(self):
        self.env.reset()

    def agent0_forward_one_step(self, eps_threshold):
        p = random.random()
        states_all_agents = self.env.input().reshape(self.n_agents, -1)
        state_agent_0 = states_all_agents[0]
        conns = state_agent_0[2: 2 + self.n_cities]
        rewards = state_agent_0[2 + self.n_cities: 2 + self.n_cities * 2]
        history = state_agent_0[2 + self.n_cities * 2:]
        state_for_each_city = torch.cat(
            [torch.stack([state_agent_0[0], rewards[i], conns[i], history[i + self.n_cities]])
             for i in range(self.n_cities)]).reshape(self.n_cities, -1)
        value = self.DQN(state_for_each_city.to(self.device)).reshape(-1)
        if p > eps_threshold:
            action_city = value.max(0)[1]
        else:
            action_city = torch.tensor(random.randint(0, self.n_cities - 1)).to(self.device)
        reward = self.env.step_one_agent([1, action_city], 0)
        next_state_agent_0 = self.env.input().reshape(self.n_agents, -1)[0]
        next_conns = state_agent_0[2: 2 + self.n_cities]
        next_rewards = state_agent_0[2 + self.n_cities: 2 + self.n_cities * 2]
        next_history = state_agent_0[2 + self.n_cities * 2:]
        next_state_for_each_city = torch.cat(
            [torch.stack([next_state_agent_0[0], next_rewards[i], next_conns[i], next_history[i + self.n_cities]])
             for i in range(self.n_cities)]).reshape(self.n_cities, -1)
        return state_agent_0, next_state_agent_0, action_city.reshape(1), reward, value, \
               state_for_each_city[action_city], next_state_for_each_city

    def agent0_forward_one_epoch(self):
        epoch_states = []
        epoch_next_state = []
        epoch_action = []
        epoch_reward = []
        epoch_values = []
        eps_threshold = \
            self.EPS_END + (self.EPS_START - self.EPS_END) * math.exp(-1. * self.epoch_finished / self.EPS_DECAY)
        # print(str(eps_threshold) + "\n")
        transitions = set()
        for step in range(self.args.steps):
            state, next_state, action, reward, value, state_for_action, next_state_for_action \
                = self.agent0_forward_one_step(eps_threshold)
            epoch_states.append(state)
            epoch_next_state.append(next_state)
            epoch_action.append(action)
            epoch_reward.append(reward)
            epoch_values.append(value)
            transitions.add(ExtendTransition(state=state_for_action, action=action,
                                             next_state=next_state_for_action, reward=reward))
        self.epoch_finished += 1
        return list(
            transitions), epoch_states, epoch_next_state, epoch_action, epoch_reward, epoch_values, eps_threshold

    def calc_loss_and_optim_for_one_epoch(self, epoch_states, epoch_next_state, epoch_action, epoch_reward):
        epoch_states = torch.cat(epoch_states).to(self.device).reshape(self.args.steps, -1)
        epoch_next_state = torch.cat(epoch_next_state).to(self.device).reshape(self.args.steps, -1)
        epoch_action = torch.cat(epoch_action).to(self.device).reshape(-1, 1)

        values = self.DQN(epoch_states)
        values = values.gather(1, epoch_action)
        expected_values = (self.args.gamma * self.DQN(epoch_next_state)).max(1)[0]
        expected_values = [(expected_values[i] + epoch_reward[i][1]).reshape(1)
                           if epoch_reward[i][0] != -1
                           else torch.tensor(0.0, dtype=torch.float).reshape(1).to(self.device)
                           for i in range(self.args.steps)]
        expected_values = torch.cat(expected_values).to(self.device).reshape(-1, 1)
        loss = self.mse(values, expected_values)
        loss.backward()
        self.adam.step()
        return values, expected_values, loss


if __name__ == '__main__':
    # testing_ground(sys.argv[1])
    evl(sys.argv[2])
