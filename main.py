import torch
from utils import arg_parser
from data_loader import MADataset
from torch.utils.data import DataLoader
from trainer import Trainer
from tqdm import tqdm
from memory import ReplayMemory

SEED = 0
DEVICE = 0
MEMORY_CAPACITY = 10000
# os.environ["CUDA_VISIBLE_DEVICES"] = "3"


def main():
    torch.manual_seed(SEED)
    args = arg_parser()

    dataset = MADataset(args.len_dataset, args.connectivity_path, args.task_path,
                        args.city_path, args.reward_path, args.destination_path)
    data_loader = DataLoader(dataset)

    trainer = Trainer(args=args, n_agents=args.n_agents, n_cities=args.n_cities,
                      device=DEVICE, data_loader=data_loader)

    memory = ReplayMemory(MEMORY_CAPACITY)

    for i in range(args.n_envs):
        trainer.gen_env()
        print("+----------------------------------------------+\n"
              "|               Environment: {0}                 |\n"
              "+----------------------------------------------+\n".format(trainer.idx_env))
        for epoch in tqdm(range(args.epochs)):
            epoch_reward = 0.0
            trainer.optimizer.zero_grad()

            for step in range(args.steps):
                transitions = trainer.step()
                if transitions == "done":
                    break
                epoch_reward += sum([t.reward for t in transitions])
                memory.push_all(transitions)
            # passed all time-steps and reset
            trainer.env.reset()
            if epoch % 10 == 0:
                samples = memory.sample(args.batch_size)
                # print("\n")
                # print(samples)
                # calc maxQ(s_t+1) then calc loss
                loss = trainer.calc_loss(samples)
                loss.backward()
                # for param in trainer.DQN.parameters():
                #     param.grad.data.clamp_(-1, 1)
                trainer.optimizer.step()
            # if epoch % 100 == 0:
                print("\n Epoch: {0} reward: {1} loss: {2}\n".format(epoch, epoch_reward, loss.float()))



if __name__ == '__main__':
    main()
