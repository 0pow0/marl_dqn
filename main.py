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
            for step in range(args.steps):
                transitions = trainer.step()
                if transitions == "done":
                    break
                epoch_reward += sum([t.reward for t in transitions])
                memory.push_all(transitions)
            # passed all time-steps and reset
            trainer.env.reset()
            samples = memory.sample(args.batch_size)
            # TODO: calc maxQ(s_t+1) then calc loss




if __name__ == '__main__':
    main()
