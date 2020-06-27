import torch
from utils import arg_parser
from data_loader import MADataset
from torch.utils.data import DataLoader
from trainer import Trainer
from tqdm import tqdm
from memory import ReplayMemory
from torch.utils.tensorboard import SummaryWriter
import datetime

SEED = 0
DEVICE = 0
MEMORY_CAPACITY = 10000
LOG_PATH = "/home/multiagent_share/ruiz/{0}.log".format(datetime.date.today())
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

    writer = SummaryWriter()

    for i in range(args.n_envs):
        trainer.gen_env()
        print("+----------------------------------------------+\n"
              "|               Environment: {0}                 |\n"
              "+----------------------------------------------+\n".format(trainer.idx_env))
        ten_epoch_reward = 0.0
        max_reward = 0.0
        with open(LOG_PATH, "w+") as logger:
            for epoch in tqdm(range(args.epochs)):
                trainer.optimizer.zero_grad()
                rewards = 0.0
                logger.write("\nEpoch {0}\n".format(epoch))
                for step in range(args.steps):
                    transitions = trainer.step()
                    if transitions == "done":
                        break
                    logger.write("\n   Step {0}\n".format(step))
                    for j in range(len(transitions)):
                        logger.write("\n      ID: {0}  FROM: {1}  ACTION: {2}  REWARD: {3}\n"
                                     .format(j, transitions[j].from_, transitions[j].action, transitions[j].reward))
                    rewards += sum([t.reward for t in transitions])
                    memory.push_all(transitions)
                ten_epoch_reward += rewards
                max_reward = max(max_reward, rewards)
                # passed all time-steps and reset
                trainer.env.reset()
                if epoch % 10 == 0:
                    samples = memory.sample(args.batch_size)
                    # calc maxQ(s_t+1) then calc loss
                    loss = trainer.calc_loss(samples)
                    loss.backward()
                    trainer.optimizer.step()
                    memory.clear(MEMORY_CAPACITY)
                    print("\n Epoch: {0} reward: {1} loss: {2}\n".format(epoch, ten_epoch_reward, loss.float()))
                    writer.add_scalar('Loss', loss, epoch)
                    writer.add_scalar('Sum of ten epoch reward', ten_epoch_reward, epoch)
                    writer.add_scalar('max reward in ten epoch', max_reward, epoch)
                    ten_epoch_reward = 0.0
                    max_reward = 0.0
        logger.close()




if __name__ == '__main__':
    main()
