import argparse
from data_loader import MADataset
from torch.utils.data import DataLoader


def load_data(len_data, connectivity_path, task_path, city_path, reward_path, destination_path):
    data_set = MADataset(len_data, connectivity_path, task_path, city_path, reward_path, destination_path)
    data_loader = DataLoader(data_set)
    return data_loader


def arg_parser():
    parser = argparse.ArgumentParser()
    # necessary path
    parser.add_argument("--connectivity_path", help="connectivity of cities", required=True, type=str)
    parser.add_argument("--task_path", help="task here means initial distance between one uav to all cities",
                        required=True, type=str)
    parser.add_argument("--city_path", help="coordination of cities", required=True, type=str)
    parser.add_argument("--reward_path", help="reward for each UAVs at all cities", required=True, type=str)
    parser.add_argument("--destination_path", help="distance from cities to destination", required=True, type=str)
    parser.add_argument("--log_dir_path", type=str)
    parser.add_argument("--tensor_board_path", type=str)

    # hyper parameters
    parser.add_argument("--len_encoder", help="output shape (1,L) of embedding network", type=int)
    parser.add_argument("--steps", help="total steps for one env", type=int)
    parser.add_argument("--epochs", help="num_epochs", type=int)
    parser.add_argument("--n_envs", help="load first n data", type=int)
    parser.add_argument("--n_agents", help="number of agent in env", type=int)
    parser.add_argument("--n_cities", help="number of cities in env", type=int)
    parser.add_argument("--len_dataset", help="number of envs", type=int)
    parser.add_argument("--gamma", help="target=reward+gamma*Q_t+1", type=float)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--budget", type=int)
    parser.add_argument("--eps_start", help="initial epsilon and will decay toward eps_end", type=float)
    parser.add_argument("--eps_end", help="end epsilon", type=float)
    parser.add_argument("--eps_decay", help="decay rate", type=float)


    # dirs
    parser.add_argument("--checkpoint_dir", help="checkpoint_dir", type=str)

    # optional
    # consuming for each uav in one time-step
    parser.add_argument("--step_consuming", type=int)
    # whether set reward to unit mode, which means use
    # the unit reward = reward / distance to measure performance
    parser.add_argument("--unit_reward", type=bool)
    parser.add_argument("--load_from_main_checkpoint", type=str)
    parser.add_argument("--encoder_checkpoint", type=str)
    parser.add_argument("--len_test_data", type=int)
    parser.add_argument("--test_start_idx", type=int)

    args, unknown = parser.parse_known_args()
    return args
