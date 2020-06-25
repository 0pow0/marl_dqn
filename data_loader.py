from torch.utils.data import Dataset, DataLoader
import numpy as np

np.random.seed(0)


def data_loader(n, conn_path, task_path, cities_path, reward_path, destination_path, num_workers=2):
    dataset = MADataset(n, conn_path, task_path, cities_path, reward_path, destination_path)
    data_loader_ = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=num_workers)
    return data_loader_


class MADataset(Dataset):
    def __init__(self, n, connectivity_path, task_path, cities_path, reward_path, destination_path):
        self.n = n
        self.conn_path = connectivity_path
        self.task_path = task_path
        self.cities_path = cities_path
        self.rewards_path = reward_path
        self.destination_path = destination_path
        self.conn = []
        self.tasks = []
        self.cities = []
        self.rewards = []
        self.destinations = []
        self.init_dest = []
        self.load()

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        return {"conn": self.conn[idx],
                "tasks": self.tasks[idx],
                "cities": self.cities[idx],
                "rewards": self.rewards[idx],
                "destinations": self.destinations[idx]}

    def load(self):
        for i in range(self.n):
            self.conn.append(np.load(self.conn_path + "connectivity_" + str(i) + ".npy"))
            self.tasks.append(np.load(self.task_path + "agent_all_" + str(i) + ".npy"))
            self.cities.append(np.load(self.cities_path + "task_" + str(i) + ".npy"))
            self.rewards.append(np.load(self.rewards_path + "reward_" + str(i) + ".npy"))
            self.destinations.append(np.load(self.destination_path + "destination_" + str(i) + ".npy"))

    def info(self):
        return "=" * 24 + "data info" + "=" * 26 + "\n" \
                                                   "conn:         {0}    shape: {1}\n" \
                                                   "tasks:        {2}    shape: {3}\n" \
                                                   "cities:       {4}    shape: {5}\n" \
                                                   "rewards:      {6}    shape: {7}\n" \
                                                   "destinations: {8}    shape: {9}" \
            .format(len(self.conn), self.conn[0].shape,
                    len(self.tasks), self.tasks[0].shape,
                    len(self.cities), self.cities[0].shape,
                    len(self.rewards), self.rewards[0].shape,
                    len(self.destinations), self.destinations[0].shape) \
               + "\n" + "=" * 60

