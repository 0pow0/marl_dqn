from utils import load_data
from utils import arg_parser
from gcn.vGraph import vGraph
import torch

args = arg_parser()
data_loader = load_data(args.len_dataset, args.connectivity_path, args.task_path,
                        args.city_path, args.reward_path, args.destination_path)
data_loader = list(data_loader)

G = vGraph(20, data_loader[0]['conn'].squeeze(), data_loader[0]['rewards'].squeeze().T)
waveA = G.adj()
waveD = G.deg(waveA)
X = G.X()

hatA = torch.chain_matmul(torch.inverse(torch.sqrt(waveD)).double(), waveA, torch.inverse(torch.sqrt(waveD)).double())
print()


