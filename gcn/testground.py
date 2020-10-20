from utils import load_data
from utils import arg_parser
from gcn.vGraph import vGraph

args = arg_parser()
data_loader = load_data(args.len_dataset, args.connectivity_path, args.task_path,
                        args.city_path, args.reward_path, args.destination_path)
data_loader = list(data_loader)

G = vGraph(20, data_loader[0]['conn'].squeeze(), data_loader[0]['rewards'].squeeze().T)
waveA = G.adj()
waveD = G.deg(waveA)
X = G.X()
print()
