from data_loader import MADataset
import sys
import re

conn_path = "/Users/zuorui/Desktop/multi_agent/data/all/numpy_data/connectivity/"
task_path = "/Users/zuorui/Desktop/multi_agent/data/all/numpy_data/agent/"
city_path = "/Users/zuorui/Desktop/multi_agent/data/all/numpy_data/task/"
reward_path = "/Users/zuorui/Desktop/multi_agent/data/all/numpy_data/reward/"
destination_path = "/Users/zuorui/Desktop/multi_agent/data/all/numpy_data/destination/"


dataset = MADataset(1, conn_path, task_path, city_path, reward_path, destination_path)
dataset = dataset[0]


def main(cplex_path):
    with open(cplex_path) as file:
        data = file.read().split('\n')
    print(data)
    c = 1
    agent = -1
    sum_reward = 0
    sum_cost = 0
    for e in data:
        if re.match('Agents*', e):
            print(e)
            c = 0
            agent = re.search('[0-9]+', e).group()
            sum_reward = 0.0
            sum_cost = 0.0
        elif c == 0:
            action = re.search('End:[0-9]*', e).group()
            action = re.search('[0-9]+', action).group()
            print(e, '    Reward', dataset['rewards'][int(agent)][int(action)],
                  '    Cost', dataset['tasks'][int(agent)][int(action)])
            sum_reward += dataset['rewards'][int(agent)][int(action)]
            sum_cost += dataset['tasks'][int(agent)][int(action)]
            c = 1
        elif re.match('Star*', e):
            from_ = re.search('Start:[0-9]*', e).group()
            from_ = re.search('[0-9]+', from_).group()
            action = re.search('End:[0-9]*', e).group()
            action = re.search('[0-9]+', action).group()
            if action != '23':
                print(e, '    Reward', dataset['rewards'][int(agent)][int(action)],
                      '    Cost', dataset['conn'][int(from_)][int(action)])
                sum_reward += dataset['rewards'][int(agent)][int(action)]
                sum_cost += dataset['conn'][int(from_)][int(action)]
            else:
                print(e)
                print('Sum Reward:', sum_reward, 'Sum_cost:', sum_cost)


if __name__ == '__main__':
    main(sys.argv[1])
