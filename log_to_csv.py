import re
import sys
import os
import pandas


def log_to_csv(log_path, out_path, n_agent=3):
    n_agent = int(n_agent)
    env_name = re.findall('/env_[0-9]+\.log$', log_path)
    print(env_name)
    env_name = re.search('env_[0-9]+', env_name[0]).group(0)
    out_path = out_path + '/' + env_name
    os.mkdir(out_path)

    df = pandas.DataFrame()

    with open(log_path) as file:
        data = file.read().splitlines()
        data = list(filter(lambda e: e is not '', data))
        data = list(map(lambda e: e.strip(), data))
    print(data)
    for i in range(n_agent):
        reg_id = 'ID: {0}.*'.format(i)
        agent_data = list(map(lambda e: re.findall(reg_id, e), data))
        agent_data = list(filter(lambda e: len(e) > 0, agent_data))
        agent_data = [e[0] for e in agent_data]

        dFrom = list(map(lambda e: re.findall('.?FROM: .{0,2}', e), agent_data))
        dFrom = [e[0].strip() for e in dFrom]
        dFrom = list(map(lambda e: re.findall('..$', e), dFrom))
        dFrom = [e[0] for e in dFrom]
        print(dFrom)
        df['FROM'] = dFrom

        dAction = list(map(lambda e: re.findall('.?ACTION: .{0,14}', e), agent_data))
        dAction = [e[0].strip() for e in dAction]
        dAction = list(map(lambda e: re.findall('.{14}$', e), dAction))
        dAction = [e[0] for e in dAction]
        print(dAction)
        df['ACTION'] = dAction

        dReward = list(map(lambda e: re.findall('.?REWARD: .{0,5}', e), agent_data))
        dReward = [e[0].strip() for e in dReward]
        dReward = list(map(lambda e: re.findall('.{4}$', e), dReward))
        dReward = [e[0] for e in dReward]
        df['REWARD'] = dReward

        to_path = out_path + '/' + 'agent_{}'.format(i) + '.csv'
        df.to_csv(to_path, index=False)
        print(df)


if __name__ == '__main__':
    log_to_csv(sys.argv[1], sys.argv[2])
