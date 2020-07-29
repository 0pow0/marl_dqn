# Multi-agent System with Deep Q-learning Network



Weekly update: Jul 28th

- Distance is accumulated from start to each step, and env will allocate reward back based on this distance.
- Mask cities that remain budget is not sufficient to reach.

- If agent has made penalty on reward, env will no make penalty agin. Ex. prior, agent has -3, env made another -3-3=6; currently, will be -3
- Reward will keep same with agent returned budget if agent has already made penalty on reward. Prior this version, reward will be 0 not excepted negative reward returned by agent. Ex. Prior, agent give back -3, and env make -3 to 0; currently will be -3.



