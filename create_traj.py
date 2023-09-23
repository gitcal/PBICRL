import math
import random
import pickle
import gym
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal

from IPython.display import clear_output
import matplotlib.pyplot as plt
from matplotlib import animation
from IPython.display import display
import IPython
from custom_env_w import Point

use_cuda = torch.cuda.is_available()
device   = torch.device("cuda" if use_cuda else "cpu")



class PolicyNetwork(nn.Module):

	def __init__(self, num_inputs, num_actions, hidden_size, init_w=3e-3, log_std_min=-20, log_std_max=2):
		super(PolicyNetwork, self).__init__()
		
		self.log_std_min = log_std_min
		self.log_std_max = log_std_max
		
		self.linear1 = nn.Linear(num_inputs, hidden_size)
		self.linear2 = nn.Linear(hidden_size, hidden_size)
		
		self.mean_linear = nn.Linear(hidden_size, num_actions)
		self.mean_linear.weight.data.uniform_(-init_w, init_w)
		self.mean_linear.bias.data.uniform_(-init_w, init_w)
		
		self.log_std_linear = nn.Linear(hidden_size, num_actions)
		self.log_std_linear.weight.data.uniform_(-init_w, init_w)
		self.log_std_linear.bias.data.uniform_(-init_w, init_w)
		
	def forward(self, state):

		x = F.relu(self.linear1(state))
		x = F.relu(self.linear2(x))
		
		mean    = self.mean_linear(x)
		log_std = self.log_std_linear(x)
		log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
		
		return mean, log_std
	
	def evaluate(self, state, epsilon=1e-6):

		mean, log_std = self.forward(state)
		std = log_std.exp()
		
		normal = Normal(0, 1)
		z      = normal.sample()
		action = torch.tanh(mean+ std*z.to(device))
		log_prob = Normal(mean, std).log_prob(mean+ std*z.to(device)) - torch.log(1 - action.pow(2) + epsilon)
		return action, log_prob, z, mean, log_std
		
	
	def get_action(self, state):

		state = torch.FloatTensor(state).unsqueeze(0).to(device)
		mean, log_std = self.forward(state)
		std = log_std.exp()
		
		normal = Normal(0, 1)
		z      = normal.sample().to(device)
		action = torch.tanh(mean + std*z)
		
		action  = action.cpu()#.detach().cpu().numpy()
		return action[0]

goal = np.array([20.0,20.0])
bad_cnstr = np.array([5.5,14.5])
very_bad_cnstr = np.array([14.5,5.5])

r_constr = 3
env = Point('name', 20, -1, 20, -1, goal, bad_cnstr, very_bad_cnstr,0,0)#gym.make(args.env_name)


action_dim = 2#env.action_space.shape[0]
state_dim  = 2#env.observation_space.shape[0]
hidden_dim = 256

policy_net_bad1 = PolicyNetwork(state_dim, action_dim, hidden_dim).to(device)
policy_net_bad2 = PolicyNetwork(state_dim, action_dim, hidden_dim).to(device)
policy_net_good = PolicyNetwork(state_dim, action_dim, hidden_dim).to(device)
policy_net_nocnstr = PolicyNetwork(state_dim, action_dim, hidden_dim).to(device)


policy_net_good.load_state_dict(torch.load('model/policy_net_cnstr_-10.0_-100.0.pth'))

policy_net_nocnstr.load_state_dict(torch.load('model/policy_net_cnstr_0.0_0.0.pth'))

N_tot_traj = 1500
N_max_episode = 60
N_good_traj = 750

traj_dict = {}
cum_reward = []
states_list = []
traj_index = []
for iteration in range(N_tot_traj):
	print(iteration)
	states_l = []
	state = np.zeros(2)
	if np.random.rand()<0.5:
		state[0] = np.random.uniform(9,19)
		state[1] = 1*np.random.rand()
	else:
		state[0] = 1*np.random.rand()
		state[1] = np.random.uniform(9,19)

	env.set_position(state[0], state[1])
	states_l.append(state)
	cum_r = 0
	
	for t in range(N_max_episode):
		# Render into buffer. 
		if iteration<=N_good_traj:
			action = policy_net_good.get_action(state)
			ind = 1
		elif iteration>N_good_traj and iteration<=N_tot_traj:
			action = policy_net_nocnstr.get_action(state)
			ind = 2
		else:
			action = policy_net_good.get_action(state)
			ind = 3

		action = np.clip(action.detach(),-1, 1)
		state, reward, done, info = env.step(action)
		if done:
			print(t)
		states_l.append(state)
		cum_r += reward
		if done or t>=N_max_episode-1:
			print(t)
			cum_reward.append(cum_r)
			states_list.append(states_l)
			traj_index.append(ind)
			break
	print(np.linalg.norm(state-env.target)<=env.r_target)

traj_index = np.zeros(len(states_list))
cnt = 0
for traj in states_list:
	traj_index[cnt] = 1
	for tr in traj:
		if np.linalg.norm(tr-bad_cnstr) < r_constr:
			traj_index[cnt] = 2
		if np.linalg.norm(tr-very_bad_cnstr) < r_constr:
			traj_index[cnt] = 3

	cnt += 1


traj_dict['trajectories'] = states_list
traj_dict['rewards'] = cum_reward
traj_dict['traj_index'] = traj_index

with open('data/trajectories_m10_m100.pickle', 'wb') as handle:
	pickle.dump(traj_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
IPython.embed()


