import math
import random

# import gym
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal
import pickle, argparse

from IPython.display import clear_output
import matplotlib.pyplot as plt
from matplotlib import animation
from IPython.display import display
import IPython
from custom_env_w import Point

use_cuda = torch.cuda.is_available()
device   = torch.device("cuda" if use_cuda else "cpu")


parser = argparse.ArgumentParser()
parser.add_argument('--flname', type=str, default='policy_net_cnstr_')
parser.add_argument('--pen_1', type=float, default=-10)
parser.add_argument('--pen_2', type=float, default=-20)
args = parser.parse_args()


class ReplayBuffer:

	def __init__(self, capacity):
		self.capacity = capacity
		self.buffer = []
		self.position = 0
	
	def push(self, state, action, reward, next_state, done):

		if len(self.buffer) < self.capacity:
			self.buffer.append(None)
		self.buffer[self.position] = (state, action, reward, next_state, done)
		self.position = (self.position + 1) % self.capacity
	
	def sample(self, batch_size):

		batch = random.sample(self.buffer, batch_size)
		state, action, reward, next_state, done = map(np.stack, zip(*batch))
		return state, action, reward, next_state, done
	
	def __len__(self):
		return len(self.buffer)


def plot(frame_idx, rewards):

	clear_output(True)
	plt.figure(figsize=(20,5))
	plt.subplot(131)
	plt.title('frame %s. reward: %s' % (frame_idx, rewards[-1]))
	plt.plot(rewards)
	plt.show()


class ValueNetwork(nn.Module):

	def __init__(self, state_dim, hidden_dim, init_w=3e-3):
		super(ValueNetwork, self).__init__()
		
		self.linear1 = nn.Linear(state_dim, hidden_dim)
		self.linear2 = nn.Linear(hidden_dim, hidden_dim)
		self.linear3 = nn.Linear(hidden_dim, 1)
		
		self.linear3.weight.data.uniform_(-init_w, init_w)
		self.linear3.bias.data.uniform_(-init_w, init_w)
		
	def forward(self, state):
		x = F.relu(self.linear1(state))
		x = F.relu(self.linear2(x))
		x = self.linear3(x)
		return x
		
		
class SoftQNetwork(nn.Module):

	def __init__(self, num_inputs, num_actions, hidden_size, init_w=3e-3):
		super(SoftQNetwork, self).__init__()
		
		self.linear1 = nn.Linear(num_inputs + num_actions, hidden_size)
		self.linear2 = nn.Linear(hidden_size, hidden_size)
		self.linear3 = nn.Linear(hidden_size, 1)
		
		self.linear3.weight.data.uniform_(-init_w, init_w)
		self.linear3.bias.data.uniform_(-init_w, init_w)
		
	def forward(self, state, action):
		x = torch.cat([state, action], 1)
		x = F.relu(self.linear1(x))
		x = F.relu(self.linear2(x))
		x = self.linear3(x)
		return x
		
		
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


def update(batch_size,gamma=0.99,soft_tau=1e-2):
	
	state, action, reward, next_state, done = replay_buffer.sample(batch_size)

	state      = torch.FloatTensor(state).to(device)
	next_state = torch.FloatTensor(next_state).to(device)
	action     = torch.FloatTensor(action).to(device)
	reward     = torch.FloatTensor(reward).unsqueeze(1).to(device)
	done       = torch.FloatTensor(np.float32(done)).unsqueeze(1).to(device)

	predicted_q_value1 = soft_q_net1(state, action)
	predicted_q_value2 = soft_q_net2(state, action)
	predicted_value    = value_net(state)
	new_action, log_prob, epsilon, mean, log_std = policy_net.evaluate(state)

	
	
# Training Q Function
	target_value = target_value_net(next_state)
	target_q_value = reward + (1 - done) * gamma * target_value
	q_value_loss1 = soft_q_criterion1(predicted_q_value1, target_q_value.detach())
	q_value_loss2 = soft_q_criterion2(predicted_q_value2, target_q_value.detach())


	soft_q_optimizer1.zero_grad()
	q_value_loss1.backward()
	soft_q_optimizer1.step()
	soft_q_optimizer2.zero_grad()
	q_value_loss2.backward()
	soft_q_optimizer2.step()    
# Training Value Function
	predicted_new_q_value = torch.min(soft_q_net1(state, new_action),soft_q_net2(state, new_action))
	# IPython.embed()
	target_value_func = predicted_new_q_value - log_prob
	value_loss = value_criterion(predicted_value, target_value_func.detach())

	
	value_optimizer.zero_grad()
	value_loss.backward()
	value_optimizer.step()
# Training Policy Function
	policy_loss = (log_prob - predicted_new_q_value).mean()

	policy_optimizer.zero_grad()
	policy_loss.backward()
	policy_optimizer.step()
	
	
	for target_param, param in zip(target_value_net.parameters(), value_net.parameters()):
		target_param.data.copy_(
			target_param.data * (1.0 - soft_tau) + param.data * soft_tau
		)


pen_1 = args.pen_1
pen_2 = args.pen_2
fl_name= args.flname
env = Point('name', 20, -1, 20, -1, np.array([20.0,20.0]), np.array([5.5,14.5]), np.array([14.5,5.5]), pen_1, pen_2)#gym.make(args.env_name)
action_dim = 2#env.action_space.shape[0]
state_dim  = 2#env.observation_space.shape[0]
hidden_dim = 256

value_net        = ValueNetwork(state_dim, hidden_dim).to(device)
target_value_net = ValueNetwork(state_dim, hidden_dim).to(device)

soft_q_net1 = SoftQNetwork(state_dim, action_dim, hidden_dim).to(device)
soft_q_net2 = SoftQNetwork(state_dim, action_dim, hidden_dim).to(device)
policy_net = PolicyNetwork(state_dim, action_dim, hidden_dim).to(device)

for target_param, param in zip(target_value_net.parameters(), value_net.parameters()):
	target_param.data.copy_(param.data)
	

value_criterion  = nn.MSELoss()
soft_q_criterion1 = nn.MSELoss()
soft_q_criterion2 = nn.MSELoss()

value_lr  = 3e-4
soft_q_lr = 3e-4
policy_lr = 3e-4

value_optimizer  = optim.Adam(value_net.parameters(), lr=value_lr)
soft_q_optimizer1 = optim.Adam(soft_q_net1.parameters(), lr=soft_q_lr)
soft_q_optimizer2 = optim.Adam(soft_q_net2.parameters(), lr=soft_q_lr)
policy_optimizer = optim.Adam(policy_net.parameters(), lr=policy_lr)


replay_buffer_size = 100000
replay_buffer = ReplayBuffer(replay_buffer_size)



max_frames  = 200000
max_steps   = 500
frame_idx   = 0
rewards     = []
batch_size  = 128
cnt=0
while frame_idx < max_frames:
	episode_reward = 0

	state = np.zeros(2)
	# starting state
	if np.random.rand()<0.5:
		state[0] = 20*np.random.rand()
		state[1] = 1*np.random.rand()
	else:
		state[0] = 1*np.random.rand()
		state[1] = 20*np.random.rand()

	env.set_position(state[0], state[1])
	for step in range(max_steps):
		if frame_idx >20000:
			action = policy_net.get_action(state).detach()
			action=np.clip(action,-1, 1)
			next_state, reward, done, _ = env.step(action.numpy())
		else:
			# explore
			action = np.random.rand(2)
			action=np.clip(action,-1, 1)
			next_state, reward, done, _ = env.step(action)				

		replay_buffer.push(state, action, reward, next_state, done)
		
		state = next_state
		episode_reward += reward
		frame_idx += 1
		
		if len(replay_buffer) > batch_size:
			update(batch_size)
		
		if frame_idx % 1000 == 0:
			print("In Frame {} and episode {} reward is {}".format(frame_idx, step, episode_reward))


		if done:
			print(frame_idx,'Done')
			break	
		
	rewards.append(episode_reward)

torch.save(policy_net.state_dict(), 'model/' + fl_name + str(pen_1) + '_' + str(pen_2) + '.pth')

IPython.embed()

def display_frames_as_gif(frames):
	
	"""
	Displays a list of frames as a gif, with controls
	"""

	patch = plt.imshow(frames[0])
	plt.axis('off')

	def animate(i):
		patch.set_data(frames[i])

	anim = animation.FuncAnimation(plt.gcf(), animate, frames = len(frames), interval=50)
	display(anim)

