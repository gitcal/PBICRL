import numpy as np 
# import cv2 
import matplotlib.pyplot as plt
# import PIL.Image as Image
# import gym
import random
import IPython
# from gym import Env, spaces
import time

class Point(object):

	metadata = {
		"render_modes": ["human", "rgb_array"],
		"render_fps": 30,
	}

	def __init__(self, name, x_max, x_min, y_max, y_min, target, bad_cnstr, very_bad_cnstr, pen_1, pen_2, r_constr=3, r_target =3.0):

		# reward is inner priduct of w and s
		# w_1 dist from target
		# w_2 dist from obstacle 1
		# w_3 distance from obstacle 2
		# w_4 living penalty
		self.x = 0
		self.y = 0
		self.x_min = x_min
		self.x_max = x_max
		self.y_min = y_min
		self.y_max = y_max
		self.target = target
		self.bad_cnstr = bad_cnstr
		self.very_bad_cnstr = very_bad_cnstr
		self.name = name
		self.observation_space = np.array([2])
		self.action_space = np.array([2])
		self.icon_w = 0
		self.icon_h = 0
		self.w = np.array([1,pen_1,pen_2,-1])#,-1]) # was -20 and -40
		self.num_features = 4
		self.w_nom = np.array([self.w[0],0,0,self.w[3]])#,-1])
		self.r_constr = r_constr
		self.r_target = r_target


	def reset(self):

		self.x = 1*np.random.rand()#0.0
		self.y = 1*np.random.rand()#0.0
		self.alive = 1
		return np.array([self.x, self.y])

	
	def set_position(self, x, y):

		self.x = self.clamp(x, self.x_min, self.x_max - self.icon_w)
		self.y = self.clamp(y, self.y_min, self.y_max - self.icon_h)
	
	def get_position(self):

		return (self.x, self.y, self.alive)
	
	def step(self, a):

		self.x += a[0]
		self.y += a[1]
		self.x = self.clamp(self.x, self.x_min, self.x_max - self.icon_w)
		self.y = self.clamp(self.y, self.y_min, self.y_max - self.icon_h)
		info = []

		s_ = np.array([self.x, self.y])
		# features
		targ_feat = 1/np.linalg.norm(s_-self.target)

		obst_1_feat = int(np.linalg.norm(s_-self.bad_cnstr) < self.r_constr)
		obst_2_feat = int(np.linalg.norm(s_-self.very_bad_cnstr) < self.r_constr)
		

	
		
		done = False
		if np.linalg.norm(s_-self.target)<=self.r_target:
			done=True
		living_feat = 1-done

		feat = np.array([targ_feat, obst_1_feat, obst_2_feat, living_feat])
		
		r = np.dot(self.w, feat)
		return s_, r, done, info

	def get_reward(self, s):

		self.x = s[0]
		self.y = s[1]
		self.x = self.clamp(self.x, self.x_min, self.x_max - self.icon_w)
		self.y = self.clamp(self.y, self.y_min, self.y_max - self.icon_h)
		info = []

		s_ = np.array([self.x, self.y])
		# features
		targ_feat = 1/np.linalg.norm(s_-self.target)

		obst_1_feat = int(np.linalg.norm(s_-self.bad_cnstr) < self.r_constr)
		obst_2_feat = int(np.linalg.norm(s_-self.very_bad_cnstr) < self.r_constr)
		
		done = False
		if np.linalg.norm(s_-self.target)<=self.r_target:
			done=True
		living_feat = 1-done

		feat = np.array([targ_feat, obst_1_feat, obst_2_feat, living_feat])
		
		r = np.dot(self.w, feat)
		return r

	def clamp(self, n, minn, maxn):
		return max(min(maxn, n), minn)


class Point_Inf(object):

	metadata = {
		"render_modes": ["human", "rgb_array"],
		"render_fps": 30,
	}

	def __init__(self, name, x_max, x_min, y_max, y_min, target, bad_cnstr, very_bad_cnstr, w,  r_constr=3, r_target =3.0):

		# reward is inner priduct of w and s
		
		self.x = 0
		self.y = 0
		self.x_min = x_min
		self.x_max = x_max
		self.y_min = y_min
		self.y_max = y_max
		self.target = target
		self.bad_cnstr = bad_cnstr
		self.very_bad_cnstr = very_bad_cnstr
		self.name = name
		self.observation_space = np.array([2])
		self.action_space = np.array([2])
		self.icon_w = 0
		self.icon_h = 0
		self.w = w
		self.num_features = 4
		self.w_nom = np.array([self.w[0],0,0,self.w[3]])
		self.r_constr = r_constr
		self.r_target = r_target


	def reset(self):

		self.x = 1*np.random.rand()
		self.y = 1*np.random.rand()
		self.alive = 1
		return np.array([self.x, self.y])

	
	def set_position(self, x, y):

		self.x = self.clamp(x, self.x_min, self.x_max - self.icon_w)
		self.y = self.clamp(y, self.y_min, self.y_max - self.icon_h)
	
	def get_position(self):

		return (self.x, self.y, self.alive)
	
	def step(self, a):

		
		self.x += a[0]
		self.y += a[1]#
		self.x = self.clamp(self.x, self.x_min, self.x_max - self.icon_w)
		self.y = self.clamp(self.y, self.y_min, self.y_max - self.icon_h)
		info = []

		s_ = np.array([self.x, self.y])
		# features
		targ_feat = 1/np.linalg.norm(s_-self.target)	
		obst_1_feat = int(np.linalg.norm(s_-self.bad_cnstr) < self.r_constr)
		obst_2_feat = int(np.linalg.norm(s_-self.very_bad_cnstr) < self.r_constr)
				
		done = False
		if np.linalg.norm(s_-self.target)<=self.r_target:
			done=True
		living_feat = 1-done

		feat = np.array([targ_feat, obst_1_feat, obst_2_feat, living_feat])
		
		r = np.dot(self.w, feat)
		return s_, r, done, info

	def clamp(self, n, minn, maxn):
		return max(min(maxn, n), minn)