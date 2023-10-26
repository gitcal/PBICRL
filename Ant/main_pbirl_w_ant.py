import bayesian_irl_w_ant
import copy
import IPython
import numpy as np
import pickle, argparse
import itertools
from itertools import permutations
from random import sample
import random
import time

parser = argparse.ArgumentParser()
parser.add_argument('--boltz_beta1', type=float, default=1)
parser.add_argument('--boltz_beta2', type=float, default=1)
parser.add_argument('--boltz_beta3', type=float, default=1)
parser.add_argument('--boltz_beta4', type=float, default=1)
parser.add_argument('--stdev1', type=float, default=1)
parser.add_argument('--stdev2', type=float, default=0.5)
parser.add_argument('--num_steps', type=int, default=20000)
parser.add_argument('--iterations', type=int, default=1)
parser.add_argument('--N_demonstrations', type=int, default=30)
parser.add_argument('--flname', type=str, default='Ant_trajectories_cnstr_100_cnstr1_10')
parser.add_argument('--m1', type=float, default=0)
parser.add_argument('--m2', type=float, default=0)
parser.add_argument('--m3', type=float, default=0)
parser.add_argument('--m4', type=float, default=0)
parser.add_argument('--s', type=float, default=1)
parser.add_argument('--norm_flag', type=int, default=1)
args = parser.parse_args()



def get_feat_rect(traj):	

	dt = 0.05
	st_list = []
	single_state = []
	act_list = []
	for i in range(len(traj)-1):
		tr = traj[i]
		tr_p = traj[i+1]
		st_tr, act_tr, st_rew = tr
		st_tr_p, act_tr_p, st_rew_p = tr_p
		st_list.append(st_rew_p-st_rew)
		single_state.append(st_tr_p)
		act_list.append(act_tr)
	s_ = np.expand_dims(np.array(st_list),axis=1)
	a_ = np.array(act_list)

	single_state = np.array(single_state)
	state = s_
	feat_1 = s_/dt
	feat_2 = np.expand_dims(np.linalg.norm(a_,axis=1)**2, axis=1)
	feat_3 = np.expand_dims(single_state[:,0], axis=1) 
	feat = np.stack((feat_1, feat_2, feat_3))
	return feat



def main():

	boltz_beta1 = args.boltz_beta1
	boltz_beta2 = args.boltz_beta2
	boltz_beta3 = args.boltz_beta3
	boltz_beta4 = args.boltz_beta4

	num_steps = args.num_steps
	iterations = args.iterations
	stdev1 = args.stdev1
	stdev2 = args.stdev2
	N_demonstrations = args.N_demonstrations
	fl_name = args.flname
	m1 = args.m1
	m2 = args.m2
	m3 = args.m3
	m4 = args.m4

	s = args.s
	norm_flag = args.norm_flag
	res_dict = {}

	W_fix = np.array([1,-0.5])
	filename = 'data/' + fl_name + '_' + str(num_steps) + '_' + str(N_demonstrations)+\
	'_'+str(boltz_beta1)+'_'+str(boltz_beta2)+'_'+str(boltz_beta3)+'_'+str(s)+'_'+str(m1)+'_'+str(m2)+'_'+str(m3)+'_'+str(norm_flag)       

	with open('data/' + fl_name + '.pickle', 'rb') as pickle_file:
		data = pickle.load(pickle_file)

	trajectory_demos_states = data['trajectories']
	rews = data['rewards']

	traj_feat = []
	res_map_cnstr = []
	res_map_rew = []
	for k in range(len(trajectory_demos_states)):# iterate over 3 categories of pairwise preferences
		feats_traj = get_feat_rect(trajectory_demos_states[k])
		traj_feat.append(feats_traj)
	preferences = data['traj_index']

	good_ind = np.where(preferences == 1)[0]
	bad_ind = np.where(preferences == 2)[0]
	vbad_ind = np.where(preferences == 3)[0]
	vvbad_ind = np.where(preferences == 4)[0]
	good_bad_pairs = []
	bad_vbad_pairs = []
	good_vbad_pairs = []
	vbad_vvbad_pairs = []

	for i in good_ind:
		for j in bad_ind:
			good_bad_pairs.append((i,j))

	for i in bad_ind:
		for j in vbad_ind:
			bad_vbad_pairs.append((i,j))

	for i in good_ind:
		for j in vbad_ind:
			good_vbad_pairs.append((i,j))

	for i in vbad_ind:
		for j in vvbad_ind:
			vbad_vvbad_pairs.append((i,j))
	

	for iters in range(iterations):
		time_start = time.time()
		print('Iteration {} for beta1 {} beta2 {} beta3 {} s {} m1 {} m2 {} m3 {} demos {} norm {}'.format(iters, boltz_beta1, \
			boltz_beta2, boltz_beta3, s, m1, m2, m3, N_demonstrations, norm_flag))

		random.shuffle(good_bad_pairs)
		random.shuffle(bad_vbad_pairs)
		random.shuffle(good_vbad_pairs)
		random.shuffle(vbad_vvbad_pairs)
		
		pref_pairs = [good_bad_pairs]#, bad_vbad_pairs, good_vbad_pairs]#,vbad_vvbad_pairs]

		d12 = []
		d23 = []
		d13 = []
		for ii in range(1):
			birl = bayesian_irl_w_ant.BIRL(traj_feat, trajectory_demos_states, pref_pairs, boltz_beta1, boltz_beta2, boltz_beta3,boltz_beta4, N_demonstrations, s, m1, m2, m3, m4, norm_flag)
			birl.run_mcmc_bern_constraint(num_steps, W_fix, stdev1, stdev2)
			acc_rate = birl.accept_rate
			print('acceptance rate {}'.format(acc_rate)) #self.map_W_ind, self.map_rew, self.map_x_loc
			chain_W_ind, chain_rew, chain_x_loc = birl.get_mean_solution(burn_frac=0.2, skip_rate=1)
			map_W_ind, map_rew, map_x_loc = birl.get_map_solution()
			res_dict[iters,'indicators'] = map_W_ind
			res_dict[iters,'map_rew'] = map_rew
			res_dict[iters,'map_x_loc'] =map_x_loc 
			res_dict[iters,'chain_W_ind'] = chain_W_ind[-500:,:]
			res_dict[iters,'chain_rew'] = chain_rew[-500:,:]
			res_dict[iters,'chain_x_loc'] = chain_x_loc[-500:,:]
			res_dict[iters,'time'] = time.time()-time_start

		print('Time of iteration {} is {}'.format(iters, time.time()-time_start))

	with open(filename + '.pickle', 'wb') as handle:
		pickle.dump(res_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
if __name__=="__main__":
	main()
