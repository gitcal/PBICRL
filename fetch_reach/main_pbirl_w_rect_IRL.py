import mdp_utils
import bayesian_irl_w_rect_IRL
import copy
import IPython
import numpy as np
import pickle, argparse
import itertools
from itertools import permutations
from random import sample
import random

parser = argparse.ArgumentParser()
parser.add_argument('--boltz_beta1', type=float, default=1)
parser.add_argument('--boltz_beta2', type=float, default=1)
parser.add_argument('--boltz_beta3', type=float, default=1)
parser.add_argument('--boltz_beta4', type=float, default=1)
parser.add_argument('--stdev', type=float, default=0.1)
parser.add_argument('--num_steps', type=int, default=6000)
parser.add_argument('--iterations', type=int, default=10)
parser.add_argument('--N_demonstrations', type=int, default=10)
parser.add_argument('--flname', type=str, default='results_fetch_reach_rect_IRL_')
parser.add_argument('--m1', type=float, default=0)
parser.add_argument('--m2', type=float, default=0)
parser.add_argument('--m3', type=float, default=0)
parser.add_argument('--m4', type=float, default=0)
parser.add_argument('--s', type=float, default=1)
parser.add_argument('--norm_flag', type=int, default=0)
args = parser.parse_args()


def get_feat(bad_cnstr, very_bad_cnstr, target, r_constr, r_target, traj):
	
	s_ = np.array(traj)
	targ_feat = np.expand_dims(1/np.linalg.norm(s_-target, axis=1),axis=1)
	IPython.embed()
	obst_1_feat = np.expand_dims(1*(np.linalg.norm(s_-bad_cnstr, axis=1)<r_constr),axis=1)
	obst_2_feat = np.expand_dims(1*(np.linalg.norm(s_-very_bad_cnstr, axis=1)<r_constr),axis=1)

	tt=np.zeros((len(targ_feat),1))
	s_f = traj[-1]
	if np.linalg.norm(s_f-target)<=r_target:
		tt[-1]=1
	feat = np.stack((targ_feat, obst_1_feat, obst_2_feat,tt))
	return feat

def get_feat_rect(cnstr1, cnstr2, cnstr3, target, r_constr, r_target, traj):
	
	s_ = np.array(traj)
	targ_feat = np.expand_dims(1/np.linalg.norm(s_-target, axis=1),axis=1)
	
	state = s_

	obst_1_feat=np.expand_dims(1*((state[:,0]<=cnstr1[0][1]) &  (state[:,0]>=cnstr1[0][0]) & (state[:,1]<=cnstr1[1][1]) &  (state[:,1]>=cnstr1[1][0])\
		& (state[:,2]<=cnstr1[2][1]) &  (state[:,2]>=cnstr1[2][0])), axis=1)

	obst_2_feat=np.expand_dims(1*((state[:,0]<=cnstr2[0][1]) &  (state[:,0]>=cnstr2[0][0]) & (state[:,1]<=cnstr2[1][1]) &  (state[:,1]>=cnstr2[1][0])\
		& (state[:,2]<=cnstr2[2][1]) &  (state[:,2]>=cnstr2[2][0])), axis=1)


	tt=np.ones((len(targ_feat),1))
	s_f = traj[-1]
	if np.linalg.norm(s_f-target)<=r_target:
		tt[-1]=0
	feat = np.stack((targ_feat, obst_1_feat, obst_2_feat, tt))
	return feat



def main():

	boltz_beta1 = args.boltz_beta1
	boltz_beta2 = args.boltz_beta2
	boltz_beta3 = args.boltz_beta3
	boltz_beta4 = args.boltz_beta4

	num_steps = args.num_steps
	iterations = args.iterations
	stdev = args.stdev
	N_demonstrations = args.N_demonstrations
	fl_name = args.flname
	m1 = args.m1
	m2 = args.m2
	m3 = args.m3
	m4 = args.m4

	s = args.s
	norm_flag = args.norm_flag
	res_dict = {}

	z_thresh = 0.7
	cnstr1 = [(0.7,1.1),(0.7,0.85),(0.7,1.4)]
	cnstr2 = [(0.7,1.1),(1.0,1.1),(0.7,1.4)]
	cnstr3 = [(0.7,0.9),(0.75,1.0),(0.7,1.4)]
	target = np.array([0.95,0.9,z_thresh+0.02])

	bad_cnstr = cnstr1
	very_bad_cnstr = cnstr2
	r_constr = 0.15
	r_target = 0.05


	x_min = 0.5
	x_max = 1.4
	y_min = 0.5
	y_max = 1.4
	z_min = z_thresh
	z_max = 0.9

	W_fix = np.array([0.1,0,0,-5])

	filename = 'data/' + fl_name + '_' + str(num_steps) + '_' + str(N_demonstrations)+\
	'_'+str(boltz_beta1)+'_'+str(boltz_beta2)+'_'+str(boltz_beta3)+'_'+str(s)+'_'+str(m1)+'_'+str(m2)+'_'+str(m3)+'_'+str(norm_flag)       

	with open('data/fetch_reach_trajectories.pickle', 'rb') as pickle_file:
		data = pickle.load(pickle_file)

	trajectory_demos_states = data['trajectories']
	rews = data['rewards']
	# shift to features

	traj_feat = []
	res_map_cnstr = []
	res_map_rew = []
	for k in range(len(trajectory_demos_states)):# iterate over 3 categories of pairwise preferences
		feats_traj = get_feat_rect(cnstr1, cnstr2, cnstr3, target, r_constr, r_target, trajectory_demos_states[k])
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
		print('Iteration {} for beta1 {} beta2 {} beta3 {} s {} m1 {} m2 {} m3 {} demos {} norm {}'.format(iters, boltz_beta1, \
			boltz_beta2, boltz_beta3, s, m1, m2, m3, N_demonstrations, norm_flag))
		# randomly shuffle the preferences
		random.shuffle(good_bad_pairs)
		random.shuffle(bad_vbad_pairs)
		random.shuffle(good_vbad_pairs)
		random.shuffle(vbad_vvbad_pairs)
		
		
		pref_pairs = [good_bad_pairs, bad_vbad_pairs, good_vbad_pairs]

		d12 = []
		d23 = []
		d13 = []
		for ii in range(1):
			birl = bayesian_irl_w_rect_IRL.BIRL(traj_feat, trajectory_demos_states, pref_pairs, boltz_beta1, boltz_beta2, boltz_beta3,boltz_beta4, N_demonstrations, s, m1, m2, m3, m4, norm_flag)
			birl.run_mcmc_bern_constraint(num_steps, W_fix, stdev)
			acc_rate = birl.accept_rate
			print('acceptance rate {}'.format(acc_rate)) 
			chain_weights = birl.get_mean_solution(burn_frac=0.2, skip_rate=1)
			map_weights = birl.get_map_solution()
			res_dict[iters,'map_cnstr'] = map_weights
			res_dict[iters,'mean_weights'] = chain_weights[-500:,:]
			
	with open(filename + '.pickle', 'wb') as handle:
		pickle.dump(res_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
if __name__=="__main__":
	main()
