import bayesian_irl_w
import copy
import IPython
import numpy as np
import pickle, argparse
from custom_env_w import Point
import itertools
from itertools import permutations
from random import sample
import random

parser = argparse.ArgumentParser()
parser.add_argument('--boltz_beta1', type=float, default=1)
parser.add_argument('--boltz_beta2', type=float, default=1)
parser.add_argument('--boltz_beta3', type=float, default=1)
parser.add_argument('--stdev', type=float, default=0.1)
parser.add_argument('--num_steps', type=int, default=60000)
parser.add_argument('--iterations', type=int, default=10)
parser.add_argument('--N_demonstrations', type=int, default=100)
parser.add_argument('--fl_name', type=str, default='results_marg')
parser.add_argument('--m1', type=float, default=0)
parser.add_argument('--m2', type=float, default=0)
parser.add_argument('--m3', type=float, default=0)
parser.add_argument('--s', type=float, default=1)
parser.add_argument('--norm_flag', type=int, default=0)
parser.add_argument('--data_fl_name', type=str, default='trajectories_m10_m100')
args = parser.parse_args()


def get_feat(env, traj):
    
    s_ = np.array(traj)
    targ_feat = np.expand_dims(1/np.linalg.norm(s_-env.target, axis=1),axis=1)
    obst_1_feat = np.expand_dims(1*(np.linalg.norm(s_-env.bad_cnstr, axis=1)<env.r_constr),axis=1)
    obst_2_feat = np.expand_dims(1*(np.linalg.norm(s_-env.very_bad_cnstr, axis=1)<env.r_constr),axis=1)
    tt=np.zeros((len(targ_feat),1))
    s_f = traj[-1]
    if np.linalg.norm(s_f-env.target)<=env.r_target:
        tt[-1]=1
    feat = np.stack((targ_feat, obst_1_feat, obst_2_feat,tt))
    return feat


def main():

    boltz_beta1 = args.boltz_beta1
    boltz_beta2 = args.boltz_beta2
    boltz_beta3 = args.boltz_beta3
    num_steps = args.num_steps
    iterations = args.iterations
    stdev = args.stdev
    N_demonstrations = args.N_demonstrations
    fl_name = args.fl_name
    data_fl_name = args.data_fl_name
    m1 = args.m1
    m2 = args.m2
    m3 = args.m3
    norm_flag = args.norm_flag
    s = args.s
    res_dict = {}

    env = Point('name', 20, -1, 20, -1, np.array([20.0,20.0]), np.array([5.5,14.5]), np.array([14.5,5.5]), 0, 0)#gym.make(args.env_name)

    terminal_states = [0]
    constraints = [1]

    filename = 'data/' + fl_name + '_' + str(num_steps) + '_' + str(N_demonstrations)+\
    '_'+str(boltz_beta1)+'_'+str(boltz_beta2)+'_'+str(boltz_beta3)+'_'+str(s)+'_'+str(m1)+'_'+str(m2)+'_'+str(m3)+'_'+str(norm_flag)              
    with open('data/' + data_fl_name + '.pickle', 'rb') as pickle_file:
        data = pickle.load(pickle_file)

    trajectory_demos_states = data['trajectories']
    rews = data['rewards']
    
    traj_feat = []
    res_map_cnstr = []
    res_map_rew = []
    for k in range(len(trajectory_demos_states)):# iterate over 3 categories of pairwise preferences
        feats_traj = get_feat(env, trajectory_demos_states[k])
        traj_feat.append(feats_traj)
    preferences = data['traj_index']

    good_ind = np.where(preferences == 1)[0]
    bad_ind = np.where(preferences == 2)[0]
    vbad_ind = np.where(preferences == 3)[0]
    
    good_bad_pairs = []
    bad_vbad_pairs = []
    good_vbad_pairs = []
    # good_vvbad_pairs = []
    # bad_vvbad_pairs = []

    for i in good_ind:
        for j in bad_ind:
            good_bad_pairs.append((i,j))

    for i in bad_ind:
        for j in vbad_ind:
            bad_vbad_pairs.append((i,j))

    for i in good_ind:
        for j in vbad_ind:
            good_vbad_pairs.append((i,j))


    for iters in range(iterations):
        print('Iteration {} for beta1 {} beta2 {} beta3 {} s {} m1 {} m2 {} m3 {}'.format(iters, boltz_beta1, boltz_beta2, boltz_beta3, s, m1, m2, m3))
        # randomly shuffle the preferences
        random.shuffle(good_bad_pairs)
        random.shuffle(bad_vbad_pairs)
        random.shuffle(good_vbad_pairs)
        
        pref_pairs = [good_bad_pairs, bad_vbad_pairs, good_vbad_pairs]

        d12 = []
        d23 = []
        d13 = []
        for ii in range(1):
            birl = bayesian_irl_w.BIRL(env, traj_feat, pref_pairs, boltz_beta1, boltz_beta2, boltz_beta3, N_demonstrations, s, m1, m2, m3, norm_flag)
            birl.run_mcmc_bern_constraint(num_steps, env.w_nom, stdev)
            acc_rate = birl.accept_rate
            print('acceptance rate {}'.format(acc_rate)) 
            chain_constraints, chain_weights = birl.get_mean_solution(burn_frac=0.2, skip_rate=1)
            map_constraints, map_cnstr = birl.get_map_solution()
            res_dict[iters,'weights'] = map_constraints
            res_dict[iters,'map_cnstr'] = map_cnstr
            res_dict[iters,'mean_weights'] = chain_constraints[-500:,:]
            res_dict[iters,'mean_map_cnstr'] = chain_weights[-500:,:]
            

    with open(filename + '.pickle', 'wb') as handle:
        pickle.dump(res_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
if __name__=="__main__":
    main()