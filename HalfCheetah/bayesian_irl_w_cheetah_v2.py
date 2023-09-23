import numpy as np
from mdp_utils import logsumexp
import copy
from random import choice
import IPython
from scipy.stats import bernoulli


class BIRL:

    def __init__(self, demos, trajectory_demos_states, preferences, beta_1, beta_2, beta_3, beta_4, num_samples, s, m1, m2, m3, m4, norm_flag):


        """
        Class for running and storing output of mcmc for Bayesian IRL
        env: the mdp (we ignore the reward)
        demos: list of (s,a) tuples 
        beta: the assumed boltzman rationality of the demonstrator
        """
        self.demonstrations = demos
        self.preferences = preferences
        self.beta1 = beta_1
        self.beta2 = beta_2
        self.beta3 = beta_3
        self.beta4 = beta_4
        self.norm_flag = norm_flag
        self.num_samples = num_samples
        self.s = s
        self.m1 = m1
        self.m2 = m2
        self.m3 = m3
        self.m4 = m4
        self.traj_state_demos = trajectory_demos_states

    def logsumexp(x):

        max_x = np.max(x)
        sum_exp = 0.0
        for xi in x:
            sum_exp += np.exp(xi - max_x)
        return max(x) + np.log(sum_exp)


    def calc_ll_pref(self, rew_tuple,  trajectories, preferences):

        #perform hypothetical given current reward hypothesis
        log_prior = 0.0  #assume unimformative prior
        log_sum = log_prior
        W_fix, cur_W_ind, cur_rew, cur_x_loc = rew_tuple
        for k in range(len(preferences)):# iterate over 3 categories of pairwise preferences
            for itr in range(self.num_samples):
                ind1 = preferences[k][itr][0]
                ind2 = preferences[k][itr][1]
                feat_traj1 = np.squeeze(trajectories[ind1]).T
                feat_traj2 = np.squeeze(trajectories[ind2]).T
                rew1 = np.dot(feat_traj1[:,:2], W_fix)
                rew2 = np.dot(feat_traj2[:,:2], W_fix)
                rew1_temp = 1*(feat_traj1[:,2]>cur_x_loc) * cur_rew * cur_W_ind
                rew2_temp = 1*(feat_traj2[:,2]>cur_x_loc) * cur_rew * cur_W_ind
                rew1 = rew1 + rew1_temp
                rew2 = rew2 + rew2_temp

                if self.norm_flag:
                    if k==0:
                        log_sum += self.s*(self.beta1 * np.mean(rew1)-self.m1) - logsumexp([self.s*(self.beta1 * np.mean(rew1)-self.m1), self.s*self.beta1 * np.mean(rew2)]) 
                    if k==1:
                        log_sum += self.s*(self.beta2 * np.mean(rew1)-self.m2) - logsumexp([self.s*(self.beta2 * np.mean(rew1)-self.m2), self.s*self.beta2 * np.mean(rew2)]) 
                    if k==2:
                        log_sum += self.s*(self.beta3 * np.mean(rew1)-self.m3) - logsumexp([self.s*(self.beta3 * np.mean(rew1)-self.m3), self.s*self.beta3 * np.mean(rew2)])
                else:
                    if k==0:
                        log_sum += self.s*(self.beta1 * np.sum(rew1)-self.m1) - logsumexp([self.s*(self.beta1 * np.sum(rew1)-self.m1), self.s*self.beta1 * np.sum(rew2)]) 
                    if k==1:
                        log_sum += self.s*(self.beta2 * np.sum(rew1)-self.m2) - logsumexp([self.s*(self.beta2 * np.sum(rew1)-self.m2), self.s*self.beta2 * np.sum(rew2)]) 
                    if k==2:
                        log_sum += self.s*(self.beta3 * np.sum(rew1)-self.m3) - logsumexp([self.s*(self.beta3 * np.sum(rew1)-self.m3), self.s*self.beta3 * np.sum(rew2)])
        
        return log_sum
    

    
    

    def generate_proposal_bern_constr_alternating(self, W_ind_old, rew_old, x_loc_old, ind, stdev1=1, stdev2=1):

        rew_new = copy.deepcopy(rew_old)
        W_ind_new = copy.deepcopy(W_ind_old)
        x_loc_new = copy.deepcopy(x_loc_old)
        if ind % 3 == 0:  
            rew_new = rew_old + stdev1 * np.random.randn() 
        elif ind % 3 == 1:
            W_ind_new = 1 if W_ind_old == 0 else 0
        else:
            x_loc_new = x_loc_old + stdev2 * np.random.randn() 
            
        return W_ind_new, rew_new, x_loc_new



    def initial_solution_bern_cnstr(self):

        # initialize problem solution for MCMC to all zeros, maybe not best initialization but it works in most cases
        pen_rew = np.random.uniform(-100, -10,size=1)#, self.env.num_states)
        x_loc_new = 5*np.random.randn(1)
        W_ind_new = np.random.randint(2,size=1)
        return W_ind_new, pen_rew, x_loc_new

   

  

    def run_mcmc_bern_constraint(self, samples, W_fix, stdev1, stdev2):

        '''
            run metropolis hastings MCMC with Gaussian symmetric proposal and uniform prior
            samples: how many reward functions to sample from posterior
            stepsize: standard deviation for proposal distribution
            normalize: if true then it will normalize the rewards (reward weights) to be unit l2 norm, otherwise the rewards will be unbounded
        '''
        
        num_samples = samples  # number of MCMC samples
        accept_cnt = 0  #keep track of how often MCMC accepts, ideally around 40% of the steps accept
        #if accept count is too high, increase stdev, if too low reduce
        self.chain_W_ind = np.zeros((num_samples, 2)) #store rewards found via BIRL here, preallocate for speed
        self.chain_rew = np.zeros((num_samples, 2)) 
        self.chain_x_loc = np.zeros((num_samples, 2)) 
        cur_W_ind, cur_rew, cur_x_loc = self.initial_solution_bern_cnstr()
        cur_sol = (W_fix, cur_W_ind, cur_rew, cur_x_loc)
       
        cur_ll = self.calc_ll_pref(cur_sol, self.demonstrations, self.preferences)  # log likelihood
        #keep track of MAP loglikelihood and solution
        map_ll = cur_ll  
        map_W_ind = cur_W_ind
        map_rew = cur_rew
        map_x_loc = cur_x_loc
        for i in range(num_samples):
        
            prop_W_ind, prop_rew, prop_x_loc = self.generate_proposal_bern_constr_alternating(cur_W_ind, cur_rew, cur_x_loc, i, stdev1=1, stdev2=0.5)
            prop_sol = (W_fix, prop_W_ind, prop_rew, prop_x_loc)
            prop_ll = self.calc_ll_pref(prop_sol, self.demonstrations, self.preferences)
    
            if prop_ll > cur_ll:
            
                self.chain_W_ind[i,] = prop_W_ind
                self.chain_rew[i,:] = prop_rew
                self.chain_x_loc[i,:] = prop_x_loc
                accept_cnt += 1
                cur_W_ind = prop_W_ind
                cur_rew = prop_rew
                cur_x_loc = prop_x_loc
                cur_ll = prop_ll
                if prop_ll > map_ll:  # maxiumum aposterioi
                    map_ll = prop_ll
                    map_W_ind = cur_W_ind
                    map_rew = cur_rew
                    map_x_loc = cur_x_loc
            else:
                # accept with prob exp(prop_ll - cur_ll)
                if np.random.rand() < np.exp(prop_ll - cur_ll):
                    self.chain_W_ind[i,:] = prop_W_ind
                    self.chain_rew[i,:] = prop_rew
                    self.chain_x_loc[i,:] = prop_x_loc
                    accept_cnt += 1
                    cur_W_ind = prop_W_ind
                    cur_rew = prop_rew
                    cur_x_loc = prop_x_loc
                    cur_ll = prop_ll
                else:
                    # reject
                    self.chain_W_ind[i,:] = cur_W_ind
                    self.chain_rew[i,:] = cur_rew
                    self.chain_x_loc[i,:] = cur_x_loc
            
        self.accept_rate = accept_cnt / num_samples
        self.map_rew = map_rew
        self.map_W_ind = map_W_ind
        self.map_x_loc = map_x_loc
         

    def get_map_solution(self):

        return self.map_W_ind, self.map_rew, self.map_x_loc


    def get_mean_solution(self, burn_frac=0.1, skip_rate=1):

        ''' get mean solution after removeing burn_frac fraction of the initial samples and only return every skip_rate
            sample. Skiping reduces the size of the posterior and can reduce autocorrelation. Burning the first X% samples is
            often good since the starting solution for mcmc may not be good and it can take a while to reach a good mixing point
        '''    
        Chain_W_ind = self.chain_W_ind
        Chain_rew = self.chain_rew
        Chain_x_loc = self.chain_x_loc
        
        return Chain_W_ind, Chain_rew, Chain_x_loc
