from mdp_utils import  logsumexp
import numpy as np
import copy
from random import choice
import IPython
from scipy.stats import bernoulli
from plot_temp_file import plot_temp



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
        self.num_features = 4
        self.num_mcmc_dims = 4
        self.traj_state_demos = trajectory_demos_states

    def logsumexp(x):

        max_x = np.max(x)
        sum_exp = 0.0
        for xi in x:
            sum_exp += np.exp(xi - max_x)
        return max(x) + np.log(sum_exp)

    def calc_ll_pref(self, w_sample,  trajectories, preferences):

        #perform hypothetical given current reward hypothesis
        log_prior = 0.0  #assume unimformative prior
        log_sum = log_prior
        for k in range(len(preferences)):
            for itr in range(self.num_samples):
               
                ind1 = preferences[k][itr][0]
                ind2 = preferences[k][itr][1]
                feat_traj1 = np.squeeze(trajectories[ind1]).T
                feat_traj2 = np.squeeze(trajectories[ind2]).T
                rew1 = np.dot(feat_traj1, w_sample)
                rew2 = np.dot(feat_traj2, w_sample)
                mean_traj1 = np.mean(feat_traj1,0)
                mean_traj2 = np.mean(feat_traj2,0)
                if self.norm_flag:
                    if k==0:
                        log_sum += self.s*(self.beta1 * np.mean(rew1)-self.m1) - logsumexp([self.s*(self.beta1 * np.mean(rew1)-self.m1), self.s*self.beta1 * np.mean(rew2)]) 
                    if k==1:
                        log_sum += self.s*(self.beta2 * np.mean(rew1)-self.m2) - logsumexp([self.s*(self.beta2 * np.mean(rew1)-self.m2), self.s*self.beta2 * np.mean(rew2)]) 
                    if k==2:
                        log_sum += self.s*(self.beta3 * np.mean(rew1)-self.m3) - logsumexp([self.s*(self.beta3 * np.mean(rew1)-self.m3), self.s*self.beta3 * np.mean(rew2)])
                    if k==3:
                        log_sum += self.s*(self.beta4 * np.mean(rew1)-self.m4) - logsumexp([self.s*(self.beta4 * np.mean(rew1)-self.m4), self.s*self.beta4 * np.mean(rew2)])
        
                else:
                    if k==0:
                        log_sum += self.s*(self.beta1 * np.sum(rew1)) - logsumexp([self.s*(self.beta1 * np.sum(rew1)), self.s*self.beta1 * np.sum(rew2)]) 
                    if k==1:
                        log_sum += self.s*(self.beta2 * np.sum(rew1)) - logsumexp([self.s*(self.beta2 * np.sum(rew1)), self.s*self.beta2 * np.sum(rew2)]) 
                    if k==2:
                        log_sum += self.s*(self.beta3 * np.sum(rew1)) - logsumexp([self.s*(self.beta3 * np.sum(rew1)), self.s*self.beta3 * np.sum(rew2)])
                    if k==3:
                        log_sum += self.s*(self.beta4 * np.mean(rew1)) - logsumexp([self.s*(self.beta4 * np.mean(rew1)), self.s*self.beta4 * np.mean(rew2)])
        
        return log_sum
    

    def calc_ll_pref_marg(self, w_sample,  trajectories, preferences):

        log_prior = 0.0  #assume unimformative prior
        log_sum = log_prior
        
        for k in range(len(preferences)):# iterate over 3 categories of pairwise preferences
            for itr in range(self.num_samples):
              
                ind1 = preferences[k][itr][0]
                ind2 = preferences[k][itr][1]
                feat_traj1 = np.squeeze(trajectories[ind1]).T
                feat_traj2 = np.squeeze(trajectories[ind2]).T
                rew1 = np.dot(feat_traj1, w_sample)
                rew2 = np.dot(feat_traj2, w_sample)
                mean_traj1 = np.mean(feat_traj1,0)
                mean_traj2 = np.mean(feat_traj2,0)
                nn1 = np.linalg.norm(mean_traj1)*np.linalg.norm(w_sample)
                nn2 = np.linalg.norm(mean_traj2)*np.linalg.norm(w_sample)

                if self.norm_flag:
                    if k==0:
                        log_sum += self.s*(self.beta1 * np.mean(rew1)/nn1-self.m1) - \
                        logsumexp([self.s*(self.beta1 * np.mean(rew1)/nn1-self.m1), self.s*self.beta1 * np.mean(rew2)/nn2]) 
                    if k==1:
                        log_sum += self.s*(self.beta2 * np.mean(rew1)/nn1-self.m2) - \
                        logsumexp([self.s*(self.beta2 * np.mean(rew1)/nn1-self.m2), self.s*self.beta2 * np.mean(rew2)/nn2]) 
                    if k==2:
                        log_sum += self.s*(self.beta3 * np.mean(rew1)/nn1-self.m3) - \
                        logsumexp([self.s*(self.beta3 * np.mean(rew1)/nn1-self.m3), self.s*self.beta3 * np.mean(rew2)/nn2])
                    if k==3:
                        log_sum += self.s*(self.beta4 * np.mean(rew1)/nn1-self.m4) - \
                        logsumexp([self.s*(self.beta4 * np.mean(rew1)/nn1-self.m4), self.s*self.beta4 * np.mean(rew2)/nn2])
        
                elif norm_flag==0:
                    if k==0:
                        log_sum += self.s*(self.beta1 * np.sum(rew1)/nn1-self.m1) - \
                        logsumexp([self.s*(self.beta1 * np.sum(rew1)/nn1-self.m1), self.s*self.beta1 * np.sum(rew2)/nn2]) 
                    if k==1:
                        log_sum += self.s*(self.beta2 * np.sum(rew1)/nn1-self.m2) - \
                        logsumexp([self.s*(self.beta2 * np.sum(rew1)/nn1-self.m2), self.s*self.beta2 * np.sum(rew2)/nn2]) 
                    if k==2:
                        log_sum += self.s*(self.beta3 * np.sum(rew1)/nn1-self.m3) - \
                        logsumexp([self.s*(self.beta3 * np.sum(rew1)/nn1-self.m3), self.s*self.beta3 * np.sum(rew2)/nn2])
                    if k==3:
                        log_sum += self.s*(self.beta4 * np.sum(rew1)/nn1-self.m4) - \
                        logsumexp([self.s*(self.beta4 * np.sum(rew1)/nn1-self.m4), self.s*self.beta4 * np.sum(rew2)/nn2])
                
                else:
                    if k==0:
                        log_sum += nn1*(self.beta1 * np.sum(rew1)/nn1-self.m1) - \
                        logsumexp([nn1*(self.beta1 * np.sum(rew1)/nn1-self.m1), nn2*self.beta1 * np.sum(rew2)/nn2]) 
                    if k==1:
                        log_sum += nn1*(self.beta2 * np.sum(rew1)/nn1-self.m2) - \
                        logsumexp([nn1*(self.beta2 * np.sum(rew1)/nn1-self.m2), nn2*self.beta2 * np.sum(rew2)/nn2]) 
                    if k==2:
                        log_sum += nn1*(self.beta3 * np.sum(rew1)/nn1-self.m3) - \
                        logsumexp([nn1*(self.beta3 * np.sum(rew1)/nn1-self.m3), nn2*self.beta3 * np.sum(rew2)/nn2])
                    if k==3:
                        log_sum += nn1*(self.beta4 * np.sum(rew1)/nn1-self.m4) - \
                        logsumexp([nn1*(self.beta4 * np.sum(rew1)/nn1-self.m4), nn2*self.beta4 * np.sum(rew2)/nn2])
         
        return log_sum
    

    def generate_proposal_bern_constr_alternating(self, W_old, rew_old, ind, stdev=1):

        rew_new = copy.deepcopy(rew_old)
        W_new = copy.deepcopy(W_old)
        if ind % 4 == 0:  
            index = np.random.randint(len(W_old))
            rew_new[index] = rew_new[index] + stdev * np.random.randn() 
        else:
            index = np.random.randint(len(W_old))
            W_new[index] = 1 if W_old[index] == 0 else 0
            
        return W_new, rew_new

    def generate_proposal_bern_constr_alternating2(self, W_old, rew_old, ind, stdev=1):

        rew_new = copy.deepcopy(rew_old)
        W_new = copy.deepcopy(W_old)
        index = np.random.randint(len(W_old))
        rew_new[index] = rew_new[index] + stdev * np.random.randn() 
       
        return W_new, rew_new
                


    def initial_solution_bern_cnstr(self):

        pen_rew = np.random.uniform(-100, -10, size=self.num_features)
        W_new = np.random.randint(2, size=self.num_features)

        return W_new, pen_rew

   

  

    def run_mcmc_bern_constraint(self, samples, W_fix, stdev):

        '''
            run metropolis hastings MCMC with Gaussian symmetric proposal and uniform prior
            samples: how many reward functions to sample from posterior
            stepsize: standard deviation for proposal distribution
            normalize: if true then it will normalize the rewards (reward weights) to be unit l2 norm, otherwise the rewards will be unbounded
        '''
        
        num_samples = samples  # number of MCMC samples

        accept_cnt = 0  #keep track of how often MCMC accepts, ideally around 40% of the steps accept

        self.chain_cnstr = np.zeros((num_samples, self.num_mcmc_dims)) #store rewards found via BIRL here, preallocate for speed
        self.chain_rew = np.zeros((num_samples, self.num_mcmc_dims)) 
       
        cur_cnstr, cur_rew = self.initial_solution_bern_cnstr()
        cur_W = np.zeros(len(W_fix))
        for ii in range(len(cur_W)):
            if cur_cnstr[ii] == 1:
                cur_W[ii] = cur_rew[ii]

        cur_sol = W_fix + cur_W       
        
        cur_ll = self.calc_ll_pref(cur_sol, self.demonstrations, self.preferences)  # log likelihood
        #keep track of MAP loglikelihood and solution
        map_ll = cur_ll  
        map_cnstr = cur_cnstr
        map_W = cur_W
        map_list = []
        Perf_list = []
        for i in range(num_samples):
           
            prop_cnstr, prop_rew = self.generate_proposal_bern_constr_alternating(cur_cnstr, cur_W, i, stdev)
        
            prop_W = np.zeros(len(W_fix))
            for ii in range(len(prop_W)):
                if prop_cnstr[ii] == 1:
                    prop_W[ii] = prop_rew[ii]

            prop_sol = W_fix + prop_W
            prop_ll = self.calc_ll_pref(prop_sol, self.demonstrations, self.preferences)
    
            if prop_ll > cur_ll:
               
                self.chain_cnstr[i,:] = prop_cnstr
                self.chain_rew[i,:] = prop_rew
                accept_cnt += 1
                cur_W = prop_W
                cur_rew = prop_rew
                cur_cnstr = prop_cnstr
                cur_ll = prop_ll
                if prop_ll > map_ll:  # maxiumum aposterioi
                    map_ll = prop_ll
                    map_W = prop_W
                    map_rew = prop_rew
                    map_cnstr = prop_cnstr
            else:
                # accept with prob exp(prop_ll - cur_ll)
                if np.random.rand() < np.exp(prop_ll - cur_ll):
                    self.chain_cnstr[i,:] = prop_cnstr
                    self.chain_rew[i,:] = prop_rew
                    accept_cnt += 1
                    cur_W = prop_W
                    cur_rew = prop_rew
                    cur_ll = prop_ll
                    cur_cnstr = prop_cnstr
                else:
                    # reject
                    self.chain_cnstr[i,:] = cur_cnstr               
                    self.chain_rew[i,:] = cur_rew            
           
        self.accept_rate = accept_cnt / num_samples
        self.map_rew = map_rew
        self.map_W = map_W
        self.map_list = map_list
        self.map_cnstr = map_cnstr
        self.Perf_list = Perf_list
        
         

    def get_map_solution(self):

        return self.map_W, self.map_cnstr


    def get_mean_solution(self, burn_frac=0.1, skip_rate=1):

        ''' get mean solution after removeing burn_frac fraction of the initial samples and only return every skip_rate
            sample. Skiping reduces the size of the posterior and can reduce autocorrelation. Burning the first X% samples is
            often good since the starting solution for mcmc may not be good and it can take a while to reach a good mixing point
        '''
        Chain_W = self.chain_cnstr
        Chain_rew = self.chain_rew
        
        return Chain_W, Chain_rew
