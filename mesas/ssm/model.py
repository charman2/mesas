# %%
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

import scipy.stats as ss
from typing import List
import math
from sklearn.linear_model import LinearRegression

from mesas.sas.model import Model as sas_Model
from mesas.sas.model import _processinputs
# %%
# TODO: segmentate the model data
"""
# model structure notes

    Considering for MESAS:
    m_T|m_0 = model.get_mT(model.solorder) at the last timestep T
    C_Q | m_T = model.data_df['C --> Uniform']
    Goal: use particle filter to estimate theta

    Problem:
        current m_T|m_0 have no randomness:
        will it be possible to just run C_Q|m_T ???

"""
# %%
'''
# util function structure notes

1. I need a function to get m_T at the end of each timestep
--> this is my transition_model()
    --> this result will be use to initiate the next timestep
--> integrate prob dist'n using f_theta():
    --> currently normal, need improvement
    --> parameter, sig_v

2. I need a function to get c_Q at the end of each timestep
--> this is my observation_model()
--> this result is the actual estimation using m_T = f_theta(*)
    and will be compared with the model
--> prob dist'n will be estimate using g_theta()
    --> 

3. I need a function/wrapper to provide MESAS info at each timestep
'''

    

# %%   
# MODEL part
# models
from ssm_utils import transition_model,observation_model,f_theta,g_theta
# other quick utils
from ssm_utils import cal_distn, cal_CI

class Model(sas_Model):
    def __init__(self, *args, num_solute_ensemble:int=1, num_sas_ensemble:int=1, block_size: int=1, **kwargs) -> None:
        """
            input:  data_df: all data that MESAS needs
                    solute_parameter: specify solute parameters
                    sas_specs: specify function type and parameters, etc
                    M: number of particles
                    dT: length in time to run MESAS, e.g., 7 if run MESAS weekly for daily data
                    dt: time interval between measurements
        """
        # TODO: maybe inherit Class Model here??
        # parameters related to data

        self._block_size = block_size
        self._num_solute_ensemble = num_solute_ensemble
        self._num_sas_ensemble = num_sas_ensemble

        if "solute_parameters" in kwargs:
            solute_parameters = _processinputs(kwargs.pop("solute_parameters"))
        elif "config" in kwargs:
            config = _processinputs(kwargs.pop("config"))
            solute_parameters = config['solute_parameters']
        else:
            raise ValueError("No solute_parameters found!")
        self._sas_numsol = len(solute_parameters)
        self._sas_solnames = list(solute_parameters.keys())
        ssm_solute_parameters = {}
        for sol_name in self._sas_solnames:
            for m in range(self._num_solute_ensemble):
                new_sol_name = sol_name + " sol ensemble member " + m
                ssm_solute_parameters[new_sol_name] = solute_parameters[sol_name]

        kwargs["solute_parameters"] = ssm_solute_parameters

        super().__init__(*args, **kwargs)

        self._num_blocks = np.ceil(self._timeseries_length/self.block_size)

        # A records the geneology of the ancestor sampling
        self.A = np.zeros((self._num_blocks+1, self._num_sas_ensemble, self._num_solute_ensemble, self._sas_numsol), dtype=int)
        self.W_sol = np.zeros((self._num_blocks+1, self._num_sas_ensemble, self._num_solute_ensemble, self._sas_numsol), dtype=float)
        self.W_sas = np.zeros((self._num_blocks+1, self._num_sas_ensemble), dtype=float)

        for b in range(self._num_sas_ensemble):
            self.A[0,b,:,:] = range(self._num_solute_ensemble)
            self.W_sol[0,b,:,:] = 1./self._num_solute_ensemble
            self.W_sas[0,b] = 1./self._num_sas_ensemble
        #self.params_f = [] # params for f
        #self.params_g = [] # params for g
        #self.m_T_hat = {} # store all C_old for different solutes
    
    def transition_model(self, block_model, i):
        # update the block model data for the new block
        block_model.data_df = self.data_df[self.block_size*i:self.block_size*(i+1)]

        # Add stochastic error to the solute input
        for sol_name in self._sas_solnames:
            for m in range(self._num_solute_ensemble):
                new_sol_name = sol_name + " sol ensemble member " + m
                block_model.data_df[new_sol_name] = self._generate_input_error(block_model, sol_name)
                # TODO: make self._generate_input_error

        #update the block model initial conditions using 
        block_model.sT_init = block_model.get_sT(timestep=-1)
        block_model.mT_init = block_model.get_mT(timestep=-1)
        block_model.max_age = np.min(self._block_size * (i+1), self._timeseries_length)

        return block_model.run()  

    # running the actual model
    def run_sMC(self):
        '''
            Input:  params_f: parameters for state transition process
                    params_g: parameters for observation process
                                - dictionary key = sol name
            output: 
        
        '''
        # for each time segment to run MESAS
        #block_models = [self.copy_without_results(self) for i in self._num_sas_ensemble]

        block_model = self.copy_without_results(self)
        b = 0

        for i in range(self._numblocks):
            # data is chopped

            if i>0:
            # resampling step: choose particles according to last step
                for s in self._solorder:
                    # sampling from ancestor based on previous weight
                    self.A[i+1,b,:,s] = cal_distn(self.W_sol[i,b,:,s],self.A[i,b,:,s], num = self._num_solute_ensemble) 
            else:
                self.A[i+1,b,:,s] = range(self._num_solute_ensemble)

            block_model = transition_model(self, block_model, i)

            # update weights using the observation model
            for s in self._solorder:
                for m in range(self._num_solute_ensemble):
                    ancestor = self.A[i+1,b,m,s]
                    self.W_sol[i+1,b,m,s] = self.W_sol[i,b,ancestor,s] * self.g_theta(block_model)
                # normalize
                self.W_sol[i+1,b,:,s] = self.W_sol[i+1,b,:,s]/self.W_sol[i+1,b,:,s].sum()

        return

    def run_pMCMC(self,params_f: dict, params_g: dict,nstep: int):
        '''
        pMCMC inside loop, run this entire function as many as possible
        update w/ Ancestor sampling
            theta           - let it be params_f, params_g for now
            nstep           - number of steps in MCMC            
        '''
        T = int(np.ceil(self.max_age/self.block_size)) # the number of chunks in time
        p_theta_y = {}
        # for each solute
        for s in self.sol_names:
            # sample an ancestral path based on final weight
            B = np.zeros(T).astype(int)
            B[-1] = cal_distn(P[-1],qh[-1],num = 1)
            for i in reversed(range(1,T)):
                B[i-1] =  self.A[s][i][B[i]]
            # B is the current lineage for current solute
            notB = np.arange(0,self.num_particles)!=B[0]
            # save distribution
            p_theta_y[s] = 0
            # for each time chunk-------------------------------
            for t in range(1,T):
                notB = np.arange(0,self.num_particles)!=B[t] # sample states at last time step [t-1]
                # sampling from ancestor based on previous weight
                aa = cal_distn(P[t-1],A[t-1], num = len(notB)-1)
                self.A[s][t][notB] = aa
                # compute new state and weights based on saved model output
                self.m_T_hat[s][t] = self.m_T_hat[s][t][aa]
                phtp1 = P[t] 
                phtp1[notB] = pht * g_theta(qhtp1, sig_w, Q[t])
                p_theta_y.append(phtp1.sum())
                phtp1 = phtp1/phtp1.sum() # normalize
                # update the info
                qh[t][notB] = qhtp1
                P[t] = phtp1

        return

    def run_pGS(J,Q, delta_t, M,chain_len: int, k0 = 0.8, sig_v0 = 0.1, sig_w0=0.01):
        """
            for now, assume we just don't know k, and k ~ N(k0, 1)
        """
        # for the inital guess on theta_0 = {k_0, sig_v0, sig_w0}
        # run sMC to get a first step estimation
        qh, P, A = run_sMC(J, Q, k0, delta_t, M, sig_v0, sig_w0)
        k = [k0/delta_t]
        qh, P, A, p_theta_y = run_pMCMC(k[-1], sig_v0,sig_w0, qh, P, A, J, Q, delta_t)
        lm = LinearRegression(fit_intercept=False)
        for i in range(chain_len):
            # now run pMCMC
            k_new = []
            for j in range(M):
                reg = lm.fit(delta_t*(J[:-1]-qh[:-1,j]).reshape(-1,1),(qh[1:,j] - qh[:-1,j]).reshape(-1,1))        
                k_new.append(reg.coef_[0][0])
            counts, bins = np.histogram(k_new)
            k_new = dits(counts/counts.sum(),(bins[:-1]+bins[1:])/2, num = 1)
            k_new = ((bins[:-1]+bins[1:])/2)[k_new][0]
            # M-H 
            qh_new, P_new, A_new, p_theta_y_new = run_pMCMC(k_new, sig_v0,sig_w0, qh, P, A, J, Q, delta_t)
            ratio = min(1,ss.norm(k[-1],0.3).pdf(k_new)/ss.norm(k_new,0.3).pdf(k[-1])*math.prod(np.array(p_theta_y_new)/np.array(p_theta_y)))
            u = ss.uniform.rvs()
            if u <= ratio:
                k.append(k_new)
                qh = qh_new
                P = P_new
                A = A_new
            else:
                k.append(k[-1])
        return k
    

    
     

# %%
if __name__ == "__main__":
    """
        J: precip
        Q: discharge
        k: linear reservoir factor
        delta_t: timestep
        sig_q: importance sampling on state transition
        sig_v: noise on transition process
        sig_w: noise on observation process
        M: number of particles
    """
    # run_particle_filter_wrapper(J: List[float], Q: List[float], k: float,\
    # delta_t:float, sig_q: float, sig_v: float, sig_w: float, \
    # M: int = 100, plot = True)

    # read data
    precip = pd.read_csv("precip.csv", index_col = 0)
    precip = precip['45'].values
    # define constant
    l = 100 # control the length of the time series
    delta_t = 1./24/60*15
    k = 1
    sig_v = 0.1
    sig_w = 0.01 
    M = 50
    plot = True
    #
    J,Q = data_generation(precip, delta_t, k , l)
    Q_true = Q.copy()
    # pretend we know obs error
    Q += np.random.normal(0, sig_w,size = len(Q))
    T = len(Q) # total age
    k /= delta_t # adjust k
    
    qh, P, A = run_sMC(J, Q, k, delta_t,M,sig_v,sig_w)
    L, U, MLE = cal_CI(qh,P)

    # ------------
    if plot == True:
        plt.figure()

        plt.plot(np.linspace(0,T,len(Q)),Q,".", label = "Observation")
        plt.plot(np.linspace(0,T,len(Q_true)),Q_true,'k', label = "True")
        plt.plot(np.linspace(0,T,len(MLE)),MLE,'r:', label = "MLE")
        plt.fill_between(np.linspace(0,T,len(L)), L, U, color='b', alpha=.3, label = "95% CI")
        
        plt.legend(ncol = 4)
        plt.title(f"sig_v {sig_v}, sig_w {sig_w}")
        plt.xlim([600,630])

    qh, P, A,p_theta_y = run_pMCMC(k, sig_v, sig_w,qh, P, A, J, Q, delta_t)
    L, U, MLE = cal_CI(qh,P)
    # ------------
    if plot == True:
        plt.figure()

        plt.plot(np.linspace(0,T,len(Q)),Q,".", label = "Observation")
        plt.plot(np.linspace(0,T,len(Q_true)),Q_true,'k', label = "True")
        plt.plot(np.linspace(0,T,len(MLE)),MLE,'r:', label = "MLE")
        plt.fill_between(np.linspace(0,T,len(L)), L, U, color='b', alpha=.3, label = "95% CI")
        
        plt.legend(ncol = 4)
        plt.title(f"sig_v {sig_v}, sig_w {sig_w}")
        plt.ylim([-0.05,0.2])
        plt.xlim([600,630])

    k = run_pGS(J,Q, delta_t, M = M,chain_len = 100, k0 = 0.6, sig_v0 = 0.1, sig_w0=0.01)
    if plot == True:
        plt.figure()

        plt.plot(np.linspace(0,T,len(Q)),Q,".", label = "Observation")
        plt.plot(np.linspace(0,T,len(Q_true)),Q_true,'k', label = "True")
        plt.plot(np.linspace(0,T,len(MLE)),MLE,'r:', label = "MLE")
        plt.fill_between(np.linspace(0,T,len(L)), L, U, color='b', alpha=.3, label = "95% CI")
        
        plt.legend(ncol = 4)
        plt.title(f"sig_v {sig_v}, sig_w {sig_w}")
        plt.ylim([-0.05,0.2])
        plt.xlim([600,630])
        plt.figure()
        plt.plot(k, label= "MCMC")
        plt.plot([0,100],[1./delta_t,1./delta_t],"k:", label = "True k")
        plt.legend()