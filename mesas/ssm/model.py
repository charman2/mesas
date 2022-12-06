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
    def __init__(self, *args, obs:dict, num_solute_ensemble:int=1, num_sas_ensemble:int=1, \
         block_size: int=1, sig_v: float = 0.001,  **kwargs) -> None:
        """
            input:  
        """
        # parameters related to data
        self.sig_v = sig_v
        self.obs = obs

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

        self._num_blocks = np.ceil(self._timeseries_length/self._block_size)

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
        
    def _generate_input_error(self,model, sol_name, distribution = "normal",params = 0.01):
        if distribution == "normal":
            return ss.norm(model.data_df[f'{sol_name}'].values,params).rvs()
        else:
            raise ValueError("This distribution has not been implemented yet!") 

    def transition_model(self, block_model, i, precip_distribution: str = "normal"):
        # update the block model data for the new block
        block_model.data_df = self.data_df[self._block_size*i:self._block_size*(i+1)]

        # Add stochastic error to the solute input
        if precip_distribution == "GP":
            # Gaussian process result will be included in input file
            pass
        else:    
            for sol_name in self._sas_solnames:
                for m in range(self._num_solute_ensemble):
                    new_sol_name = sol_name + " sol ensemble member " + m
                    block_model.data_df[new_sol_name] = self._generate_input_error(block_model, sol_name, precip_distribution)
    

        #update the block model initial conditions using 
        block_model.sT_init = block_model.get_sT(timestep=-1)
        block_model.mT_init = block_model.get_mT(timestep=-1)
        block_model.max_age = np.min(self._block_size * (i+1), self._timeseries_length)

        return block_model.run()
    
    def g_theta(self, block_model,sig_v):
        pdf = np.zeros((self._numflux,self._numsol,self._num_solute_ensemble))
        for i,flux_name in enumerate(self._fluxorder):
            for j,sol_name in enumerate(self._sas_solnames):
                for m in range(self._num_solute_ensemble):
                    new_sol_name = sol_name + " sol ensemble member " + m
                    C_q_hat = block_model.data_df[f'{new_sol_name} --> {flux_name}']
                    C_q = self.obs[f'{flux_name}']
                    pdf[i,j,m] = ss.norm(C_q,sig_v).pdf(C_q_hat)
        return pdf
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

        block_model = self.copy_without_results()
        b = 0

        for i in range(self._num_blocks):
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
                    # TODO: double check the dimension for g_theta
                    self.W_sol[i+1,b,m,s] = self.W_sol[i,b,ancestor,s] * self.g_theta(block_model,self.sig_v)
                # normalize
                self.W_sol[i+1,b,:,s] = self.W_sol[i+1,b,:,s]/self.W_sol[i+1,b,:,s].sum()

        return 

    def run_pGAS(self, params):
        """
            put parameters here
        """
        self.run_sMC()
        b = 0
        cq_hat = np.zeros((self._num_blocks,self._num_solute_ensemble,self._sas_numsol))
        py = np.zeros((self._num_blocks,self._num_solute_ensemble,self._sas_numsol))
        # for each solute
        # TODO: add flux
        for s in range(self._sas_numsol):
            # sample an ancestral path based on final weight
            B = np.zeros(self._num_blocks).astype(int)
            B[-1] = cal_distn(self.W_sol[-1,b,:,s], self.get_mT(timestep=-1),num = 1)
            for t in reversed(range(self._num_blocks-1)):
                B[t-1] = self.A[t,b,B[t],s]
            # B is the current lineage for current solute
            notB = np.arange(0,self.num_particles)!=B[0]
            # TODO: is this the right way to get flux?
            cq_hat[:,B,s] = self.data_df[f'{self.sol_order[B]} --> {self.flux_order}']
            py[:,B,s] = self.W_sol[:,b,B,s]
            # for each time chunk-------------------------------
            for t in range(self._num_blocks):
                notB = np.arange(0,self.num_particles)!=B[t] # sample states at last time step [t-1]
                # sampling from ancestor based on previous weight
                aa = cal_distn(self.W_sol[t+1,b,:,s], self.get_mT(timestep=t+1), num = len(notB)-1)
                self.A[t,b,notB,s] = aa
                # mix chain 0:t with B t:T_max
                cq_hat[:,notB,s] = np.concatenate((self.data_df[f'{self.sol_order[notB]} --> {self.flux_order}'].values[:t+1][aa],\
                    cq_hat[:,B,s][t+1:self._num_blocks]),axis = 1)
                py[:,notB,s] = np.concatenate((self.W_sol[:t+1,b,notB,s], self.W_sol[t+1:self._num_blocks,b,notB,s]),axis = 1)
                # TODO: how to put the model here??
                py[:,notB,s] = py[:,notB,s] * self.g_theta(self, self.sig_v) 
                py[:,:,s] = py[:,:,s]/py[:,:,s].sum() # normalize
            self.W_sol[:,b,:,:] = py
        return cq_hat, py

    def run_pMMH(self, params: List[float], chain_len: int = 50, sig_v0: float = 0.001):
        """ 
            initial guess of parameters: params
            chain_len: evolution steps of MCMC chain
        """
        theta = np.zeros((params.shape[0],chain_len))
        # for the inital guess on theta_0 = {k_0, sig_v0, sig_w0}
        # run sMC to get a first step estimation
        # TODO: 1. draw theta for sas
        theta_new = 
        # TODO: 2. run pGAS
        cq_hat, py = self.run_pMCMC()
        for i in range(chain_len):
            cq_hat_new, py_new = self.run_pMCMC()
            # M-H 
            ratio = min(1,ss.norm(theta[-1],0.3).pdf(theta_new)/ss.norm(theta_new,0.3).pdf(theta[-1])*math.prod(np.array(py_new)/np.array(py)))
            u = ss.uniform.rvs()
            if u <= ratio:
                theta[i] = theta_new
                # TODO: Update get_mT, W ??
            else:
                theta[i] = theta[i-1]
        return theta
    

    
