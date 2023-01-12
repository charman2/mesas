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
# MODEL STRUCTURE NOTE

* Observation
    C_Q | m_T = model.data_df['C --> Uniform']
    Goal: use particle filter to estimate theta

"""
# %%
'''
# SECTION FOR ALGO NOTE

'''

    

# %%   
# MODEL part
# other quick utils
from ssm_utils import pmf_inv, cal_CI

class Model(sas_Model):
    def __init__(self, *args, obs:dict, theta_0: dict, num_solute_ensemble: int=1, num_sas_ensemble: int=1, \
         block_size: int=1, sig_v: float = 0.001,  **kwargs) -> None:
        """
            Analysis of dimensions of the models:
            For D replicated of theta we have:
                For each flux q and each observation o at given theta:
                    generate N particles corresponding to precipitation
            ===========================================================
            +   sig_v - observation uncertainty, currently specified by users but infuture train with theta
            +   obs - observations of outflux concentrations 
            +   theta0 - initial guess on all thetas
            +   _block_size - this should be concordant with irregular observation in the future. Now it should be 
            +   _num_solute_ensemble - 
            +   _num_sas_ensemble - 
        """
        # parameters related to data
        self.sig_v = sig_v
        self.obs = obs
        self.theta_0 = theta_0

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

        # Create a thing called self._m_solorder that will be useful later
        # This stores the index of a solute ensemble member in the _solorder
        self._m_solorder = np.zeros(self._sas_numsol)
        for sol in self._sas_solnames:
            self._m_solorder[sol] = np.zeros(self._num_solute_ensemble)
            for m in range(self._num_solute_ensemble):
                new_sol_name = sol_name + " sol ensemble member " + m
                self._m_solorder[sol][m] = list(self._solorder).index(new_sol_name)

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
        
    def input_model(self, params={"mode":"white","distribution":"normal","sig":0.01}):

                # Add stochastic error to the solute input
        if params.mode == "GP":
            # Gaussian process result will be included in input file
            pass
        elif params.mode == "white":
            for sol_name in self._sas_solnames:
                for m in range(self._num_solute_ensemble):
                    new_sol_name = sol_name + " sol ensemble member " + m
                    if params.distribution == "normal":
                        C_J_m = ss.norm(self.data_df[sol_name].values, params.sig).rvs()
                        self.data_df[new_sol_name] =  C_J_m
                    else:
                        raise ValueError("This distribution has not been implemented yet!")  
        raise ValueError("Input model not implemented yet!") 

    def get_block_ind(self, k):
        return np.arange(self._block_size*(k-1),self._block_size*(k))+1

    def transition_model(self, block_model, k, ancestors, input_model_params=None, transition_model_params=None):
        """
        For solute o at flux q
            m_T^{k}|m_T^{k-1} = model.get_mT(model.solorder) 
        """
        # update the block model data for the new block
        block_model.data_df = self.data_df[self.get_block_ind(k)]
        block_model.max_age = np.min(self._block_size * k, self._timeseries_length)

        # Ask the block model to generate input scenarios
        block_model.input_model(input_model_params)

        #update the block model initial conditions using 
        block_model.sT_init = block_model.get_sT(timestep=-1)

        # Shuffle the initial conditions according to the ancestor resampling
        mT_prev = block_model.get_mT(timestep=-1)
        for sol in self._sas_solnames:
            for m in range(self._num_solute_ensemble):
                block_model.mT_init[:,self._m_solorder[sol][m]] = mT_prev[:,ancestors[self._m_solorder[sol][m]]]


        block_model.run()
        block_model.get_residuals()

        return block_model
    
    def observation_model(self, block_model,obs_model_params):
        likelihood = np.ones((self._num_solute_ensemble, self._numsol))
        for isol, sol in enumerate(self._solorder):
            if 'observations' in self.solute_parameters[sol]:
                for iflux, flux in enumerate(self._comp2learn_fluxorder):
                    if flux in self.solute_parameters[sol]['observations']:
                        obs = self.solute_parameters[sol]['observations'][flux]
                    residual = self.data_df[f'residual {flux}, {sol}, {obs}']
                    likelihood[:,isol] = likelihood[:,isol] * ss.norm(0,obs_model_params.sig_v).pdf(residual)
        return likelihood

    # running the actual model
    def run_sMC(self, input_model_params=None, obs_model_params=None, transition_model_params=None):
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

        for k in range(1, self._num_blocks+1):
            # data is chopped
            for isol, sol in enumerate(self._solorder):
                # sampling from ancestor based on previous weight
                self.A[k,b,:,isol] = pmf_inv(self.W_sol[k-1,b,:,isol],self.A[k-1,b,:,isol], num = self._num_solute_ensemble) 
            ancestors = self.A[k,b,:,:]

            block_model = self.transition_model(block_model, k, ancestors, input_model_params)

            # update weights using the observation model
            self.W_sol[k,b,:,:] = self.W_sol[k-1,b,ancestors,:] * block_model.observation_model(obs_model_params)
            # normalize
            self.W_sol[k,b,:,:] = self.W_sol[k,b,:,:]/self.W_sol[k,b,:,:].sum(axis=-2)

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
            B[-1] = pmf_inv(self.W_sol[-1,b,:,s], self.get_mT(timestep=-1),num = 1)
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
                aa = pmf_inv(self.W_sol[t+1,b,:,s], self.get_mT(timestep=t+1), num = len(notB)-1)
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

    def run_GS(self, params: List[float], chain_len: int = 50, sig_v0: float = 0.001):
        """ 
            This is the model to run from parameters
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
    

    
