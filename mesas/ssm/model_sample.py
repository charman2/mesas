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
    def __init__(self, *args, obs:dict, num_solute_ensemble: int=1, num_sas_ensemble: int=1, \
         block_size: int=1, sig_v: float = 0.001,  **kwargs) -> None:
        """
            Analysis of dimensions of the models:
            For D replicated of theta we have:
                For each flux q and each observation o at given theta:
                    generate N particles corresponding to precipitation
            ===========================================================
            +   sig_v - observation uncertainty, currently specified by users but infuture train with theta
                        dict{"sol_name": sig_v}
            +   obs - observations of outflux concentrations 
                        dict{"sol_name": obs}
            +   _block_size - this should be concordant with irregular observation in the future. 
                              Now it is simply a timestep to compare against observation
                        int: # of timesteps for a block
            +   _num_blocks: should be concordant with observation
                        int: # of calibration timesteps
            +   _num_solute_ensemble - total number of particles for each solutes --> input uncertianty
                        int: N
            +   _num_sas_ensemble - total number of sas functions --> parameter uncertainty
                        int: D --> not needed for multiple theta sampling
            +   _sas_numsol - total number of solutes
                        int: O
                +   _sas_solnames:  list(1, ..., o, ..., O)
            +   ssm_solute_parameters - initial guess on all pass to the function through kwags 
                        dict: {"sol_name for replicate 1"}: parameter for sol m  
                        dict key size = Q x D  
            + TODO: number of fluxes?
            + TODO: number of MCMC evolution
            + TODO: need a model to store block model outputted states??


        """
        self.theta = np.zeros(L,D)
        self.K # should follow model observation gap
        
        
        
        
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

        # set up parameter format
        ssm_solute_parameters = {}
        for sol_name in self._sas_solnames:
            for m in range(self._num_sas_ensemble):
                new_sol_name = sol_name + " sas ensemble member " + m
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
        self.W = np.zeros((self._num_blocks+1, self._num_sas_ensemble, self._num_solute_ensemble, self._sas_numsol), dtype=float)
        for b in range(self._num_sas_ensemble):
            self.A[0,b,:,:] = range(self._num_solute_ensemble)
            self.W[0,b,:,:] = 1./self._num_solute_ensemble


        
    # ================================
    # This is the model to draw inputs, will need to integrate from previous GP model
    # ================================  
    def input_model(self, params={"mode":"noise","distribution":"normal","sig":0.01}):
        '''
            For input uncertainty, currently two modes:
                1. Gaussian Process using processed input data
                2. Add noise to input observation
                    - currently, normal distribution
        '''

        # Add stochastic error to the solute input
        if params.mode == "GP":
            # Gaussian process result will be included in input file
            pass

        # use white noise around average
        elif params.mode == "noise":
            for sol_name in self._sas_solnames:
                # for each input solutes
                for n in range(self._num_solute_ensemble):
                    new_sol_name = sol_name + " sol ensemble member " + n
                    # add a random normal noise on C_J
                    if params.distribution == "normal":
                        # TODO: this name may not be correct! Double check!!!!!!!!---------------------------------------------------------
                        C_J_m = ss.norm(self.data_df[sol_name].values, params.sig).rvs() # this should have the same length as block model length 
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
        block_model.input_model(params = input_model_params)

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
            if 'observations' in block_model.solute_parameters[sol]:
                for iflux, flux in enumerate(self._comp2learn_fluxorder):
                    if flux in block_model.solute_parameters[sol]['observations']:
                        obs = block_model.solute_parameters[sol]['observations'][flux]
                    residual = block_model.data_df[f'residual {flux}, {sol}, {obs}']
                    likelihood[:,isol] = likelihood[:,isol] * ss.norm(0,obs_model_params.sig_v).pdf(residual)
        return likelihood

    # running the actual model
    def run_sMC(self, input_model_params=None, obs_model_params=None, transition_model_params=None):
        '''
            This is the first run to get ref trajectory
        
        '''
        # for each time segment to run MESAS
        #block_models = [self.copy_without_results(self) for i in self._num_sas_ensemble]

        block_model = self.copy_without_results()
        b = 0

        for k in range(1, self._num_blocks+1):
            # input model to generate new samples are included in transition model

            # data is chopped1
            for isol, sol in enumerate(self._solorder):
                # sampling from ancestor based on previous weight
                self.A[k,b,:,isol] = pmf_inv(self.W[k-1,b,:,isol],self.A[k-1,b,:,isol], num = self._num_solute_ensemble) 
            ancestors = self.A[k,b,:,:]

            block_model = self.transition_model(block_model, k, ancestors, input_model_params)

            # update weights using the observation model, TODO: I am not sure about this way to call block model
            self.W[k,b,:,:] = self.W[k-1,b,ancestors,:] * block_model.observation_model(obs_model_params)
            # normalize
            self.W[k,b,:,:] = self.W[k,b,:,:]/self.W[k,b,:,:].sum(axis=-2)

        return 

    def run_pGAS(self, input_model_params):
        """
            put parameters here
        """
        
        b = 0
        cq_hat = np.zeros((self._num_blocks,self._num_solute_ensemble,self._sas_numsol))
        w = np.zeros((self._num_blocks,self._num_solute_ensemble,self._sas_numsol))
        # for each solute
        # TODO: add flux
        # for solute o
        for o in range(self._sas_numsol):
            # sample an ancestral path based on final weight
            B = np.zeros(self._num_blocks).astype(int)
            B[-1] = pmf_inv(self.W[-1,b,:,o], self.get_mT(timestep=-1),num = 1)
            for k in reversed(range(self._num_blocks-1)):
                B[k-1] = self.A[k,b,B[k],o]
            # B is the current lineage for current solute
            notB = np.arange(0,self.num_particles)!=B[0]
            # TODO: is this the right way to get flux?
            cq_hat[:,B,o] = self.data_df[f'{self.sol_order[B]} --> {self.flux_order}']
            
            # the first time step will be uniformly sampled 
            w[:,B,o] = 1/self._num_solute_ensemble
            # for each time chunk-------------------------------
            for k in range(self._num_blocks):
                notB = np.arange(0,self.num_particles)!=B[k] # sample states at last time step [t-1]
                # sampling from ancestor based on previous weight
                aa = pmf_inv(self.W[k+1,b,:,o], self.get_mT(timestep=k+1), num = len(notB)-1)
                self.A[k,b,notB,o] = aa
                
                # now run the model for notB
                block_model = self.transition_model(block_model, k, aa, input_model_params)
                # mix chain 0:t with B t:T_max
                # TODO: write a function to get joint inverse distribution for m_T and s,T
                

                cq_hat[:,notB,o] = np.concatenate((self.data_df[f'{self.sol_order[notB]} --> {self.flux_order}'].values[:k+1][aa],\
                    cq_hat[:,B,o][k+1:self._num_blocks]),axis = 1)
                w[:,notB,o] = np.concatenate((self.W[:k+1,b,notB,o], self.W[k+1:self._num_blocks,b,notB,o]),axis = 1)
                # TODO: how to put the model here??
                w[:,notB,o] = w[:,notB,o] * self.g_theta(self, self.sig_v) 
                w[:,:,o] = w[:,:,o]/w[:,:,o].sum() # normalize
            self.W[:,b,:,:] = w
        return cq_hat, w

# --------------------this is for sample \theta, which enables parallel computing---------------------
def run_pGS(self, theta0: dict, L: int = 1000):
    """ 
        #### Step: particle Gibbs sampler ####
        
        Inputs: 
            +   theta0: initial guess p(\theta) 
                    D x number of theta
                +   two sets of theta: one controls q through pq
                                       one controls o through correlation
                +   input_model_params (o), obs_model_params (o), transition_model_params (q)
            +   L: chain length for MCMC
        Return:
            +   self.theta: total length of theta
                    L x D x number of theta
            +   best estimated X and Z
    """
    # ------------------- initialization step --------------------------
    # store the initial guess on parameter
    self.theta[0] = theta0 
    D = self._num_sas_ensemble # total combination of sas functions
    O = self._sas_numsol
    # first, run sMC to draw the very first sample trajectory
    self.run_sMC(theta0[input_model_params], theta0[obs_model_params], theta0[transition_model_params])
    # ------------------- MCMC step --------------------------
    # for the length of MCMC chain
    for l in range(1, L):
        # ------------------- theta* -------------------------
        # the optimal output from last model run is:
        theta_new = np.zeros_like(theta0)
        for d in range(D):
            X_K,W_K,theta_index = {},{},{}
            for sol_name in self._sas_solname: # each o
                # From d-th block model, the reference trajectory:
                X_K[sol_name] = f_theta(self.model, sol_name) # or model.get_CT
                for flux_name in self.fluxorder: # each q
                    X_K[sol_name] * pq
                    # TODO: W_K is for qo how to make it to be suitable for o
                    #       Maybe W_{\ o} <- \sum_q W_{qo}
                    W_K[sol_name] = self.W_sol # sum over o
                    W_K[sol_name] = W_K[sol_name]/W_K[sol_name].sum()
            # for theta related to q
                theta_index[sol_name] = pmf_inv(W_K[sol_name],X_K[sol_name],num = D)
        self.theta[l] = theta_new
        # ------------------- X* -------------------------
        self.run_pGAS(self.theta[l])
    return self.theta, 

    
