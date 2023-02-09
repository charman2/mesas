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

from ssm_utils import pmf_inv
# %%   
# MODEL part
# other quick utils

class Model(sas_Model):
    def __init__(self, *args, obs: dict, num_solute_ensemble: int=1, num_sas_ensemble: int=1, \
         block_size: int=1, sig_v: float = 0.001,
         input_model_params: dict = {"mode":"noise", "type":"constant","params": 0.01},
         obs_model_params: dict = {"mode":"noise", "type":"constant","params": [0.015, 0.01,0.02]},
           **kwargs) -> None:
        """
            This is the block for detailed settings
                +   input_model_params, theta
                        dictionary include  mode: GP or noise
                                            type: constant or varying
                                            params: for gaussian noise, variance around weekly bulk mean
                        If chosen mode == GP, no estimation on input_model_params
                +   obs_model_params, theta
                        dictionary include  mode: noise
                                            type: constant or varying
                                            params: in the order of sig_1, sig_2, sig_3, ..., sig_12, sig_13, ...
                +   transit_model_params, \theta
                        dictionary include ????? from transit model


                +   _num_blocks, K
                +   _block_length, total sample/K
                        currently constant, 
                +   _num_solute_ensemble, N
                        each represent one possible *input* scenario
                        total number of particles
                +   _num_sas_ensemble, D
                        each represent one possible *model* scenario
                        total number of SAS function set drawn from the model
                +   _sas_numsol, O
                +   _sas_solnames, each o

                +   A
                        ancestor particle index
                +   W
                        weight at last time step
                +   X
                        store state variable info

                +   Cqo
                        store observation values



        """
        # parameters related to input_model and obs_model, \theta
        self.input_model_params = input_model_params
        self.obs_model_params = obs_model_params
        
        # model settings 
        self._block_size = block_size # TODO: varying with observation
        self._num_solute_ensemble = num_solute_ensemble
        self._num_sas_ensemble = num_sas_ensemble

        # parameters related to transition_model, \theta
        if "solute_parameters" in kwargs:
            solute_parameters = _processinputs(kwargs.pop("solute_parameters"))
        elif "config" in kwargs:
            config = _processinputs(kwargs.pop("config"))
            solute_parameters = config['solute_parameters']
        else:
            raise ValueError("No solute_parameters found!")

        # more model settings
        self._sas_numsol = len(solute_parameters)
        self._sas_solnames = list(solute_parameters.keys())

        # for each o, initialize every sas function with the same set of parameter
        ssm_solute_parameters = {}
        for sol_name in self._sas_solnames:
            for d in range(self._num_sas_ensemble):
                # TODO: change this to extract from p(\theta) for future assignment
                new_sol_name = sol_name + " sas ensemble member " + d
                ssm_solute_parameters[new_sol_name] = solute_parameters[sol_name] 

        kwargs["solute_parameters"] = ssm_solute_parameters # TODO: does this still need to be there
        self.transit_model_params = ssm_solute_parameters # TODO: is this the correct thing to do?
        # inherit
        super().__init__(*args, **kwargs)

        # Create a thing called self._n_solorder that will be useful later
        # This stores the index of a solute ensemble member in the _solorder
        self._n_solorder = np.zeros(self._sas_numsol)
        # for each solute, record the n number of input sample' names
        for sol_name in self._sas_solnames:
            self._n_solorder[sol_name] = np.zeros(self._num_solute_ensemble)
            for d in range(self._num_solute_ensemble):
                new_sol_name = sol_name + " sol ensemble member " + d
                # give the index of given name in sol_order
                self._n_solorder[sol_name][d] = list(self._solorder).index(sol_name) 

        # more model settings, could be updated later
        self._num_blocks = np.ceil(self._timeseries_length/self._block_size)

        # A records the geneology of the ancestor sampling
        # Ancestor particle index matrix A: (K+1) x D x N x O (x obs # q)
        self.A = np.zeros((self._num_blocks+1, self._num_sas_ensemble, self._num_solute_ensemble, self._sas_numsol), dtype=int)
        # Weight only records at the end: D x N x O (x obs # q)
        self.W = np.zeros((self._num_sas_ensemble, self._num_solute_ensemble, self._sas_numsol), dtype=float)
        for d in range(self._num_sas_ensemble):
            self.A[0,d,:,:] = range(self._num_solute_ensemble)
            self.W[d,:,:] = 1./self._num_solute_ensemble
        # store X
        # for sT: (K+1) x D x N
        self.all_sT =  np.zeros((self._num_blocks+1, self._num_sas_ensemble, self._num_solute_ensemble), dtype=int)
        # for mT: (K+1) x D x N x O
        self.all_mT =  np.zeros((self._num_blocks+1, self._num_sas_ensemble, self._num_solute_ensemble, self._sas_numsol), dtype=int)

        # specify Z # TODO: what's the best way to pass observation into the model
        self.Cqo = obs

    # %%
    # =========================
    # Algo: Gibbs Sampler part
    # =========================
    def p_theta(self, pdf = False):
        '''
                +  prior distribution for \theta, as p(\theta), includes three types of estimation
                    +   input_model_params (o)
                        +   currently set to constant
                    +   obs_model_params (o)
                    +   transition_model_params (q)
        '''
        # TODO: add distribution on input_model and obs_model
        if pdf == False:
            if self.input_model_params.mode == "noise" and self.input_model_params.type == "constant":
                self.input_model_params.params = [self.input_model_params.params]*self._num_sas_ensemble
            else:
                pass

            if self.obs_model_params.type == "constant":
                self.obs_model_params.params = [self.obs_model_params.params]*self._num_sas_ensemble
            else:
                pass

            # for transit model params, assume all normal for now
            for sol_name in self._sas_solnames: # for each solute
                for d in range(self._num_sas_ensemble): # for each sas function
                    new_sol_name = sol_name + " sas ensemble member " + d
                    # TODO: what's the best way to extract theta
                    for each_param in self.transit_model_params[new_sol_name]:
                        # suppose all parameters follow a normal distribution

                        pass
        else: 

        return

    def W_wrapper(self, theta_candidates):
        '''
            provide a way to wrap theta according to q/o
            and then output theta_MAP
            theta_candidates:
                samples of theta (sas scenarios) passed to the model
                 _num_sas_ensemble (D) x _sas_numsol (O)

        '''
        # to think about this simply, we made 1, ..., o,..., O measurements of solute o
        # in one single flux q, q'' is not observed
        # TODO: how to get q from observation
        W_temp = np.zeros_like(self.W)
        # in the shape of _num_sas_ensemble (D) x _sas_numsol (O)
        p_theta_temp = self.p_theta(pdf= True) 
        # MAP estimator
        theta_MAP = np.zeros(self._sas_numsol)
        for sol in range(self._sas_numsol): # for each o
            for d in range(self._num_sas_ensemble): 
                W_temp[d,:,sol] = self.W[d,:,sol]*p_theta_temp[d,sol]
            ind_MAP = np.argmax(W_temp[d,:,sol].ravel)
            # TODO: this needs more work regarding the shape of theta
            theta_MAP[sol] = theta_candidates[(ind_MAP)//self._num_solute_ensemble,sol]
        return theta_MAP


    # --------------------Algo 1---------------------
    def run_pGS(self, L: int = 10000):
        """ 
            #### Step: particle Gibbs sampler ####
            
            Inputs: 

                +   L: chain length for MCMC
            Return:
                +   self.theta: total length of theta
                        L x D x number of theta
                +   best estimated X and Z
        """
        # ------------------- initialization step ------------------------
        self.theta = [None]*L
        # store the initial guess on parameter
        theta = self.p_theta()

        # first, run sMC to draw the very first sample trajectory
        self.run_sMC(theta.input_model_params, theta.obs_model_params,theta.transit_model_params,)
        theta[0] = self.W_wrapper() # general idea: theta[np.argmax(W)]
        # ------------------- MCMC step --------------------------
        # for the length of MCMC chain
        for l in range(1, L):
            # ------------------- theta* -------------------------
            # the optimal output from last model run is:
            theta_new = self.p_theta()
            # ------------------- X* -------------------------
            self.run_pGAS(theta_new.input_model_params,theta_new.obs_model_params,theta_new.transit_model_params)
            self.theta[l] = self.W_wrapper()
        return
                                                                                                                                                                
    # --------------------Algo 2---------------------
    def run_sMC(self, input_model_params=None, obs_model_params=None, transit_model_params=None):
        """ 
            #### Step: sequential Monte Carlo ####
            
            Inputs: 
                +   input_model_params: specify C_J model
                    +   input_model(*,input_model_params)

                +   obs_model_params: specify \Sigma = [\simga_D, \rho; \rho, \sigma_O18]
                    +   observation_model(*,obs_model_params)

                +   transition_model_params: specify p_q
            Return:
                +   self.theta: total length of theta
                        L x D x number of theta
                +   best estimated X and Z
        """
        # for each time segment to run MESAS
        #block_models = [self.copy_without_results(self) for i in self._num_sas_ensemble]

        block_model = self.copy_without_results()
        b = 0

        for k in range(1, self._num_blocks+1):
            # input model to generate new samples are included in transition model            

            block_model = self.transition_model(block_model, k, ancestors, input_model_params,transit_model_params)

            # update weights using the observation model, TODO: I am not sure about this way to call block model
            self.W_sol[k,b,:,:] = block_model.observation_model(obs_model_params)
            # normalize
            self.W_sol[k,b,:,:] = self.W_sol[k,b,:,:]/self.W_sol[k,b,:,:].sum(axis=-2)
            # set andcestors for the next round
                        
            for isol, sol in enumerate(self._solorder):
                # sampling from ancestor based on previous weight
                self.A[k+1,b,:,isol] = pmf_inv(self.W_sol[k,b,:,isol],self.A[k,b,:,isol], num = self._num_solute_ensemble) 
            ancestors = self.A[k,b,:,:]

        return 