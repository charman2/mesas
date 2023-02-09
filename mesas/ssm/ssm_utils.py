# %%
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from mesas.sas.model import Model
import scipy.stats as ss
import math
from typing import List
from sklearn.linear_model import LinearRegression 
# %%
def f_theta(model: Model, sol: str) -> dict:
    """
        #### State transition ####
        Inputs:
            +   model: MESAS model class
            +   sol: name of the solute
        Return:
            +   Estimated C_To at final timestep
    """
    m_To = model.get_mT(sol)
    s_T = model.get_sT()
    return m_To[:,-1]/s_T[:,-1]
# %%
# TODO: how to isolate transition model and observation model????????
def observation_model(model: Model,sol:str, flux:str) -> dict:
    """
        input:  MESAS model class
                sol: name of the solute
                flux: name of the flux
        output: np.array of C
    """   
    return model.data_df[f"{sol} --> {flux}"].values
# %%
# TODO: it's not really been used
def f_theta(model: Model, params: List[float], pdf_x = None, output = "rvs", distribution = "normal"):
    """
        input: model: transition model output 
               params: corresponding to a stats model (specified in "distribution")
               pdf_x: resampled states 
               output: "pdf" or "rvs" depends on output need
               distribution: currently normal # TODO: multiple distribution for each sol
        output: pdf(x)/rvs dict based on probabilistic distribution of transition model
    """
    dic_mT = transition_model(model)
    sols = list(dic_mT.keys())
    result = {}
    # compute for each solution
    for i,sol in enumerate(sols):
        mT = dic_mT.get(sol)
        if distribution == "normal":
            # for normal distribution, there is only one parameter: st.dev
            stats_model = ss.norm(mT,params[i])
        else:
            # TODO: test other distribution
            raise Exception("You have not implement stats model other than normal!!")
        # based on output type
        if output == "pdf":
            result[sol] = stats_model.pdf(pdf_x)
        else:
            result[sol] = stats_model.rvs()
    return result
# %%
def g_theta(model: Model, sol:str,flux:str, params: List[float], pdf_x, distribution = "normal"):
    """
        input:  model: transition model output 
                sol: current solute
                flux: current flux
                params: corresponding to a stats model (specified in "distribution")
                pdf_x: the input to generate pdf
        output: pdf(x)
    """
    cT = observation_model(model,sol,flux)
    if distribution == "normal":
        # for normal distribution, there is only one parameter: st.dev
        stats_model = ss.norm(cT,params)
    else:
        # TODO: test/add other distribution
        raise Exception("You have not implement stats model other than normal!!")
    return stats_model.pdf(pdf_x)
# %%
# calculate discrete inverse transform sampling, no close form
def pmf_inv(pmf: List[float],x: List[float],num: int = 1):
    '''
        give x ~ pmf
        num: number of samples needed, default set to 1
    return: index of x that are been sampled according to discrete pdf
    '''
    ind = np.argsort(x) # sort x according to its magnitude
    pmf = pmf[ind] # sort pdf accordingly
    cmf = pmf.cumsum()
    u = np.random.uniform(size = num)
    u_sort_ind = np.argsort(u)
    u_sort = u[u_sort_ind]
    result_ind = np.zeros(num)
    left = 0
    for i, uu in enumerate(u_sort):  
        while left <= len(cmf):
            if uu >= cmf[left]:
                left += 1
            if uu < cmf[left]:
                break               
        result_ind[u_sort_ind[i]] = left
    return ind[result_ind.astype(int)]
# %%
# calculate CI
def cal_CI(x: List[float],P: List[float], alpha: float = 0.05):
    '''
        give x ~ P at time i
        calculate (1 - alpha)% CI
    return: L - lower bound, U - upper bound, MLE
    '''
    L = []
    U = []
    MLE = []
    for i in range(len(x)):
        X_ind = np.argsort(x[i]) # sort states
        pmf = P[i][X_ind] # calculate pmf
        cmf = pmf.cumsum() # cmf
        i_mle = np.argmax(pmf)
        i_lower = np.argmin(abs(cmf - alpha/2))
        i_upper = np.argmin(abs(cmf - (1-alpha/2)))
        MLE.append(x[i][X_ind[i_mle]])
        L.append(x[i][X_ind[i_lower]])
        U.append(x[i][X_ind[i_upper]])
    return L, U, MLE
# %%
def pmf_rv(pmf, x, x_prime):
    '''
        give x ~ pmf
        x_prime: number to find where it is in the pmf
    return: index of x that are been sampled according to discrete pdf
    '''
    ind = np.argsort(x) # sort x according to its magnitude
    x_sort = x[ind] # sort x accordingly
    pmf = pmf[ind] # sort pdf accordingly
   
    for i,xx in enumerate(x_sort):
        if x_prime < xx:
            break
    pmf[i] *= 
    return ind[i]

# %%
steady_benchmarks = {
	'Uniform':{
		'spec':{
			#"func": "kumaraswamy",
			#"args": {"a": 1.0-0.000000001, "b":1.0-0.000000001, "scale": "S_0", "loc": "S_m"},
			'ST': ['S_m', 'S_m0']
        },
        'pQdisc': lambda delta, i: (-1 + np.exp(delta))**2/(np.exp((1 + i)*delta)*delta),
        'pQdisc0':lambda delta: (1 + np.exp(delta)*(-1 + delta))/(np.exp(delta)*delta),
        'subplot': 0,
        'distname': 'Uniform'
	},
    'Exponential':{
		'spec':{
			"func": "gamma",
			"args": {"a": 1.0, "scale": "S_0", "loc": "S_m"},
        },
        'pQdisc': lambda delta, i: (2*np.log(1 + i*delta) - np.log((1 + (-1 + i)*delta)*(1 + delta + i*delta)))/delta,
        'pQdisc0':lambda delta: (delta + np.log(1/(1 + delta)))/delta,
        'subplot': 1,
        'distname': 'Gamma(1.0)'
    },
    'Biased old (Beta)':{
		'spec':{
			"func": "beta",
			"args": {"a": 2.0-0.000000001, "b":1.0-0.000000001, "scale": "S_0", "loc": "S_m"},
        },
        'pQdisc': lambda delta, i: (2*1/np.cosh(delta - i*delta)*1/np.cosh(delta + i*delta)*np.sinh(delta)**2*np.tanh(i*delta))/delta,
        'pQdisc0':lambda delta: 1 - np.tanh(delta)/delta,
        'subplot': 2,
        'distname': 'Beta(2,1)'
    },
    'Biased young (Beta)':{
		'spec':{
			"func": "beta",
			"args": {"a": 1.0-0.000000001, "b":2.0-0.000000001, "scale": "S_0", "loc": "S_m"},
        },
        'pQdisc': lambda delta, i: (2*delta)/((1 + (-1 + i)*delta)*(1 + i*delta)*(1 + delta + i*delta)),
        'pQdisc0':lambda delta: delta/(1 + delta),
        'subplot': 3,
        'distname': 'Beta(1,2)'
    },
    'Partial bypass (Beta)':{
		'spec':{
			"func": "beta",
			"args": {"a": 1.0/2, "b":1.0, "scale": "S_0", "loc": "S_m"},
        },
        'pQdisc': lambda delta, i: (M(delta, -1 + i) - 2*M(delta, i) + M(delta, 1 + i))/delta,
        'pQdisc0':lambda delta: (-1 + delta + M(delta, 1))/delta,
        'subplot': 4,
        'distname': 'Beta(1/2,1)'
    },
    'Partial piston (Beta)':{
		'spec':{
			"func": "beta",
			"args": {"a": 1.0, "b":1.0/2, "scale": "S_0", "loc": "S_m"},
        },
        'pQdisc': lambda delta, i: delta/2,
        'pQdisc0':lambda delta: delta/4,
        'subplot': 5,
        'distname': 'Beta(1,1/2)'
    },
    'Biased old (Kumaraswamy)':{
		'spec':{
			"func": "kumaraswamy",
			"args": {"a": 2.0-0.000000001, "b":1.0-0.000000001, "scale": "S_0", "loc": "S_m"},
        },
        'pQdisc': lambda delta, i: (2*1/np.cosh(delta - i*delta)*1/np.cosh(delta + i*delta)*np.sinh(delta)**2*np.tanh(i*delta))/delta,
        'pQdisc0':lambda delta: 1 - np.tanh(delta)/delta,
        'subplot': 2,
        'distname': 'Kumaraswamy(2,1)'
    },
    'Biased young (Kumaraswamy)':{
		'spec':{
			"func": "kumaraswamy",
			"args": {"a": 1.0-0.000000001, "b":2.0-0.000000001, "scale": "S_0", "loc": "S_m"},
        },
        'pQdisc': lambda delta, i: (2*delta)/((1 + (-1 + i)*delta)*(1 + i*delta)*(1 + delta + i*delta)),
        'pQdisc0':lambda delta: delta/(1 + delta),
        'subplot': 3,
        'distname': 'Kumaraswamy(1,2)'
    },
    'Partial bypass (Kumaraswamy)':{
		'spec':{
			"func": "kumaraswamy",
			"args": {"a": 1.0/2, "b":1.0, "scale": "S_0", "loc": "S_m"},
        },
        'pQdisc': lambda delta, i: (M(delta, -1 + i) - 2*M(delta, i) + M(delta, 1 + i))/delta,
        'pQdisc0':lambda delta: (-1 + delta + M(delta, 1))/delta,
        'subplot': 4,
        'distname': 'Kumaraswamy(1/2,1)'
    },
    'Partial piston (Kumaraswamy)':{
		'spec':{
			"func": "kumaraswamy",
			"args": {"a": 1.0, "b":1.0/2, "scale": "S_0", "loc": "S_m"},
        },
        'pQdisc': lambda delta, i: delta/2,
        'pQdisc0':lambda delta: delta/4,
        'subplot': 5,
        'distname': 'Kumaraswamy(1,1/2)'
    }
}