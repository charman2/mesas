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