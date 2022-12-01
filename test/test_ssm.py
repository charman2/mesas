# %%
from frame import SSM_frame
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mesas.sas.model import Model
import logging
# %%
# create input 
np.random.seed(2)
timeseries_length = 100
max_age = timeseries_length
dt = 0.1
Q_0 = 1.0 # <-- steady-state flow rate
C_J = 1.0 + np.random.randn(timeseries_length)*1.0

C_old = 1.0
J = Q_0
S_0 = 5 * Q_0
S_m = 1 * Q_0
n_substeps = 1
debug = False
verbose = False
jacobian = False

data_df = pd.DataFrame(index=range(timeseries_length))
data_df['t'] = data_df.index * dt
data_df["J"] = J
data_df["S_0"] = S_0
data_df["S_m"] = S_m
data_df["S_m0"] = S_m + S_0
data_df["C"] = C_J

from ssm_utils import steady_benchmarks
letter = 'abcdefghijklmnopqrstuvwxyz'

def M(delta, i):
    return np.where(i==0, 1, -lambertw(-np.exp(-1 - (i*delta)/2.))*(2 + lambertw(-np.exp(-1 - (i*delta)/2.))))

def RMS(x):
    return np.sqrt(np.mean(x**2))
makefigure = True
if makefigure:
    fig = plt.figure(figsize=[21,12])
    nrow = 4
    ncol = 6 #int(len(steady_benchmarks)/nrow)+1
    figcount=0
    ax0_dict = {}
    ax1_dict = {}
    ax2_dict = {}
    ax3_dict = {}
for name, bm in steady_benchmarks.items():
    i = np.arange(timeseries_length)
    pQdisc = np.zeros_like(i, dtype=float)
    delta = dt * Q_0 / (S_0)
    Tm = S_m/Q_0
    pQdisc[0] = bm['pQdisc0'](delta) / dt
    pQdisc[1:] = bm['pQdisc'](delta, i[1:]) / dt
    im = int(Tm/dt)
    data_df[f'{name} benchmark C'] = C_old
    data_df[f'{name} benchmark C'][im:] = (np.convolve(C_J, pQdisc, mode='full')[:timeseries_length-im] * dt
    + C_old * (1-np.cumsum(pQdisc)[:timeseries_length-im]*dt))
    data_df[f'{name} benchmark t'] = data_df['t']

    data_df[name] = Q_0
    solute_parameters = {
        "C":{
            "C_old": C_old, 
            "observations": {name:f'{name} benchmark C'},
        }
    }
    sas_specs = {name:{f'{name}_SAS':bm['spec']}}

    model = Model(
        data_df,
        sas_specs=sas_specs,
        solute_parameters=solute_parameters,
        debug=debug,
        verbose=verbose,
        dt=dt,
        n_substeps=n_substeps,
        jacobian=jacobian,
        max_age=max_age,
        warning=True
    )
    model.run()
    data_df = model.data_df
    err01 = -(data_df[f'{name} benchmark C'].values-data_df[f'C --> {name}'].values)#/data_df[f'{name} benchmark C'].values
    logging.info(f'{name} error01 = {err01.mean()}')

    model = Model(
        data_df,
        sas_specs=sas_specs,
        solute_parameters=solute_parameters,
        debug=debug,
        verbose=verbose,
        dt=dt,
        n_substeps=n_substeps*10,
        jacobian=jacobian,
        max_age=max_age,
        warning=True
    )
    model.run()
    data_df = model.data_df
    err10 = -(data_df[f'{name} benchmark C'].values-data_df[f'C --> {name}'].values)#/data_df[f'{name} benchmark C'].values
    logging.info(f'{name} error10 = {err10.mean()}')
    assert err10.mean()<1E-3
    if makefigure:
        #icol = int(figcount/nrow)
        #irow = figcount - nrow * icol
        if bm['subplot'] not in ax0_dict.keys():
            ax0 = plt.subplot2grid((nrow, ncol),(0, bm['subplot']))
            ax0.set_title(name.split('(')[0].strip())
            ax0.set_ylim((-0.02, 0.6))
            #ax0.set_ytick([1000, 2000])
            ax0.set_xlabel(r'$S_T$')
            ax0.spines.top.set_visible(False)
            #ax0.spines.bottom.set_visible(False)
            ax0.spines.right.set_visible(False)
            ax0.spines.bottom.set_position(('outward', 5))
            if bm['subplot']>0:
                ax0.get_yaxis().set_visible(False)
                ax0.spines.left.set_visible(False)
            else:
                ax0.spines.left.set_position(('outward', 10))
                ax0.set_ylabel(r'SAS function $\omega(S_T)$')
            ax0_dict[bm['subplot']] = ax0
            ax0.annotate(letter[0*ncol+bm['subplot']], (-0.12,1.1), xycoords='axes fraction', fontsize='x-large', fontweight='bold')
        else:
            ax0 = ax0_dict[bm['subplot']]
        print(name)
        ST = np.r_[[0, S_m-0.0001, S_m], np.linspace(S_m,(S_0+S_m),1000), [S_0+S_m], S_0+S_m+0.0001+np.linspace(0,1,10)]
        if name=='Uniform':
            SASpdf = np.zeros_like(ST)
            SASpdf[3:1003] = 1/S_0
        else:
            SASpdf = model.sas_specs[name].components[f'{name}_SAS'].sas_fun[0].func.pdf(ST)
        SASpdf[2]= np.NaN
        SASpdf[-11] = np.NaN
        ax0.plot(ST, SASpdf, alpha=0.6, lw=2, ls='-', label=bm['distname'])
        ax0.legend(frameon=False)
        if bm['subplot'] not in ax1_dict.keys():
            ax1 = plt.subplot2grid((nrow, ncol),(1, bm['subplot']))
            #ax1.set_title(name.split('(')[0].strip())
            #ax1.set_ylim((900, 2100))
            ax1.set_ylim((0.75, 1.25))
            #ax1.set_ytick([1000, 2000])
            ax1.set_xlabel('Time')
            ax1.plot(data_df[f'{name} benchmark t'], data_df[f'{name} benchmark C'], 'k', alpha=0.5, ls='--', lw=1.0, zorder=100, label='Benchmark')
            ax1.legend(frameon=False)
            ax1.spines.top.set_visible(False)
            #ax1.spines.bottom.set_visible(False)
            ax1.spines.right.set_visible(False)
            ax1.spines.bottom.set_position(('outward', 10))
            if bm['subplot']>0:
                ax1.get_yaxis().set_visible(False)
                ax1.spines.left.set_visible(False)
            else:
                ax1.spines.left.set_position(('outward', 10))
                ax1.set_ylabel('Tracer conc.')
            ax1_dict[bm['subplot']] = ax1
            ax1.annotate(letter[1*ncol+bm['subplot']], (-0.12,1.1), xycoords='axes fraction', fontsize='x-large', fontweight='bold')
        else:
            ax1 = ax1_dict[bm['subplot']]
        ax1.plot(data_df['t'], data_df[f'C --> {name}'], alpha=0.6, lw=2)
        ax1.legend(frameon=False)
        if bm['subplot'] not in ax2_dict.keys():
            ax2 = plt.subplot2grid((nrow, ncol),(2, bm['subplot']))
            #ax2.set_ylim((500, 2500))
            #ax2.set_ytick([1000, 2000])
            ax2.set_xlabel('Time')
            ax2.spines.top.set_visible(False)
            #ax2.spines.bottom.set_visible(False)
            ax2.spines.right.set_visible(False)
            ax2.spines.bottom.set_position(('outward', 10))
            if bm['subplot']>0:
                pass
            #    ax2.get_yaxis().set_visible(False)
            #    ax2.spines.left.set_visible(False)
            else:
                ax2.set_ylabel('Error (1 substep)')
            #    ax2.spines.left.set_position(('outward', 10))
            ax2_dict[bm['subplot']] = ax2
            ax2.annotate(letter[2*ncol+bm['subplot']], (-0.12,1.1), xycoords='axes fraction', fontsize='x-large', fontweight='bold')
        else:
            ax2 = ax2_dict[bm['subplot']]
        ax2.plot(data_df['t'], err01, alpha=0.6, lw=2, label=f" RMSE = {RMS(err01):.2e}")
        ax2.legend(frameon=False, loc='lower left')
        if bm['subplot'] not in ax3_dict.keys():
            ax3 = plt.subplot2grid((nrow, ncol),(3, bm['subplot']))
            #ax3.set_ylim((500, 2500))
            #ax3.set_ytick([1000, 2000])
            ax3.set_xlabel('Time')
            ax3.spines.top.set_visible(False)
            #ax3.spines.bottom.set_visible(False)
            ax3.spines.right.set_visible(False)
            ax3.spines.bottom.set_position(('outward', 10))
            if bm['subplot']>0:
                pass
            #    ax3.get_yaxis().set_visible(False)
            #    ax3.spines.left.set_visible(False)
            else:
                ax3.set_ylabel('Error (10 substeps)')
            #    ax3.spines.left.set_position(('outward', 10))
            ax3_dict[bm['subplot']] = ax3
            ax3.annotate(letter[3*ncol+bm['subplot']], (-0.12,1.1), xycoords='axes fraction', fontsize='x-large', fontweight='bold')
        else:
            ax3 = ax3_dict[bm['subplot']]
        ax3.plot(data_df['t'], err10, alpha=0.6, lw=2, label=f" RMSE = {RMS(err10):.2e}")
        ax3.legend(frameon=False, loc='lower left')
        figcount+=1
if makefigure:
    fig.tight_layout()
    fig.savefig('test_steady.pdf')
# %%
ssm = SSM_frame(data_df, data_df['Biased young (Beta) benchmark C'], solute_parameters,\
        sas_specs, M = 1, dT=10)
ssm.run_sMC(0.1,0.1,1)
# %%
