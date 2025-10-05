#!/usr/bin/env python

from __future__ import division
import matplotlib
import matplotlib.pyplot as plt

import numpy as np, pickle
import math, sys, os, glob, h5py, json
from astropy import units as u
from sklearn.neighbors import KernelDensity

import pint
from pint import toa
from pint import models
from pint.residuals import Residuals
from pint.simulation import make_fake_toas_fromMJDs
pint.logging.setup(sink=sys.stderr, level="ERROR", usecolors=True)

import pta_replicator
from pta_replicator import simulate
from pta_replicator import white_noise
from pta_replicator import red_noise

#from plot_utils import make_residual_plot
#plotting function
def make_residual_plot(psr, save=False, simdir='pint_sims1/'):

    # switch to rainbow colormap grouped by telescopes
    colors = {
                "CHIME": "#FFA733",
                "327_ASP": "#BE0119",
                "327_PUPPI": "#BE0119",
                "430_ASP": "#FD9927",
                "430_PUPPI": "#FD9927",
                "L-wide_ASP": "#BDB6F6",
                "L-wide_PUPPI": "#BDB6F6",
                "Rcvr1_2_GASP": "#79A3E2",
                "Rcvr1_2_GUPPI": "#79A3E2",
                "Rcvr1_2_VEGAS": "#79A3E2",
                "Rcvr_800_GASP": "#8DD883",
                "Rcvr_800_GUPPI": "#8DD883",
                "Rcvr_800_VEGAS": "#8DD883",
                "S-wide_ASP": "#C4457A",
                "S-wide_PUPPI": "#C4457A",
                "1.5GHz_YUPPI": "#EBADCB",
                "3GHz_YUPPI": "#E79CC1",
                "6GHz_YUPPI": "#DB6BA1",
            }

    fig, axs = plt.subplots(1, 1)

    flags = list(np.unique(psr.toas['f']))

    for f in flags:

        idx = (psr.toas['f'] == f)
    
        axs.errorbar(psr.toas[idx].get_mjds(), psr.residuals.calc_time_resids()[idx], 
                     yerr=psr.toas[idx].get_errors(), marker='.', ls='', alpha=0.5, 
                     label=f, color=colors[f])

    plt.title('{0} -- {1} TOAs'.format(psr.name, len(psr.toas)))
    plt.xlabel('MJD')
    plt.ylabel('Residual [s]')
    plt.legend()
    plt.tight_layout()

    if save:
        plt.savefig(simdir + '/{0}.png'.format(psr.name))


seed_efac_equad = 10660
seed_red = 19870
seed_gwb = 16666


#datadir = '/Users/vigeland/Documents/Research/NANOGrav/nanograv_data/NG20/Data/ng20_v1p0_dmx/'

#my paths:
mypardir = '/Users/ashokan/osu/sarahspta/ng20/results/'
mytimdir = '/Users/ashokan/osu/sarahspta/NG20_prelim_v1p0_excised_toas/'
#myparfile = mypardir + 'J1909-3744_PINT_20250319.nb.par'
#mytimfile = mytimdir + 'J1909-3744_PINT_20250411.nb.tim'

# # parfiles = list(np.genfromtxt(datadir + 'parfile_names.txt', dtype='str'))
# #parfiles = list(np.genfromtxt(mypardir + 'parfile_names.txt', dtype='str'))
# parfiles = os.listdir(mypardir)
# timfiles = os.listdir(mytimdir)
# print(len(parfiles), len(timfiles))

# list all files in directories
parfiles_all = os.listdir(mypardir)
timfiles_all = os.listdir(mytimdir)

# filter only the ones ending with nb.par and nb.tim
parfiles = sorted([f for f in parfiles_all if f.endswith('nb.par')])
timfiles = sorted([f for f in timfiles_all if f.endswith('nb.tim')])
print(len(parfiles), len(timfiles))

#rn_dict_file = '/Users/vigeland/Documents/Research/NANOGrav/nanograv_20yr_gwb/20yr_noisedict_rn+curn-bpl.json'
rn_dict_file = '/Users/ashokan/osu/sarahspta/20yr_noisedict_rn+curn-bpl.json'

with open(rn_dict_file, 'r') as f:
    rn_dict = json.load(f)


print('Making simulation with noise only...')
 
psrs = []

for ii in range(len(parfiles)):
    print('Working on {0}...'.format(parfiles[ii]))
    print(parfiles[ii], timfiles[ii])
    psr = simulate.load_pulsar(parfiles[ii], timfiles[ii], ephem='DE440')

    # remove red noise from the model (we will add it back later)
    if 'PLRedNoise' in psr.model.components.keys():
        psr.model.remove_component('PLRedNoise')

    # remove Shapiro delay parameters if they are in the binary model
    if 'BinaryDD' in psr.model.components.keys():
        if 'SINI' in psr.model.components['BinaryDD'].params:
            psr.model.components['BinaryDD'].remove_param('SINI')
    
        if 'M2' in psr.model.components['BinaryDD'].params:
            psr.model.components['BinaryDD'].remove_param('M2')

    elif 'BinaryELL1' in psr.model.components.keys():

        if 'SINI' in psr.model.components['BinaryELL1'].params:
            psr.model.components['BinaryELL1'].remove_param('SINI')
    
        if 'M2' in psr.model.components['BinaryELL1'].params:
            psr.model.components['BinaryELL1'].remove_param('M2')
            
    psr.generate_daily_avg_toas(ideal=True)
    
    white_noise.add_measurement_noise(psr, efac=1, seed = seed_efac_equad + ii, FACTOR=1.5420348502877987)
    
    for _ in range(3):
        psr.fit(fitter='downhill')    
    
    red_noise.add_red_noise(psr, log10_amplitude = rn_dict[psr.name + '_red_noise_log10_A'], 
                            spectral_index = rn_dict[psr.name + '_red_noise_gamma'], 
                            components = 30, seed = seed_red + ii)
    
    for _ in range(3):
        psr.fit(fitter='downhill')
        
    simdir = '../data/NG20_irn_with_factor/'
    if not os.path.isdir(simdir):
        os.mkdir(simdir)

    make_residual_plot(psr, save=True, simdir=simdir)
    
    psr.write_partim(simdir + psr.name + '.par', simdir + psr.name + '.tim', tempo2=False)
    
    psrs.append(psr)

# print('Making simulation with GWB...')
# red_noise.add_gwb(psrs, log10_amplitude = rn_dict['gw_log10_A'], spectral_index = rn_dict['gw_gamma'], 
#                   seed = seed_gwb)

# simdir = '../data/NG20_gwb/'
# if not os.path.isdir(simdir):
#     os.mkdir(simdir)

# for psr in psrs:
    
#     print('Refitting residuals for {0}...'.format(psr.name))
    
#     for _ in range(3):
#         try:
#             psr.fit(fitter='downhill')
#         except:
#             print('Downhill fitter didn\'t work for {0}'.format(psr.name))
#             print('Trying to fit with gls fitter...'.format(psr.name))
            
#             try:
#                 psr.fit(fitter='gls')
#             except:
#                 print('gls fitter didn\'t work either!')

#     make_residual_plot(psr, save=True, simdir=simdir)
    
#     psr.write_partim(simdir + psr.name + '.par', simdir + psr.name + '.tim', tempo2=False)

print('Done!')