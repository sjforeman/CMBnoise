#!/usr/bin/env python
"""Plot sample noise curves from CMBInoise module.
"""
import numpy as np
from scipy.constants import arcmin

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

import CMBnoise as cn

# Filenames for plot ouptut
out_uK_file = "sample_plots/uK_plot.pdf"
out_Jy_file = "sample_plots/Jy_plot.pdf"

# Initialize each experiment
planck = cn.Planck()
pico_base = cn.PICO_baseline()
pico_cbe = cn.PICO_CBE()
litebird = cn.LiteBIRD()

# Import sample CMB TT power spectrum, and convert a copy from D_l to C_l
dl_T = np.loadtxt("sample_plots/planck_lensing_scalCls.dat")
cl_T = dl_T.copy()
for i in range(1,4):
    cl_T[:,i] *= 2*np.pi/cl_T[:,0]/(cl_T[:,0]+1.)

# Define ells for plot
ell_for_plot = np.arange(1,5000)

# Plot CMB and noise power spectra as D_l in units of uK^2
plt.loglog(dl_T[:,0],dl_T[:,1],label="CMB temperature")
plt.loglog(ell_for_plot,planck.Dl_N_uK2(143,ell_for_plot),label="Planck 143GHz")
plt.loglog(ell_for_plot,pico_base.Dl_N_uK2(155,ell_for_plot),label="PICO baseline 155GHz")
plt.loglog(ell_for_plot,pico_cbe.Dl_N_uK2(155,ell_for_plot),label="PICO CBE 155GHz")
plt.loglog(ell_for_plot,litebird.Dl_N_uK2(140,ell_for_plot),label="LiteBIRD 140GHz")
plt.ylim(1e-8,1e4)
plt.xlim(1,1e4)
plt.ylabel(r"$\mathcal{D}_\ell^{\rm N}\;[\mu{\rm K}^2]$",size=14)
plt.xlabel(r"$\ell$",size=14)
plt.legend()
plt.grid()
# plt.show()
plt.savefig(out_uK_file)
plt.close()

# Plot CMB and noise power spectra as C_l in units of Jy^2 sr^-1.
# Convert input "signal" power spectrum using uK-to-Jypersr conversion factor.
plt.loglog(cl_T[:,0],cn.uKCMB_to_Jypersr(143,1)**2 * cl_T[:,1],label="CMB intensity at 143GHz")
plt.loglog(ell_for_plot,planck.Cl_N_Jy2persr(143,ell_for_plot),label="Planck 143GHz")
plt.loglog(ell_for_plot,pico_base.Cl_N_Jy2persr(155,ell_for_plot),label="PICO baseline 155GHz")
plt.loglog(ell_for_plot,pico_cbe.Cl_N_Jy2persr(155,ell_for_plot),label="PICO CBE 155GHz")
plt.loglog(ell_for_plot,litebird.Cl_N_Jy2persr(140,ell_for_plot),label="LiteBIRD 140GHz")
plt.ylim(1e-2,1e8)
plt.xlim(1,1e4)
plt.ylabel(r"$C_\ell^{\rm N}\;[{\rm Jy}^2\, {\rm sr}^{-1}]$",size=14)
plt.xlabel(r"$\ell$",size=14)
plt.legend()
plt.grid()
# plt.show()
plt.savefig(out_Jy_file)
plt.close()
