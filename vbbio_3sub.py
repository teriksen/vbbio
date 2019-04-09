import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.backends.backend_pdf import PdfPages
from numpy import random as rand
from scipy import constants as const
from scipy import stats as st
from scipy.integrate import simps
from matplotlib import cm
import time as time
#%matplotlib inline

fontsize = 20
newparams = {
    'axes.titlesize': fontsize,
    'axes.labelsize': fontsize,
    'lines.linewidth': 2, 
    'lines.markersize': 7,
    'text.usetex': True,
    'font.family': "serif",
    "pgf.texsystem": "pdflatex",
    "pgf.rcfonts": False,
    'figure.figsize': (16,8), 
    'ytick.labelsize': fontsize,
    'xtick.labelsize': fontsize,
    'legend.fontsize': fontsize,
    'legend.handlelength': 1.5,
    'xtick.major.pad': 8,
    'ytick.major.pad': 8,
}

plt.style.use('classic')
plt.rcParams.update(newparams)

n = 1000
t = 100
h = 1
d = 1
k_B = const.Boltzmann
temp = 310
beta = 1/(k_B*temp)
k = 1/beta
binv = 1/beta
L = 50
V0_Na = binv#*50
V0_K  = binv#*50
elem_C = const.elementary_charge
Cc = 0.07

def phi(xs,t, D):
    return 1/(np.sqrt(4*np.pi*D*t))*np.exp(-((xs-mu)**2)/(4*D*t))

def rwalk_nopot(n_particles, t_steps):
    pos = np.zeros(n_particles)
    for t in range(t_steps):
        new_steps = np.random.randint(0,2,size=n_particles)
        new_steps *= 2
        new_steps -= 1
        pos += new_steps*h
    return pos

def steps(x, V):
    ran = rand.rand(len(x))
    return np.where(ran < p_stepright(x, V),1, -1)*h

def norm(mu, sig, xs):
    return np.exp(-(xs-mu)**2 / sig**2) / np.sqrt(2*np.pi*sig**2)

def oppg3():
    #vbbio_3 = PdfPages("vbbio_3.pdf")
    fig = plt.figure(1, figsize=(16,12))
    big = plt.subplot(1,1,1, frameon = False, title="Random walk in zero potential")
    plt.tick_params(labelsize=1, bottom = False, 
                    left = False, top = False,
                    right = False)
    xs = np.linspace(-40,40,2*t)
    for i in range(3):
        pos = rwalk_nopot(n, t)
        unique, count = np.unique(pos, return_counts=True)
        D = np.var(pos)/(2*t) #5/(2*t)
        #print(D)
        ax = fig.add_subplot(3,1,i+1)
        #ax.plot(x_randwalk1)
        ax.bar(unique, count/n, width=2*h, align="center", color=cm.magma(0.25*(i+1)+0.2), label="Random walk {}".format(i+1))
        #ax.plot(xs, phi(xs,t,D,0), color=cm.magma(0.8))
        mu, sig = st.norm.fit(pos)
        ps = st.norm.pdf(xs, mu, sig)
        ax.plot(xs, 2*ps, color=cm.magma(0.01), linewidth=2.5, label="Theoretical distribution")
        ax.set_xlim(-40,40)
        #ax.plot(xs, phi(xs,t, D), color=cm.magma(0.01), linewidth=2.5)
        ax.legend()
    big.set_ylabel("Probability $P(x)$", labelpad=44)
    big.set_xlabel("Position $x$", labelpad=22)
    plt.tight_layout()
    plt.show()
    plt.savefig('vbbio_3.pdf', bbox_inches='tight',pad_inches = 0)
    #vbbio_3.savefig(1, bbox_inches='tight')
    #vbbio_3.close()
