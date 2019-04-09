import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from numpy import random as rand
from scipy import constants as const
from matplotlib import cm
import time as time
%matplotlib notebook

n = 10000
t = 2500
D = 1
h = 1
d = 1
k_B = const.Boltzmann
temp = 310
beta = 1/(k_B*temp)
k = 1/beta
binv = 1/beta
L = 50
V0_Na = 50*binv
V0_K  = 50*binv
elem_C = const.elementary_charge
Cc = 0.07


def phi(x,t,D,x0):
    return np.exp( -( (x-x0)**2 / (4*D*t) ) ) / np.sqrt(4*np.pi*D*t)

def norm(mu, sig, xs):
    return np.exp(-(xs-mu)**2 / sig**2) / np.sqrt(2*np.pi*sig**2)

def rwalk_nopot(n_particles, t_steps):
    pos = np.zeros(n_particles)
    for t in range(t_steps):
        new_steps = np.random.randint(0,2,size=n_particles)
        new_steps *= 2
        new_steps -= 1
        pos += new_steps*h
    return pos

def p_ratio(x, V):
    rpot = np.zeros(x.shape)
    lpot = np.zeros(x.shape)
    for it, elem in enumerate(x):
        rpot[it] = V(elem + h)
        lpot[it] = V(elem - h)
        
    return np.exp(-beta*(lpot-rpot))

def p_stepright(x, V):
    return 1/(1+p_ratio(x,V))

def steps(x, V):
    ran = rand.rand(len(x))
    return np.where(ran < p_stepright(x, V),1, -1)*h

def rwalk_pot(x0, n_particles, t_steps, h, V):
    pos = x0
    poses = np.zeros([t_steps, n_particles])
    for t in range(t_steps):
        poses[t] = pos
        new_steps = steps(pos, V)
        pos += new_steps
    return poses

def V_51(k, x):
    return k*x

def V_52(k, x):
    if np.abs(x)<3*h:
        return k
    return 0

def V_53(k, x):
    if x < -3*h:
        return -k
    elif x > 3*h:
        return k
    return k*(-1+2*(x+3*h)/(6*h))

def V_elec(x, V_val):
    if x < -h:
        return V_val*elem_C
    elif x > h:
        return 0
    else:
        return (1-(x+h)/2) * V_val * elem_C

def Vmem_N(x, V_val):
    if abs(x) > L / 2:
        return np.inf
    if -h <= x <= h:
        return V_val
    else:
        return 0
    
def Vmem_K(x, V_val):
    if abs(x) > L / 2:
        return np.inf
    elif -h <= x <= h:
        return V_val
    else:
        return 0

def oppg3():
    pos = rwalk_nopot(n, t)
    unique, count = np.unique(pos, return_counts=True)

    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    #ax1.plot(x_randwalk1)
    ax1.bar(unique, count/n, width=2*h, align="center", color=cm.viridis(0.1))
    xs = np.linspace(-t*h,t*h,2*t)
    #ax1.plot(xs, phi(xs,t,D,0), color=cm.viridis(0.8))
    ax1.plot(xs, norm(0,np.sqrt(2*D*t),xs), color=cm.viridis(0.8))
    plt.show()

    
def oppg5_1():
    def V(x): return V_51(k,x) #k*x
    pos = rwalk_pot(np.zeros(n), n, t, h, V)
    unique, count = np.unique(pos, return_counts=True)
    
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    #ax1.plot(x_randwalk1)
    ax1.bar(unique, count/n, width=2*h, align="center", color=cm.viridis(0.1))
    plt.show()


def oppg5_2(ts=t):
    def V(x): return V_52(k,x)
    fig = plt.figure()
    
    if type(ts)==list:
        tlen = len(ts)
    else:
        tlen = t
    
    tmax = max(ts)

    if tlen==1:
        axes = [fig.subplots(tlen,1,sharex=True)]
    else:
        axes = fig.subplots(tlen,1,sharex=True)

    pos = np.zeros(n)
    pos[0::2] = +12*h
    pos[1::2] = -12*h
    poses = rwalk_pot(pos, n, tmax, h, V)
        
    for it, t in enumerate(ts):
        unique, count = np.unique(poses[t-1], return_counts=True)
        axes[it].axvline(-3*h, color=cm.viridis(0.1))
        axes[it].axvline(3*h, color=cm.viridis(0.1))
        axes[it].bar(unique, count/n, width=2*h, align="center", color=cm.viridis(0.7))
    
    plt.tight_layout()
    plt.show()


def oppg5_3(ts=t):
    def V(x): return V_53(k,x)
    fig = plt.figure()
    
    if type(ts)==list:
        tlen = len(ts)
    else:
        tlen = t
    
    tmax = max(ts)
    
    if tlen==1:
        axes = [fig.subplots(tlen,1,sharex=True)]
    else:
        axes = fig.subplots(tlen,1,sharex=True)
    
    pos = np.zeros(n)
    pos[0::2] = +12*h
    pos[1::2] = -12*h
    poses = rwalk_pot(pos, n, tmax, h, V)
        
    for it, t in enumerate(ts):
        unique, count = np.unique(poses[t-1], return_counts=True)
        axes[it].axvline(-3*h, color=cm.viridis(0.1))
        axes[it].axvline(3*h, color=cm.viridis(0.1))
        axes[it].bar(unique, count/n, width=2*h, align="center", color=cm.viridis(0.7))
    
    plt.tight_layout()
    plt.show()


def NaK_pot(): #NB! Edita
    mult = 5
    # start parameters: T = #steps
    plot_ts = [5, 50, 150, 1000]
    # N partiklar i starttilstand
    Na_outside, Na_inside = 1450*mult, 50*mult
    N_Na = Na_outside + Na_inside
    Na = np.array([L/4 for _ in range(Na_outside)] \
                + [-L/4 for _ in range(Na_inside)])

    K_outside, K_inside = 50*mult, 1400*mult
    N_K = K_outside + K_inside
    K  = np.array([L/4 for _ in range(K_outside)] \
                + [-L/4 for _ in range(K_inside)])

    phi_K, phi_Na = dict(),  dict()

    Vs = np.zeros(t + 1)
    
    for t_it in range(t + 1):
        
        # count the particles inside
        count_K  = len(K[np.where(K < -h)])
        count_Na = len(Na[np.where(Na < -h)])
        # convert to Moles
        charge_conc = (count_K + count_Na) * 0.1 / mult # mM
        charge_delta = (charge_conc - 150)
        
        voltage = 1e-3 * charge_delta / Cc
        Vs[t_it] = voltage
        
        if voltage > -70e-3:
            def V_K(x):  return V_elec(x, voltage) + Vmem_K(x, V0_K)
            def V_Na(x): return V_elec(x, voltage) + Vmem_N(x, binv)
        elif voltage < 30e-3:
            def V_K(x):  return V_elec(x, voltage) + Vmem_K(x, binv)
            def V_Na(x): return V_elec(x, voltage) + Vmem_N(x, V0_Na)

        K  += steps(K, V_K)
        Na += steps(Na, V_Na)
        
        if t_it in plot_ts:
            phi_K[t_it]  = np.copy(K)
            phi_Na[t_it] = np.copy(Na)
    return Vs

def get_average_potential(n_samples):
    Vis = np.zeros((10,t+1))
    for i in range(10):
        Vis[i,:],_,_ = NaK_pot()
        
    return np.mean(Vis, axis=0)

def plot_V_elec():
    xs = np.linspace(-3*h,3*h,1000)
    epotfig = plt.figure()
    vs = np.zeros(len(xs))
    for i in range(len(xs)):
        vs[i] = V_elec(xs[i], -70e-3) + Vmem_N(xs[i], V0_Na)
    plt.plot(xs, vs)
    plt.show()

def avg_subslice(A, n):
    D = (len(A)//n)*n
    return np.mean(A[:D].reshape(-1, n), axis=1)
