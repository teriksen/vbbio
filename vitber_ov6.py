import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from numpy import linalg as LA
from matplotlib import cm
mpl.style.use('default')

def ImprovedEuler(t0,tend,y0,f,h0,tol=1e-6,alph=0.8):
    h = h0
    k = 0
    F1 = f(t0, y0)
    ts = np.array([t0,np.nan])
    ys = [F1, np.nan]
    fcalls = 1
    skips = 0
    while tend - ts[k] > 0:
        h = min(h, tend-ts[k])
        ts[k+1] = ts[k] + h
        F2 = f(ts[k+1], ys[k] + h*F1)
        ys[k+1] = y[k] + h/2*(F1+F2)
        est = LA.norm((h/2)*(F1-F2))
        if est < tol:
            ts = np.append(ts, np.nan)
            ys.append(np.nan)
            k += 1
            F1 = f(ts[k],ys[k])
            fcalls += 1
        else:
            skips += 1
        #h = min(alph*h*min(max(tol/est, 0.3), 2))
        h = alph*h*np.sqrt(tol/est)
        fcalls += 1
    stats = {"fcalls":fcalls,"steps":len(ts), "skips":skips}
    return ts,np.asarray(ys), stats
