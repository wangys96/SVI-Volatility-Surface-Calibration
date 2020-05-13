import numpy as np
import scipy as sp
import scipy.optimize as opt
from scipy.optimize import minimize
import time
import pandas as pd
import matplotlib.pyplot as plt

def svi_2steps(iv,x,init_msigma,maxiter=10,exit=1e-12,verbose=True):
    opt_rmse=1

    def svi_quasi(y,a,d,c):
        return a+d*y+c*np.sqrt(np.square(y)+1)

    def svi_quasi_rmse(iv,y,a,d,c):
        return np.sqrt(np.mean(np.square(svi_quasi(y,a,d,c)-iv)))
    
    # 计算a,d,c
    def calc_adc(iv,x,_m,_sigma):
        y = (x-_m)/_sigma
        s = max(_sigma,1e-6)
        bnd = ((0,0,0),(max(iv.max(),1e-6),2*np.sqrt(2)*s,2*np.sqrt(2)*s))
        z = np.sqrt(np.square(y)+1)
        
        # 此处等价于坐标轴旋转45°，这样写运行更快
        A = np.column_stack([np.ones(len(iv)),np.sqrt(2)/2*(y+z),np.sqrt(2)/2*(-y+z)])
        
        a,d,c = opt.lsq_linear(A,iv,bnd,tol=1e-12,verbose=False).x
        return a,np.sqrt(2)/2*(d-c),np.sqrt(2)/2*(d+c)
    

    def opt_msigma(msigma):
        _m,_sigma = msigma
        _y = (x-_m)/_sigma 
        _a,_d,_c = calc_adc(iv,x,_m,_sigma)
        return np.sum(np.square(_a+_d*_y+_c*np.sqrt(np.square(_y)+1)-iv))

    for i in range(1,maxiter+1):
        #a_star,d_star,c_star = calc_adc(iv,x,init_msigma)       
        m_star,sigma_star = opt.minimize(opt_msigma,
                                         init_msigma,
                                         method='Nelder-Mead',
                                         bounds=((2*min(x.min(),0), 2*max(x.max(),0)),(1e-6,1)),
                                         tol=1e-12).x
        
        a_star,d_star,c_star = calc_adc(iv,x,m_star,sigma_star)
        opt_rmse1 = svi_quasi_rmse(iv,(x-m_star)/sigma_star,a_star,d_star,c_star)
        if verbose:
            print(f"round {i}: RMSE={opt_rmse1} para={[a_star,d_star,c_star,m_star,sigma_star]}     ")
        if i>1 and opt_rmse-opt_rmse1<exit:
            break
        opt_rmse = opt_rmse1
        init_msigma = [m_star,sigma_star]
        
    result = np.array([a_star,d_star,c_star,m_star,sigma_star,opt_rmse1])
    if verbose:
        print(f"\nfinished. params = {result[:5].round(10)}")
    return result

def quasi2raw(a,d,c,m,sigma):
    return a,c/sigma,d/c,m,sigma

def svi_raw(x,a,b,rho,m,sigma):
    centered = x-m
    return a+b*(rho*centered+np.sqrt(np.square(centered)+np.square(sigma)))

def svi_quasi(x,a,d,c,m,sigma):
    y = (x-m)/sigma
    return a+d*y+c*np.sqrt(np.square(y)+1)

class svi_quasi_model:
    def __init__(self,a,d,c,m,sigma):
        self.a = a
        self.d = d
        self.c = c
        self.m = m
        self.sigma = sigma
    def __call__(self,x):
        return svi_quasi(x,self.a,self.d,self.c,self.m,self.sigma)

def plot_tv(logm,tv,model,extend=0.1,n=100):
    scale = (max(logm)-min(logm))*extend
    lmax,lmin = min(logm)-scale,max(logm)+scale
    lin = np.linspace(lmin,lmax,n)
    plt.figure(figsize=(8, 4))
    plt.plot(logm, tv, '+', markersize=12)
    plt.plot(lin,model(lin),linewidth=1)
    plt.title("Total Variance Curve")
    plt.xlabel("Log-Moneyness", fontsize=12)
    plt.legend()
    
def plot_iv(logm,tv,t,model,extend=0.1,n=100):
    scale = (max(logm)-min(logm))*extend
    lmax,lmin = min(logm)-scale,max(logm)+scale
    lin = np.linspace(lmin,lmax,n)
    plt.figure(figsize=(8, 4))
    plt.plot(np.exp(logm), np.sqrt(tv/t), '+', markersize=12)
    plt.plot(np.exp(lin),np.sqrt(model(lin)/t),linewidth=1)
    plt.title("Implied Volatility Curve")
    plt.xlabel("Moneyness", fontsize=12)
    plt.legend()
    