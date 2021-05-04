#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 23 00:25:00 2020

@author: yaakov
"""

import numpy as np
from Bio import SeqIO
import pandas as pd
from scipy.spatial.distance import pdist
from scipy.io import loadmat
import matplotlib.pyplot as plt
import itertools as it
from scipy.spatial.distance import pdist,squareform
import C_MonteCarlo
from random import randint

#### Pre-Inference Functions ***************************************

def load_fasta(file):
    # input fasta file containing the MSA
    # return MSA_ohr - one hot representation of the MSA
    # and MSA the regualr representation of the MSA
    
    # Amino acid code and error letters
    code = "-ACDEFGHIKLMNPQRSTVWY"
    q=len(code);
    AA_to_num=dict([(code[i],i) for i in range(len(code))])
    errs = "BJOUXZabcdefghijklmonpqrstuvwxyz"
    AA_to_num.update(dict([(errs[i],-1) for i in range(len(errs))]))
    
    # Parse sequence contacts of fasta into MSA
    MSA=np.array([])
    for record in SeqIO.parse(file, "fasta"):
        seq=np.array([[AA_to_num[record.seq[i]] for i in range(len(record.seq))]])
        if MSA.shape[0]==0:
            MSA=seq;
        else:
            MSA=np.append(MSA,seq,axis=0)
            
    # Remove all errornous sequences (contain '-1')
    if np.min(MSA)<0:
        MSA=np.delete(MSA,(np.sum(MSA==-1,axis=1)).nonzero()[0][0],axis=0)
    
    return MSA

def read_mat_model(file):
    # input filename from BM-DCA algorithm in a specific format
    # return non-one-hot representation of h and J full matrices.
    x=loadmat(file)
    J=x['J'];
    h=x['h'];
    align=x['align']-1;
    J=J.transpose(2,3,0,1)
    h=h.transpose(1,0)
    return h,J,align




#### Functions for Inference process **************************************************
## Functions inside the minimizer ******

def ParseSMinimizerOptions(options):
    if 'maxIter' not in options.keys():
        options['maxITer']=300;
    if 'm' not in options.keys():
        options['m']=10;        
    if 'TolX' not in options.keys():
        options['TolX']=10**-5;
    if 'skip_log' not in options.keys():
        options['skip_log']=1;    
    return options
        
def sMinimizer(fun,x0,options):
    # parse options
    options=ParseSMinimizerOptions(options)
    
    #  Initalizing variables
    x=x0
    g=fun(x)
    h=-g;
    
    s=np.zeros((x.shape[0],options['m']))
    y=np.zeros((x.shape[0],options['m']))
    ys=np.zeros((options['m']))
    diag=1
    gtd=np.dot(-g,h)
    wt=x.reshape(-1,1);
    ind=np.zeros(options['m'])-1;ind[0]=0;
    skipping=0
    output={'skipping':0,'Xchange':np.zeros(0),'Grad':np.zeros(0),'gtd':np.zeros(0),'wt':x.reshape(1,-1)}
    print('{0:8}  {1:12}  {2:12}  {3:12}'.format('   Iter', '   Xchange', '   Gradient', '   Gtd'))    # minimization iteration
    for i in range(options['maxIter']):
        if i==0:
            t=1/np.sum(g**2)**0.5;
        else:
            t=1;
        #g_old=g; h_old=h; gtd_old=gtd;
        
        #  Here is the main part: iterating the minimization
        x,h,g,gtd,s,y,ys,diag,ind,output['skipping']=AdvanceSearch(x,t,h,g,fun,gtd,s,y,ys,diag,ind,output['skipping'],options)
        
        #  And logging the progress
        print('{0:8d}  {1:12f}  {2:12f}  {3:12f}'.format(i, max(abs(t*h)), max(abs(g)), gtd))
        output['Xchange']=np.append(output['Xchange'],max(abs(t*h)))
        output['Grad']=np.append(output['Grad'],max(abs(g)))
        output['gtd']=np.append(output['gtd'],gtd)
        if i%options['skip_log']==0:
            output['wt']=np.append(output['wt'],x.reshape(1,-1),axis=0)
        
    return x,output

def AdvanceSearch(x,t,h,g,fun,gtd,s,y,ys,diag,ind,skipping,options):
    flag=1;
    while flag>0:
        x_out=x+t*h;
        # calculate the gradient
        g_out=fun(x_out)
        # and use it to update the h=hessian*gradient
        h_out,s_out,y_out,ys_out,diag_out,ind_out,skipping_out=\
            UpdateHessian(g_out,g_out-g,x_out-x,s,y,ys,diag,ind,skipping,options)
        gtd_out=np.dot(-g_out,h_out)
        # sometimes this can be an irrelevant step, retry it if it strays too far
        if gtd_out<50*gtd:
            flag=0
    return x_out,h_out,g_out,gtd_out,s_out,y_out,ys_out,diag_out,ind_out,skipping_out

def UpdateHessian(g,y,s,s_out,y_out,ys_out,diag,ind,skipping,options):
    ys=np.dot(y,s)
    # if this is a meaningful step
    if ys>10**(-10):
        y_out[:,ind==max(ind)]=y.reshape(-1,1)
        s_out[:,ind==max(ind)]=s.reshape(-1,1)
        ys_out[ind==max(ind)]=ys
        diag=ys/np.dot(y,y)
    # or if not meaningful, the update will be skipped
    else:
        skipping=skipping+1
    
    # here the hessian*gradient is calculated
    h_out=-g
    order=np.argsort(ind[ind>-1])
    alpha=np.zeros(order.shape[0]);
    beta=np.zeros(order.shape[0]);
    for i in order[::-1]:
        alpha[i]=np.dot(s_out[:,i],h_out)/ys_out[i]
        h_out=h_out-alpha[i]*y_out[:,i];
    h_out=diag*h_out
    for i in order:
        beta[i]=np.dot(y_out[:,i],h_out)/ys_out[i]
        h_out=h_out+s_out[:,i]*(alpha[i]-beta[i])
        
    # update the memory steps (indices) only if it is meaningful
    if ys>10**(-10):
        if ind[options['m']-1]==-1:
            ind=ind
            ind[(ind==max(ind)).nonzero()[0]+1]=max(ind)+1;
        else:
            ind=np.roll(ind,1)    
    return h_out,s_out,y_out,ys_out,diag,ind,skipping



## Functions outside the minimizer ******

def SBM(align,lamJ,lamh,Jinit,hinit,options):
    # Evaluate the goal statistics
    W,N_eff=CalcWeights(align,options['theta']);
    fi,fij=CalcStatsWeighted(options['q'],align,W/N_eff);
    
    # initialize start position
    w0=Wj(Jinit,hinit)
    # define function to be minimized
    f=lambda x: GradLogLike(x,lamJ,lamh,fi,fij,options)
    
    # minimize the function f from starting point w0
    w,output=sMinimizer(f,w0,options)
    J,h=Jw(w,options['q'])
    
    return J,h,output

def GradLogLike(w,lambdaJ,lambdah,fi,fij,options):
    [J,h]=Jw(w,options['q']);

    align_mod=CreateAlign(options['N'],Wj(J,h),options['L'],options['q'],options['delta_t']);
    
    # now gradient of it
    #if options['']sim_weight==1
    #    [p,N_eff]=CalcWeights(align_mod,options.theta);
    #else        
    p=np.zeros(options['N'])+1/options['N'];
    #end
    #model MSA stats
    fi_tot,fij_tot=CalcStatsWeighted(options['q'],align_mod,p);
    
    #compared to data MSA stats - that is the update rule
    gradh=fi_tot-fi+2*lambdah*h;
    gradJ=(fij_tot-fij+2*lambdaJ*J);
    grad=Wj(gradJ,gradh);
   
    return grad

def CalcWeights(align,theta):
    W = 1/(np.sum(squareform(pdist(align, 'hamming'))<theta,axis=0));
    N_eff=sum(W)
    return W,N_eff

def CalcStatsWeighted(q,MSA,p):
    # input MSA in amino acid form
    # output the unweighted freqs fi, co-occurnces fij and correlations Cij=fij-fi*fj
    L=MSA.shape[1];    
    fi=np.zeros([L,q])
    x=np.array([i for i in range(L)])
    for m in range(MSA.shape[0]):
        fi[x[:],MSA[m,x[:]]]+=p[m];
    fi=fi;  
    
    
    fij=np.zeros([L,L,q,q])
    x=np.array([[i,j] for i,j in it.product(range(L),range(L))])

            
    for m in range(MSA.shape[0]):
        fij[x[:,0],x[:,1],MSA[m,x[:,0]],MSA[m,x[:,1]]]+=p[m];
    fij=fij;    
    
    #Cij=fij-(fi.reshape([L,1,q,1])*fi.reshape([1,L,1,q]))
    #for i in range(L):
    #    Cij[i,i,:,:]=0;
    return fi,fij #,Cij


def Wj(J,h):
    # translate J (L*L*q*q) and h (L*q)  into a vector of L*q + (L-1)*L*q*q/2 independent variables.
    # this removes redundent variables that we dont need to infer 
    q=J.shape[2]
    L=J.shape[0]
    W=np.zeros(int((q*L+q*q*L*(L-1)/2),))
    x=np.array([[i,j] for i,j in it.combinations(range(L),2)])
    for a in range(q):
        for b in range(q):   
            W[(q**2*((x[:,0])*(2*L-x[:,0]-1)/2+(x[:,1]-x[:,0]-1))+(a)*q+b).astype(int)]=J[x[:,0],x[:,1],a,b]
    x=np.array(range(L))
    for a in range(q):
        W[(q**2*L*(L-1)/2+q*x[:]+a).astype(int)]=h[x[:],a]
    return W

def Jw(W,q):
    L=int(((q*q-2*q)+((2*q-q*q)**2+8*W.shape[0]*q*q)**(1/2))/2/q/q);
    J=np.zeros((L,L,q,q));
    h=np.zeros((L,q));
    x=np.array([[i,j] for i,j in it.combinations(range(L),2)])
    for a in range(q):
        for b in range(q):
            J[x[:,0],x[:,1],a,b]=W[(q**2*((x[:,0])*(2*L-x[:,0]-1)/2+(x[:,1]-x[:,0]-1))+(a)*q+b).astype(int)];
            J[x[:,1],x[:,0],b,a]=J[x[:,0],x[:,1],a,b];
    x=np.array(range(L))
    for a in range(q):
        h[x[:],a]=W[(q**2*L*(L-1)/2+q*x[:]+a).astype(int)];

    return J,h


def CreateAlign(N,w,L,q,delta_t):
    C_max_int=2147483647;
    seed=randint(1, C_max_int)
    MSA=np.array(C_MonteCarlo.MC(np.array([x for x in w]),N,L,q,delta_t,seed)).reshape(-1,L).astype(int);
    return MSA




####  Post-Inference functions **********************************************************
def Frob(J):
    return np.sqrt(np.sum(np.sum(J**2,3),2))   

def DCA_energy(seqs,h,J):
    # given an MSA (must be matrix) and h,J model parameters
    # output the energy of per sequence in the MSA
    if len(seqs.shape)==2:
        L=seqs.shape[1]
        N=seqs.shape[0]
    elif len(seqs.shape)==1:
        L=seqs.shape[0]
        N=1
        seqs=seqs.reshape((1,L))
        
    energy=np.sum(np.array([h[i,seqs[:,i]] for i in range(L)]),axis=0)
    energy=energy+(np.sum(np.array([[J[i,j,seqs[:,i],seqs[:,j]] for j in range(L)] for i in range(L)]),axis=(0,1))/2)

    return energy
