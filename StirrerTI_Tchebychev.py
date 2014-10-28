# -*- coding: utf-8 -*-
"""
Created on Wed Oct 22 09:26:57 2014

@author: E68972
"""

from __future__ import division
from numpy import *
from numpy.random import *
from pylab import *
from pychebfun import *

c=299792458.

sim=load("Estirrer_MC1.npz")
freq=sim["freq"]
nf=len(freq)
E=sim["Et"]
B=sim["Bt"]
Er=real(E)
Ei=imag(E)
P=sqrt(sum((0.5*abs(cross(E,(B))))**2,axis=3))
Phase=angle(E)

M=120
for i in range(2,9):
    sim=load("Estirrer_MC"+"%s.npz" %(i))
    freq=sim["freq"]
    nf=len(freq)
    E=sim["Et"]
    B=sim["Bt"]
    Eri=real(E)
    Eii=imag(E)
    Er=concatenate((Er,Eri),axis=0)
    Ei=concatenate((Ei,Eii),axis=0)
    P=concatenate((P,sqrt(sum((0.5*abs(cross(E,(B))))**2,axis=3))),axis=0)
    Phase=concatenate((Phase,angle(E)),axis=0)

TcEx_r=zeros((M,nf))
TcEy_r=zeros((M,nf))
TcEz_r=zeros((M,nf))

TcEx_i=zeros((M,nf))
TcEy_i=zeros((M,nf))
TcEz_i=zeros((M,nf))

TcPh_x=zeros((M,nf))
TcPh_y=zeros((M,nf))
TcPh_z=zeros((M,nf))

TcdPh_x=zeros((M,nf))
TcdPh_y=zeros((M,nf))
TcdPh_z=zeros((M,nf))

TcP=zeros((M,nf))
TcmoyP=zeros((M,nf))


def maxorder(coeff,crit=0.01):
    U=find(abs(coeff)<crit)
    if len(U)==0:
        a=NaN
    if len(U)==1:
        a=U[0]
    if len(U)>1:
        a=U[1]
    return a
    



binsize=10
for k in range(M):
    print k
    for i in range(nf):
        TcEx_r[k,i]=maxorder(chebpolyfit(Er[k,i,:,0]))
        TcEy_r[k,i]=maxorder(chebpolyfit(Er[k,i,:,1]))
        TcEz_r[k,i]=maxorder(chebpolyfit(Er[k,i,:,2]))
        TcEx_i[k,i]=maxorder(chebpolyfit(Er[k,i,:,0]))
        TcEy_i[k,i]=maxorder(chebpolyfit(Er[k,i,:,1]))
        TcEz_i[k,i]=maxorder(chebpolyfit(Er[k,i,:,2]))
        TcPx=maxorder(chebpolyfit(Phase[k,i,:,0]))
        TcPy=maxorder(chebpolyfit(Phase[k,i,:,1]))
        TcPz=maxorder(chebpolyfit(Phase[k,i,:,2]))
        Px=Phase[k,i,:,0]/Phase[k,i,:,0].mean()
        Py=Phase[k,i,:,1]/Phase[k,i,:,1].mean()
        Pz=Phase[k,i,:,2]/Phase[k,i,:,2].mean()
        TcPh_x[k,i]=maxorder(chebpolyfit(Px))
        TcPh_y[k,i]=maxorder(chebpolyfit(Py))
        TcPh_z[k,i]=maxorder(chebpolyfit(Pz))
        TcdPh_x[k,i]=maxorder(chebpolyfit(diff(Px)))
        TcdPh_y[k,i]=maxorder(chebpolyfit(diff(Py)))
        TcdPh_z[k,i]=maxorder(chebpolyfit(diff(Pz)))
        TcP[k,i]=maxorder(chebpolyfit(P[k,i,:]/P[k,i,:].mean()))



figure(1)
semilogx(freq,TcEx_r.mean(axis=0),'b',label="$\Re(E_x)$")
semilogx(freq,TcEx_i.mean(axis=0),'--b',label="$\Im(E_x)$")
semilogx(freq,TcEy_r.mean(axis=0),'g',label="$\Re(E_y)$")
semilogx(freq,TcEy_i.mean(axis=0),'--g',label="$\Im(E_y)$")
semilogx(freq,TcEz_r.mean(axis=0),'r',label="$\Re(E_z)$")
semilogx(freq,TcEz_i.mean(axis=0),'--r',label="$\Im(E_z)$")


legend(loc=0)

grid()
xlabel('$f$')
ylabel("Number of dimensions")
show()
savefig("TchebSeriesE.pdf",bbox="tight")

figure(2)
semilogx(freq,TcPh_x.mean(axis=0),'b',label="$\phi_x$")
semilogx(freq,TcPh_y.mean(axis=0),'g',label="$\phi_y$")
semilogx(freq,TcPh_z.mean(axis=0),'r',label="$\phi_z$")
#semilogx(freq,entdPh_x.mean(axis=0),'--b',label="$\phi_x'$")
#semilogx(freq,entdPh_y.mean(axis=0),'--g',label="$\phi_y'$")
#semilogx(freq,entdPh_z.mean(axis=0),'--r',label="$\phi_z'$")
legend(loc=0)

grid()
xlabel('$f$')
ylabel("Number of dimensions")
show()
savefig("TchebSeriesPh.pdf",bbox="tight")

figure(3)
semilogx(freq,TcP.mean(axis=0),label="$P$")
legend(loc=0)
grid()
xlabel('$f$')
ylabel("Number of dimensions")
