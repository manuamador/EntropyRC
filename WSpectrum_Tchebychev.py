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
mu0=4*pi*1e-7
eps0=1./(mu0*c**2)



def Hertz_dipole_ff (r, p, R, phi, f, t=0, epsr=1.):
  """
  Calculate E and B field strength radaited by hertzian dipole(s) in the far field.
  p: array of dipole moments [[px0,py0,pz0],[px1,py1,pz1],...[pxn,pyn,pzn]]
  R: array of dipole positions [[X0,Y0,Z0],[X1,Y1,Z1],...[Xn,Yn,Zn]]
  r: observation point [x,y,z]
  f: array of frequencies [f0,f1,...]
  t: time
  phi: array with dipole phase angles (0..2pi) [phi0,phi1,...,phin]
  return: fields values at observation point r at time t for every frequency in f. E and B are (3 components,number of frequencies) arrays.
  """
  nf = len(f)
  rprime = r-R  # r'=r-R
  if ndim(p) < 2:
    magrprime = sqrt(sum((rprime)**2))
    magrprimep = tile(magrprime, (len(f),1)).T
    phip = tile(phi, (len(f),1))
    w = 2*pi*f  # \omega
    k = w/c     # wave number
    krp = k*magrprimep  # k|r'|
    rprime_cross_p = cross(rprime, p) # r'x p
    rp_c_p_c_rp = cross(rprime_cross_p, rprime) # (r' x p) x r'
    expfac = exp(1j*(w*t-krp+phip.T))/(4*pi*eps0*epsr)
    Ex = (w**2/(c**2*magrprimep**3) * expfac)* (tile(rp_c_p_c_rp[0],(nf,1))).T
    Ey = (w**2/(c**2*magrprimep**3) * expfac)* (tile(rp_c_p_c_rp[1],(nf,1))).T
    Ez = (w**2/(c**2*magrprimep**3) * expfac)* (tile(rp_c_p_c_rp[2],(nf,1))).T
    Bx = expfac/(magrprimep**2*c**3)*(w**2*tile(rprime_cross_p[0],(nf,1)).T)
    By = expfac/(magrprimep**2*c**3)*(w**2*tile(rprime_cross_p[1],(nf,1)).T)
    Bz = expfac/(magrprimep**2*c**3)*(w**2*tile(rprime_cross_p[2],(nf,1)).T)
    E = vstack((Ex,Ey,Ez))
    B = vstack((Bx,By,Bz))
  else:
    magrprime = sqrt(sum((rprime)**2,axis=1)) # |r'|
    magrprimep = tile(magrprime, (len(f),1)).T
    phip = tile(phi, (len(f),1))
    fp = tile(f,(len(magrprime),1))
    w = 2*pi*fp  # \omega
    k = w/c     # wave number
    krp = k*magrprimep  # k|r'|
    rprime_cross_p = cross(rprime, p) # r'x p
    rp_c_p_c_rp = cross(rprime_cross_p, rprime) # (r' x p) x r'
    expfac = exp(1j*(w*t-krp+phip.T))/(4*pi*eps0*epsr)
    Ex = (w**2/(c**2*magrprimep**3) * expfac)* (tile(rp_c_p_c_rp[:,0],(nf,1))).T
    Ey = (w**2/(c**2*magrprimep**3) * expfac)* (tile(rp_c_p_c_rp[:,1],(nf,1))).T
    Ez = (w**2/(c**2*magrprimep**3) * expfac)* (tile(rp_c_p_c_rp[:,2],(nf,1))).T
    Bx = expfac/(magrprimep**2*c**3)*(w**2*tile(rprime_cross_p[:,0],(nf,1)).T)
    By = expfac/(magrprimep**2*c**3)*(w**2*tile(rprime_cross_p[:,1],(nf,1)).T)
    Bz = expfac/(magrprimep**2*c**3)*(w**2*tile(rprime_cross_p[:,2],(nf,1)).T)
    E = vstack((sum(Ex,axis=0),sum(Ey,axis=0),sum(Ez,axis=0)))
    B = vstack((sum(Bx,axis=0),sum(By,axis=0),sum(Bz,axis=0)))
  return E,B




#observation points on a circle
radius=1
nphi=360
phi=linspace(2*pi/nphi,2*pi,nphi)
x=radius*cos(phi)
y=radius*sin(phi)
z=0*ones(len(phi))


f0=10e6
f1=10e9
nf=100
freq=10**(linspace(log10(f0),log10(f1),nf))

#random dipoles that radiates plane waves from their distance...
n_dip=1000 #number of Herztian dipoles

#dipole positions
distance=1000

M=100

Er=zeros((M,nphi,nf,3))
Ei=zeros((M,nphi,nf,3))
Phase=zeros((M,nphi,nf,3))
P=zeros((M,nphi,nf))

for i in range(M):
    phi_posdip=2*pi*random(n_dip)
    th_posdip=arccos(2*random(n_dip)-1)
    R=(array([distance*sin(th_posdip)*cos(phi_posdip),distance*sin(th_posdip)*sin(phi_posdip),distance*cos(th_posdip)])).T
    #dipole moments
    pmax=1e-7 #maximum dipole moment p
    r_dip=pmax*random(n_dip)
    phi_dip=2*pi*random(n_dip)
    th_dip=arccos(2*random(n_dip)-1)
    p=(array([r_dip*sin(th_dip)*cos(phi_dip),r_dip*sin(th_dip)*sin(phi_dip),r_dip*cos(th_dip)])).T
    #dipole phases
    phases_dip=2*pi*random(n_dip)
    for j in range(nphi):
        r=(array([x[j],y[j],z[j]])).T
        E,B=Hertz_dipole_ff (r, p, R, phases_dip, freq, t=0, epsr=1.)
        P[i,j,:]=sqrt(sum((0.5*abs(cross(E.T,(B.T))))**2,axis=1))
        Er[i,j,:,:]=real(E).T
        Ei[i,j,:,:]=imag(E).T
        Phase[i,j,:,:]=angle(E).T
    print('%s/%s'%(i+1,M))



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
    U=find(abs(coeff)>crit*max(abs(coeff)))
    a=U[-1]
    return a


binsize=10
for k in range(M):
    print k
    for i in range(nf):
        TcEx_r[k,i]=maxorder(chebpolyfit(Er[k,:,i,0]))
        TcEy_r[k,i]=maxorder(chebpolyfit(Er[k,:,i,1]))
        TcEz_r[k,i]=maxorder(chebpolyfit(Er[k,:,i,2]))
        TcEx_i[k,i]=maxorder(chebpolyfit(Er[k,:,i,0]))
        TcEy_i[k,i]=maxorder(chebpolyfit(Er[k,:,i,1]))
        TcEz_i[k,i]=maxorder(chebpolyfit(Er[k,:,i,2]))
        TcPx=maxorder(chebpolyfit(Phase[k,:,i,0]))
        TcPy=maxorder(chebpolyfit(Phase[k,:,i,1]))
        TcPz=maxorder(chebpolyfit(Phase[k,:,i,2]))
        Px=Phase[k,i,:,0]/Phase[k,:,i,0].mean()
        Py=Phase[k,i,:,1]/Phase[k,:,i,1].mean()
        Pz=Phase[k,i,:,2]/Phase[k,:,i,2].mean()
        TcPh_x[k,i]=maxorder(chebpolyfit(Px))
        TcPh_y[k,i]=maxorder(chebpolyfit(Py))
        TcPh_z[k,i]=maxorder(chebpolyfit(Pz))
        TcdPh_x[k,i]=maxorder(chebpolyfit(diff(Px)))
        TcdPh_y[k,i]=maxorder(chebpolyfit(diff(Py)))
        TcdPh_z[k,i]=maxorder(chebpolyfit(diff(Pz)))
        TcP[k,i]=maxorder(chebpolyfit(P[k,:,i]/P[k,:,i].mean()))



figure(1)
semilogx(freq,nanmean(TcEx_r,axis=0),'b',label="$\Re(E_x)$")
semilogx(freq,nanmean(TcEx_i,axis=0),'--b',label="$\Im(E_x)$")
semilogx(freq,nanmean(TcEy_r,axis=0),'g',label="$\Re(E_y)$")
semilogx(freq,nanmean(TcEy_i,axis=0),'--g',label="$\Im(E_y)$")
semilogx(freq,nanmean(TcEz_r,axis=0),'r',label="$\Re(E_z)$")
semilogx(freq,nanmean(TcEz_i,axis=0),'--r',label="$\Im(E_z)$")


legend(loc=0)

grid()
xlabel('$f$')
ylabel("Number of dimensions")
show()
savefig("WS_TchebSeriesE.pdf",bbox="tight")

figure(2)
semilogx(freq,nanmean(TcPh_x,axis=0),'b',label="$\phi_x$")
semilogx(freq,nanmean(TcPh_y,axis=0),'g',label="$\phi_y$")
semilogx(freq,nanmean(TcPh_z,axis=0),'r',label="$\phi_z$")
#semilogx(freq,entdPh_x.mean(axis=0),'--b',label="$\phi_x'$")
#semilogx(freq,entdPh_y.mean(axis=0),'--g',label="$\phi_y'$")
#semilogx(freq,entdPh_z.mean(axis=0),'--r',label="$\phi_z'$")
legend(loc=0)

grid()
xlabel('$f$')
ylabel("Number of dimensions")
show()
savefig("WS_TchebSeriesPh.pdf",bbox="tight")

figure(3)
semilogx(freq,nanmean(TcP,axis=0),label="$P$")
legend(loc=0)
grid()
xlabel('$f$')
ylabel("Number of dimensions")
