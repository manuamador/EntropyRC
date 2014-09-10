# -*- coding: utf-8 -*-
"""
This file gives the dipole radiation (E and B field) in the far field, the full radiation (near field + far field) and the near field radiation only

@author: manu
"""
from __future__ import division
from numpy import *
from numpy.random import *
from pylab import *
#from entropy import *
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


f0=1e6
f1=10e9
nf=30
freq=10**(linspace(log10(f0),log10(f1),nf))

#random dipoles that radiates plane waves from their distance...
n_dip=100 #number of Herztian dipoles

#dipole positions 
distance=1000
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

P=zeros((nphi,nf))
print("Computing the power received along the circle...")
for i in range(nphi):
    r=(array([x[i],y[i],z[i]])).T
    E,B=Hertz_dipole_ff (r, p, R, phases_dip, freq, t=0, epsr=1.)
    P[i,:]=sqrt(sum((0.5*abs(cross(E.T,(B.T))))**2,axis=1))
    #P[i,j,:]=.5*sum(abs(E)**2,axis=0) # gives the average value of the power over a period, if you want the radiated power at time t=0, please consider using 0.5*sum(real(E)**2,axis=0)
    print('%2.1f/100'%((i+1)/nphi*100))

ent=zeros(nf)
for i in range(nf):
    ent[i]=entropy(P[:,i]/P[:,i].mean(),10)
semilogx(freq,ent,'.')
grid()
xlabel('$f$')
ylabel("Entropy")
#show()
#savefig("Entropy.pdf",bbox="tight")