include("ImageCreator.jl")
include("dipole.jl")

const c = 299792458.
const mu0 = 4*pi*1e-7
const eps0 = 1/(mu0*c^2)

const freq=linspace(100e6,1e9,181)
const nf=length(freq)

#cavity
const L=5
const l=4
const h=3
const losses=0.998*ones(nf)  #losses can be frequency dependent...
#crit=0.01 #convergence criterion
#order=int(log(crit)/log(maximum(losses)))
const order=20

##dipole
#dipole position
const X=1
const Y=1
const Z=1
const tilt=pi/2-acos(sqrt(2/3));
const azimut=pi/4
const phase=0
#dipole moment
#total time averaged radiated power P= 1 W dipole moment => |p|=sqrt(12πcP/µOω⁴)
const Pow=1
const amplitude=sqrt(12*pi*c*Pow./(mu0*(2*pi*freq).^4))

#Images
POS=IC(L,l,h,X,Y,Z,tilt,azimut,phase,1,order)

const numberofimages=length(POS[:,1])

#observation points
na=360
phi=linspace(2*pi/na,2*pi,na)
radius=1
x=radius*cos(phi)+3
y=radius*sin(phi)+1.5
z=2

using PyPlot
println("Computing the radiation...")
Et=zeros(Complex128,nf,na,3)
for k=1:nf
  println("$k/$nf")
  for i=1:na
    perc=round(i/na*1000)/10
    println("$perc %")
    r=[x[i],y[i],z]
    E=zeros(Complex128,3)
    for m=1:numberofimages
      p=vec(POS[m,1:3])*amplitude[k]*losses[k]^POS[m,7]
      R=vec(POS[m,4:6])
      ord=POS[m,7]
      Ed,Bd=Hertz_dipole (r, p, R, phase, freq[k])
      E+=Ed
    end
    Et[k,i,:]=E
  end
end

#Back up
using NPZ
npzwrite("Estirrer.npz", ["Et" => Et, "freq" => freq, "x" => x, "y" => y, "z" => z])
