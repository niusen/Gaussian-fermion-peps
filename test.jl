using LinearAlgebra
using JSON
using HDF5, JLD
using Random
using Flux
using MAT
cd("D:\\My Documents\\Code\\Julia_codes\\Tensor network\\Gaussian-fermion-peps")

include("cost_function.jl")


#Hamiltonian parameters
Random.seed!(777)
Lx=80;
Ly=80;
N=Lx*Ly;
boundary_x=1;#1 or -1
boundary_y=1;#1 or -1
tx=1;
ty=1;
t2=ty/2;

#PEPS parameters
filling=1;
P=2;#number of physical fermion modes every unit-cell
M=2;#number of virtual modes per bond
M_initial=1;#number of virtual modes in initial state
#each site has 4M virtual fermion modes
Q=2*M+filling;#total number of physical and virtual fermions on a site; 
#size of W matrix: (P+4M, Q)
init_state="Hofstadter_N2_M"*string(M_initial)*".jld";#initialize: nothing
#init_state=nothing

#optimization parameters
ls_max=20;
alpha0=2;
ls_ratio=2/3;
noise_ite=10;

function initial_W(P,M,Q)
    W=rand(P+4*M,P+4*M)+im*rand(P+4*M,P+4*M);
    U,_,_=svd(W);
    W=U[:,1:Q];
end

if init_state==nothing 
    W=initial_W(P,M,Q);
else
    if M_initial==M
        W=load(init_state)["W"];
        E0=load(init_state)["E0"];
    elseif M_initial<M
        W_init=load(init_state)["W"];
        E0=load(init_state)["E0"];
        Q_initial=2*M_initial+filling;
        W=[W_init[1:P+M_initial,:];zeros(M-M_initial,Q_initial);W_init[P+M_initial+1:P+2*M_initial,:];zeros(M-M_initial,Q_initial);W_init[P+2*M_initial+1:P+3*M_initial,:];zeros(M-M_initial,Q_initial);W_init[P+3*M_initial+1:P+4*M_initial,:];zeros(M-M_initial,Q_initial)];
        W=[W zeros(size(W,1),2*(M-M_initial))];
        for cc=1:M-M_initial
            W[P+M_initial+cc,Q_initial+cc]=1;
            W[P+2*M+M_initial+cc,Q_initial+M-M_initial+cc]=1;
        end
        @assert norm(W'*W-I(Q))<1e-12;
    end


    #############
    # W[4,4]=1/2;
    # W[6,4]=1/2;
    # W[8,4]=1/2;
    # W[10,4]=1/2;
    # W[4,5]=1/2;
    # W[6,5]=1/2;
    # W[8,5]=-1/2;
    # W[10,5]=-1/2;
    #############

end

if boundary_x==1
    kxs=Float64.(1:1:Lx)*(2*pi)/Lx;
elseif boundary_x==-1
    kxs=Float64.(1:1:Lx)*(2*pi)/Lx.+pi/Lx;
end
if boundary_y==1
    kys=Float64.(1:1:Ly)*(2*pi)/Ly;
elseif boundary_y==-1
    kys=Float64.(1:1:Ly)*(2*pi)/Ly.+pi/Ly;
end

kxset=zeros(Lx,Ly);
kyset=zeros(Lx,Ly);
for ca=1:Lx
    for cb=1:Ly
        ca=ca;
        kxset[ca,cb]=kxs[ca];
        kyset[ca,cb]=kys[cb];
        kx=kxs[ca];
        ky=kys[cb];
    end
end


cost_f(W)=Hofstadter_N2(tx,ty,t2,Lx,Ly,P,M,kxs,kys,W);

println(cost_f(W))


