using LinearAlgebra
using JSON
using HDF5, JLD
using Random
using Flux
cd("D:\\My Documents\\Code\\Julia_codes\\Tensor network\\Gaussian_fermion_peps")

include("cost_function.jl")

#optimization parameters
ls_max=20;
alpha0=2;
ls_ratio=2/3;

#Hamiltonian parameters
Random.seed!(777)
Lx=16;
Ly=16;
N=Lx*Ly;
boundary_x=-1;#1 or -1
boundary_y=1;#1 or -1
t=1;

#PEPS parameters
filling=1;
P=2;#number of physical fermion modes every unit-cell
M=4;#number of virtual modes per bond
#each site has 4M virtual fermion modes
Q=2*M+filling;#total number of physical and virtual fermions on a site; 
#size of W matrix: (P+4M, Q)

W=rand(P+4*M,P+4*M)+im*rand(P+4*M,P+4*M);
U,_,_=svd(W);
W=U[:,1:Q];

if boundary_x==1
    kxs=Float64.(1:1:Lx/2)*(2*pi)/(Lx/2);
elseif boundary_x==-1
    kxs=Float64.(1:1:Lx/2)*(2*pi)/(Lx/2).+pi/(Lx/2);
end
if boundary_y==1
    kys=Float64.(1:1:Ly)*(2*pi)/Ly;
elseif boundary_y==-1
    kys=Float64.(1:1:Ly)*(2*pi)/Ly.+pi/Ly;
end

kxset=zeros(Int(Lx/2),Ly);
kyset=zeros(Int(Lx/2),Ly);
for ca=1:Lx/2
    for cb=1:Ly
        ca=Int(ca);
        kxset[ca,cb]=kxs[ca];
        kyset[ca,cb]=kys[cb];
        kx=kxs[ca];
        ky=kys[cb];
    end
end


cost_f(W)=square_dirac(Lx,Ly,P,M,kxs,kys,W);



Q_G_old=zeros(size(W));
direction_old=zeros(size(W,1),size(W,1));
for op_step=1:1000
    E0=cost_f(W);
    println("Optimization "*string(op_step)*", E0="*string(E0));flush(stdout);
    grad = gradient(Flux.params(W)) do
        cost_f(W)
    end
    g=grad[W];
    Q_G=g*W'-W*g';
    improved=false;
    improvement=0;
   
    #conjugate gradient opt
    norm_grad=norm(Q_G)
    norm_grad0=norm(Q_G_old)
    beta=(norm_grad^2)/(norm_grad0^2)
    if op_step==1
        direction=Q_G;
    else
        direction=Q_G+beta*direction_old;
    end
    for ls_step=0:ls_max-1  
        alpha=alpha0*(ls_ratio^ls_step);
        W_new=exp(-alpha*direction)*W;
        E=cost_f(W_new);
        
        println("   Conjugate gradient opt, LS="*string(ls_step+1)*", "*"E="*string(E));flush(stdout);
        improvement=E-E0;
        if E<E0
            improved=true
            W=W_new;
            break
        end
    end
    direction0=direction;
    

    if improved
    else
        #gradient opt
        for ls_step=0:ls_max-1    
            alpha=alpha0*(ls_ratio^ls_step);
            W_new=exp(-alpha*Q_G)*W;
            E=cost_f(W_new);
            
            println("   Gradient opt, LS="*string(ls_step+1)*", "*"E="*string(E));flush(stdout);
            improvement=E-E0;
            if E<E0
                improved=true
                W=W_new;
                break
            end
        end
    end
    if -improvement<1e-7
        break;
    end

end


