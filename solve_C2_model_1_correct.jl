using LinearAlgebra
using JSON
using HDF5, JLD
using Random
using Flux
using MAT
cd("D:\\My Documents\\Code\\Julia_codes\\Tensor network\\Gaussian-fermion-peps")

include("cost_function.jl")


#Hamiltonian parameters
Random.seed!(444)
Lx=80;
Ly=4;
N=Lx*Ly;
boundary_phase_x=0;#between 0 and 1
boundary_phase_y=0.02;#between 0 and 1

Delta=1/sqrt(2);

#PEPS parameters
filling=1;
P=2;#number of physical fermion modes every unit-cell
M=2;#number of virtual modes per bond
M_initial=2;#number of virtual modes in initial state
#each site has 4M virtual fermion modes
Q=2*M+filling;#total number of physical and virtual fermions on a site;
#size of W matrix: (P+4M, Q)
#init_state="C2_model1_correct_M"*string(M_initial)*".jld";#initialize: nothing
init_state=nothing

#optimization parameters
ls_max=20;
alpha0=2;
ls_ratio=2/3;
noise_ite=3;

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

end


kxs=Float64.(1:1:Lx)*(2*pi)/Lx.+boundary_phase_x*2*pi/Lx;
kys=Float64.(1:1:Ly)*(2*pi)/Ly.+boundary_phase_y*2*pi/Ly;


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


cost_f(W)=C2_model1_correct(Delta,Lx,Ly,P,M,kxs,kys,W);


function line_search(W,noise)

    M=rand(size(W,1),size(W,1))+im*rand(size(W,1),size(W,1));
    W=exp(im*noise*(M+M'))*W;



    E0=0;
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
        direction_old=direction;
        Q_G_old=Q_G;

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
    return W,E0
end


noise=0;
W,E0=line_search(W,noise);


for cn=1:noise_ite
    noise=0.6;
    W_updated,E0_updated=line_search(W,noise);
    if E0_updated<E0
        E0=E0_updated;
        W=W_updated;
    end

    noise=0.3;
    W_updated,E0_updated=line_search(W,noise);
    if E0_updated<E0
        E0=E0_updated;
        W=W_updated;
    end

    noise=0.1;
    W_updated,E0_updated=line_search(W,noise);
    if E0_updated<E0
        E0=E0_updated;
        W=W_updated;
    end

    noise=0.05;
    W_updated,E0_updated=line_search(W,noise);
    if E0_updated<E0
        E0=E0_updated;
        W=W_updated;
    end

    noise=0.01;
    W_updated,E0_updated=line_search(W,noise);
    if E0_updated<E0
        E0=E0_updated;
        W=W_updated;
    end

    noise=0.005;
    W_updated,E0_updated=line_search(W,noise);
    if E0_updated<E0
        E0=E0_updated;
        W=W_updated;
    end

    noise=0.001;
    W_updated,E0_updated=line_search(W,noise);
    if E0_updated<E0
        E0=E0_updated;
        W=W_updated;
    end

end

println(E0)


jld_filenm="C2_model1_correct_M"*string(M)*".jld";
save(jld_filenm, "W",W,"E0",E0);

mat_filenm="C2_model1_correct_M"*string(M)*".mat";
matwrite(mat_filenm, Dict(
    "W" => W,
    "E0" => E0
); compress = false)
