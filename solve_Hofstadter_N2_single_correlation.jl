using LinearAlgebra
using JSON
using HDF5, JLD
using Random
using Flux
using MAT
cd("D:\\My Documents\\Code\\Julia_codes\\Tensor network\\Gaussian-fermion-peps")

include("cost_function_correlation.jl")


#Hamiltonian parameters
Random.seed!(777)
Lx=20;
Ly=20;
N=Lx*Ly;
boundary_phase_x=0;#between 0 and 1
boundary_phase_y=0;#between 0 and 1
tx=1;
ty=1;
t2=0.125;

#PEPS parameters
filling=1;
P=2;#number of physical fermion modes every unit-cell
M=2;#number of virtual modes per bond
M_initial=2;#number of virtual modes in initial state
#each site has 4M virtual fermion modes
Q=2*M+filling;#total number of physical and virtual fermions on a site;
#size of W matrix: (P+4M, Q)
#init_state="Hofstadter_N2_M"*string(M_initial)*".jld";#initialize: nothing
init_state=nothing

#optimization parameters
ls_max=20;
alpha0=2;
ls_ratio=1/3;
noise_ite=1;

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

#############################################
#compute exact correlation functions

sigmax=[0 1;1 0];
sigmay=[0 -im;im 0];
sigmaz=[1 0;0 -1];

#x direction correlations
ob_uu_exact=zeros(1,Lx)*(1+0*im);
ob_dd_exact=zeros(1,Lx)*(1+0*im);
ob_ud_exact=zeros(1,Lx)*(1+0*im);
ob_du_exact=zeros(1,Lx)*(1+0*im);

for ca=1:Lx
    for cb=1:Ly

        kx=kxs[ca];
        ky=kys[cb];


        
        hx=-tx*(1+cos(kx))-2*t2*sin(ky)*(1-cos(kx));
        hy=-tx*sin(kx)+2*t2*sin(ky)*sin(kx);
        hz=2*ty*cos(ky);
        
        hk=hx*sigmax+hy*sigmay+hz*sigmaz;
        eu,ev=eigen(hk);

        @assert eu[1]<eu[2];
        psi=ev[:,1];
        ob_uu_exact[ca]=ob_uu_exact[ca]+psi[1]'*psi[1]/Ly;
        ob_dd_exact[ca]=ob_dd_exact[ca]+psi[2]'*psi[2]/Ly;
        ob_ud_exact[ca]=ob_ud_exact[ca]+psi[1]'*psi[2]/Ly;
        ob_du_exact[ca]=ob_du_exact[ca]+psi[2]'*psi[1]/Ly;
    end
end


correl_uu_exact_right=zeros(1,Lx)*(1+0*im);
correl_dd_exact_right=zeros(1,Lx)*(1+0*im);
correl_ud_exact_right=zeros(1,Lx)*(1+0*im);
correl_du_exact_right=zeros(1,Lx)*(1+0*im);

correl_uu_exact_left=zeros(1,Lx)*(1+0*im);
correl_dd_exact_left=zeros(1,Lx)*(1+0*im);
correl_ud_exact_left=zeros(1,Lx)*(1+0*im);
correl_du_exact_left=zeros(1,Lx)*(1+0*im);

for cxa=1:Lx
    for cka=1:Lx
        correl_uu_exact_right[cxa]=correl_uu_exact_right[cxa]+exp(im*(cxa-1)*kxs[cka])*ob_uu_exact[cka]/Lx;
        correl_dd_exact_right[cxa]=correl_dd_exact_right[cxa]+exp(im*(cxa-1)*kxs[cka])*ob_dd_exact[cka]/Lx;
        correl_ud_exact_right[cxa]=correl_ud_exact_right[cxa]+exp(im*(cxa-1)*kxs[cka])*ob_ud_exact[cka]/Lx;
        correl_du_exact_right[cxa]=correl_du_exact_right[cxa]+exp(im*(cxa)*kxs[cka])*ob_du_exact[cka]/Lx;

        correl_uu_exact_left[cxa]=correl_uu_exact_left[cxa]+exp(-im*(cxa-1)*kxs[cka])*ob_uu_exact[cka]/Lx;
        correl_dd_exact_left[cxa]=correl_dd_exact_left[cxa]+exp(-im*(cxa-1)*kxs[cka])*ob_dd_exact[cka]/Lx;
        correl_ud_exact_left[cxa]=correl_ud_exact_left[cxa]+exp(-im*(cxa)*kxs[cka])*ob_ud_exact[cka]/Lx;
        correl_du_exact_left[cxa]=correl_du_exact_left[cxa]+exp(-im*(cxa-1)*kxs[cka])*ob_du_exact[cka]/Lx;
    end
end


######

#y direction correlations
ob_uu_exact=zeros(1,Ly)*(1+0*im);
ob_dd_exact=zeros(1,Ly)*(1+0*im);
ob_ud_exact=zeros(1,Ly)*(1+0*im);
ob_du_exact=zeros(1,Ly)*(1+0*im);

for cb=1:Ly
    for ca=1:Lx
    
        kx=kxs[ca];
        ky=kys[cb];


        
        hx=-tx*(1+cos(kx))-2*t2*sin(ky)*(1-cos(kx));
        hy=-tx*sin(kx)+2*t2*sin(ky)*sin(kx);
        hz=2*ty*cos(ky);
        
        hk=hx*sigmax+hy*sigmay+hz*sigmaz;
        eu,ev=eigen(hk);

        @assert eu[1]<eu[2];
        psi=ev[:,1];
        ob_uu_exact[cb]=ob_uu_exact[cb]+psi[1]'*psi[1]/Lx;
        ob_dd_exact[cb]=ob_dd_exact[cb]+psi[2]'*psi[2]/Lx;
        ob_ud_exact[cb]=ob_ud_exact[cb]+psi[1]'*psi[2]/Lx;
        ob_du_exact[cb]=ob_du_exact[cb]+psi[2]'*psi[1]/Lx;
    end
end


correl_uu_exact_up=zeros(1,Ly)*(1+0*im);
correl_dd_exact_up=zeros(1,Ly)*(1+0*im);
#correl_ud_exact_up=zeros(1,Ly)*(1+0*im);
#correl_du_exact_up=zeros(1,Ly)*(1+0*im);

correl_uu_exact_down=zeros(1,Ly)*(1+0*im);
correl_dd_exact_down=zeros(1,Ly)*(1+0*im);
#correl_ud_exact_down=zeros(1,Ly)*(1+0*im);
#correl_du_exact_down=zeros(1,Ly)*(1+0*im);

for cya=1:Ly
    for ckb=1:Ly
        correl_uu_exact_up[cya]=correl_uu_exact_up[cya]+exp(im*(cya-1)*kys[ckb])*ob_uu_exact[ckb]/Ly;
        correl_dd_exact_up[cya]=correl_dd_exact_up[cya]+exp(im*(cya-1)*kys[ckb])*ob_dd_exact[ckb]/Ly;
        #correl_ud_exact_up[cya]=correl_ud_exact_up[cya]+exp(im*(cya-1)*kys[ckb])*ob_ud_exact[ckb]/Ly;
        #correl_du_exact_up[cya]=correl_du_exact_up[cya]+exp(im*(cya-1)*kys[ckb])*ob_du_exact[ckb]/Ly;

        correl_uu_exact_down[cya]=correl_uu_exact_down[cya]+exp(-im*(cya-1)*kys[ckb])*ob_uu_exact[ckb]/Ly;
        correl_dd_exact_down[cya]=correl_dd_exact_down[cya]+exp(-im*(cya-1)*kys[ckb])*ob_dd_exact[ckb]/Ly;
        #correl_ud_exact_down[cya]=correl_ud_exact_down[cya]+exp(-im*(cya-1)*kys[ckb])*ob_ud_exact[ckb]/Ly;
        #correl_du_exact_down[cya]=correl_du_exact_down[cya]+exp(-im*(cya-1)*kys[ckb])*ob_du_exact[ckb]/Ly;
    end
end
###############################
correl_x=[correl_uu_exact_left,correl_ud_exact_left,correl_du_exact_left,correl_dd_exact_left,  correl_uu_exact_right,correl_ud_exact_right,correl_du_exact_right,correl_dd_exact_right];
correl_y=[correl_uu_exact_up,correl_dd_exact_up,  correl_uu_exact_down,correl_dd_exact_down];
###############################


correl_distance=5;
#cost_f(W)=Hofstadter_N2(tx,ty,t2,Lx,Ly,P,M,kxs,kys,W);
cost_f(W)=Hofstadter_N2_single_correlation(tx,ty,t2,Lx,Ly,P,M,kxs,kys,W,correl_x,correl_y,correl_distance);


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


jld_filenm="Hofstadter_N2_M"*string(M)*".jld";
save(jld_filenm, "W",W,"E0",E0);

mat_filenm="Hofstadter_N2_M"*string(M)*".mat";
matwrite(mat_filenm, Dict(
    "W" => W,
    "E0" => E0
); compress = false)
