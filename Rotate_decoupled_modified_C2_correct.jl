using LinearAlgebra
using TensorKit
using JSON
using HDF5, JLD2, MAT
using Combinatorics

cd("D:\\My Documents\\Code\\Julia_codes\\Tensor network\\Gaussian-fermion-peps")



theta=0.225;#in the unit of pi, acting on the decoupled two-layer state



#PEPS parameters
filling=1;
P=2;#number of physical fermion modes every unit-cell
M=2;#number of virtual modes per bond

#each site has 4M virtual fermion modes
Q=2*M+filling;#total number of physical and virtual fermions on a site; 
#size of W matrix: (P+4M, Q)
#init_state="Hofstadter_N2_M"*string(M)*".jld";#initialize: nothing
#init_state="QWZ_M"*string(M)*".jld";#initialize: nothing
#init_state="C2_model1_correct_M"*string(M)*".jld";#initialize: nothing
init_state="C2_model1_correct_decoupled_modified_M"*string(M)*".jld";#initialize: nothing
#init_state="C2_model1_incorrect_M"*string(M)*".jld";#initialize: nothing
#init_state="C2_model1_correct_modified_M"*string(M)*".jld";#initialize: nothing

W=load(init_state)["W"];
E0=load(init_state)["E0"];

UU=Matrix(I, size(W)[1], size(W)[1])*(1.000+0*im);
U_theta=[cos(-theta/2*pi) -sin(-theta/2*pi);sin(-theta/2*pi) cos(-theta/2*pi)]*(1+0*im);
UU[1:2,1:2]=U_theta;

W=UU*W;



jld_filenm="Rotate_decoupled_modified_C2_theta_"*string(theta)*"_M"*string(M)*".jld";
save(jld_filenm, "W",W,"E0",E0);

mat_filenm="Rotate_decoupled_modified_C2_theta_"*string(theta)*"_M"*string(M)*".mat";
matwrite(mat_filenm, Dict(
    "W" => W,
    "E0" => E0
); compress = false)






