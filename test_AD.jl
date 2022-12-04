using LinearAlgebra
using Flux

#example 1
function ff(x)
    y=x[1]^2+x[2]^3;
    return y
end

x=[1,2];
gs = gradient(Flux.params(x)) do
         ff(x)
end

println(gs[x])



#example 2
function gg(x)
    eu=eigvals(x);
    y=real(eu[1]);
    println(y)
    return y
end

x=Matrix{ComplexF64}(undef, 2, 2);
x[1,1]=1;
x[1,2]=3;
x[2,1]=2;
x[2,2]=5;

gs = gradient(Flux.params(x)) do
         gg(x)
end

println(gs[x])
