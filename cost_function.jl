function square_dirac(Lx,Ly,P,M,kxs,kys,W)
    N=Lx*Ly;

    C_T=2*W*W'-Matrix(I, P+4*M, P+4*M);
    A=C_T[1:P,1:P];
    B=C_T[1:P,P+1:P+4*M];
    D=C_T[P+1:P+4*M,P+1:P+4*M];

    #define pauli matrix
    sigmax=[0 1;1 0];
    sigmay=[0 -im;im 0];
    sigmaz=[1 0;0 -1];

    E=0;
    for ca=1:Int(Lx/2)
        for cb=1:Ly
            kx=kxs[ca];
            ky=kys[cb];
            C_in_1=[zeros(M,M) exp(im*kx)*Matrix(I, M, M); exp(-im*kx)*Matrix(I, M, M) zeros(M,M)];
            C_in_2=[zeros(M,M) exp(im*ky)*Matrix(I, M, M); exp(-im*ky)*Matrix(I, M, M) zeros(M,M)];
            C_in=[C_in_1 zeros(2*M,2*M); zeros(2*M,2*M) C_in_2];
            
            C_out=A-B*pinv(D+C_in)*B';
            rho_out=(C_out+Matrix(I, P, P))/2;
            
            hk=t*2*cos(ky)*sigmaz+(t+t*cos(kx))*sigmax+t*sin(kx)*sigmay;
            E=E+real(tr(hk*rho_out));
            #println(eigvals(C_out))
        end
    end
    E=E/N;

    return E

end



function Qi_Wu_Zhang(Mz,Lx,Ly,P,M,kxs,kys,W)
    N=Lx*Ly;
    #t=1;

    C_T=2*W*W'-Matrix(I, P+4*M, P+4*M);
    A=C_T[1:P,1:P];
    B=C_T[1:P,P+1:P+4*M];
    D=C_T[P+1:P+4*M,P+1:P+4*M];

    #define pauli matrix
    sigmax=[0 1;1 0];
    sigmay=[0 -im;im 0];
    sigmaz=[1 0;0 -1];

    E=0;
    for ca=1:Int(Lx/2)
        for cb=1:Ly
            kx=kxs[ca];
            ky=kys[cb];
            C_in_1=[zeros(M,M) exp(im*kx)*Matrix(I, M, M); exp(-im*kx)*Matrix(I, M, M) zeros(M,M)];
            C_in_2=[zeros(M,M) exp(im*ky)*Matrix(I, M, M); exp(-im*ky)*Matrix(I, M, M) zeros(M,M)];
            C_in=[C_in_1 zeros(2*M,2*M); zeros(2*M,2*M) C_in_2];
            
            C_out=A-B*pinv(D+C_in)*B';
            rho_out=(C_out+Matrix(I, P, P))/2;
            
            hk=2*sin(ky)*sigmax-2*sin(kx)*sigmay+(Mz-2*cos(kx)-2*cos(ky))*sigmaz;
            E=E+real(tr(hk*rho_out));
            #println(eigvals(C_out))
        end
    end
    E=E/N;

    return E

end