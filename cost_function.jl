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

    C_T=2*W*W'-Matrix(I, P+4*M, P+4*M);
    A=C_T[1:P,1:P];
    B=C_T[1:P,P+1:P+4*M];
    D=C_T[P+1:P+4*M,P+1:P+4*M];

    #define pauli matrix
    sigmax=[0 1;1 0];
    sigmay=[0 -im;im 0];
    sigmaz=[1 0;0 -1];

    E=0;
    for ca=1:Lx
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



function Hofstadter_N2(tx,ty,t2,Lx,Ly,P,M,kxs,kys,W)
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
    for ca=1:Lx
        for cb=1:Ly
            kx=kxs[ca];
            ky=kys[cb];
            C_in_1=[zeros(M,M) exp(im*kx)*Matrix(I, M, M); exp(-im*kx)*Matrix(I, M, M) zeros(M,M)];
            C_in_2=[zeros(M,M) exp(im*ky)*Matrix(I, M, M); exp(-im*ky)*Matrix(I, M, M) zeros(M,M)];
            C_in=[C_in_1 zeros(2*M,2*M); zeros(2*M,2*M) C_in_2];

            C_out=A-B*pinv(D+C_in)*B';
            rho_out=(C_out+Matrix(I, P, P))/2;
            hx=-tx*(1+cos(kx))-2*t2*sin(ky)*(1-cos(kx));
            hy=-tx*sin(kx)+2*t2*sin(ky)*sin(kx);
            hz=2*ty*cos(ky);
            hk=hx*sigmax+hy*sigmay+hz*sigmaz;
            E=E+real(tr(hk*rho_out));
            #println(eigvals(C_out))
        end
    end
    E=E/N;

    return E

end


function Hofstadter_N4(tx,ty,t2,Lx,Ly,P,M,Q,kxs,kys,W,hh11,hh22,hh33,hh44,hh12,hh14,hh21,hh23,hh32,hh34,hh41,hh43)
    @assert size(W)[2]==Q;

    N=Lx*Ly;

    C_T=2*W*W'-Matrix(I, P+4*M, P+4*M);
    A=C_T[1:P,1:P];
    B=C_T[1:P,P+1:P+4*M];
    D=C_T[P+1:P+4*M,P+1:P+4*M];



    phi=pi/2;
    E=0;
    for ca=1:Lx
        for cb=1:Ly
            kx=kxs[ca];
            ky=kys[cb];
            C_in_1=[zeros(M,M) exp(im*kx)*Matrix(I, M, M); exp(-im*kx)*Matrix(I, M, M) zeros(M,M)];
            C_in_2=[zeros(M,M) exp(im*ky)*Matrix(I, M, M); exp(-im*ky)*Matrix(I, M, M) zeros(M,M)];
            C_in=[C_in_1 zeros(2*M,2*M); zeros(2*M,2*M) C_in_2];

            C_out=A-B*pinv(D+C_in)*B';
            rho_out=(C_out+Matrix(I, P, P))/2;

            hh=zeros(4,4)*im;
            hh=hh-ty*exp(im*phi+im*ky)*hh11;
            hh=hh-ty*exp(2*im*phi+im*ky)*hh22;
            hh=hh-ty*exp(3*im*phi+im*ky)*hh33;
            hh=hh-ty*exp(im*ky)*hh44;
            
            hh=hh-(t2*exp(2*im*phi-im*pi/4+im*ky))*hh12;
            hh=hh-(tx*exp(im*kx)+t2*exp(im*pi/4+im*kx+im*ky))*hh14;
        
            hh=hh-(tx+t2*exp(im*phi+im*pi/4+im*ky))*hh21;
            hh=hh-(t2*exp(3*im*phi-im*pi/4+im*ky))*hh23;
            
            hh=hh-(tx+t2*exp(2*im*phi+im*pi/4+im*ky))*hh32;
            hh=hh-(t2*exp(-im*pi/4+im*ky))*hh34;
            
            hh=hh-(t2*exp(im*phi-im*pi/4-im*kx+im*ky))*hh41;
            hh=hh-(tx+t2*exp(3*im*phi+im*pi/4+im*ky))*hh43;
            
            hh=hh+hh';

            E=E+real(tr(hh*rho_out));
            #println(eigvals(C_out))
        end
    end
    E=E/N;

    return E

end


function C2_model1_incorrect(Delta,t2,t3,Lx,Ly,P,M,kxs,kys,W)
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
    for ca=1:Lx
        for cb=1:Ly
            kx=kxs[ca];
            ky=kys[cb];
            C_in_1=[zeros(M,M) exp(im*kx)*Matrix(I, M, M); exp(-im*kx)*Matrix(I, M, M) zeros(M,M)];
            C_in_2=[zeros(M,M) exp(im*ky)*Matrix(I, M, M); exp(-im*ky)*Matrix(I, M, M) zeros(M,M)];
            C_in=[C_in_1 zeros(2*M,2*M); zeros(2*M,2*M) C_in_2];

            C_out=A-B*pinv(D+C_in)*B';
            rho_out=(C_out+Matrix(I, P, P))/2;

            hx=2*t2*cos(kx);
            hy=2*t3*cos(ky)-4*Delta*sin(kx)*sin(ky);
            hz=2*cos(kx)+2*cos(ky);

            hk=hx*sigmax+hy*sigmay+hz*sigmaz;
            E=E+real(tr(hk*rho_out));
            #println(eigvals(C_out))
        end
    end
    E=E/N;

    return E

end


function C2_model1_correct(Delta,Lx,Ly,P,M,kxs,kys,W)
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
    for ca=1:Lx
        for cb=1:Ly
            kx=kxs[ca];
            ky=kys[cb];
            C_in_1=[zeros(M,M) exp(im*kx)*Matrix(I, M, M); exp(-im*kx)*Matrix(I, M, M) zeros(M,M)];
            C_in_2=[zeros(M,M) exp(im*ky)*Matrix(I, M, M); exp(-im*ky)*Matrix(I, M, M) zeros(M,M)];
            C_in=[C_in_1 zeros(2*M,2*M); zeros(2*M,2*M) C_in_2];

            C_out=A-B*pinv(D+C_in)*B';
            rho_out=(C_out+Matrix(I, P, P))/2;

            hx=2*cos(kx)-2*cos(ky);
            hy=-4*Delta*sin(kx)*sin(ky);
            hz=2*cos(kx)+2*cos(ky);

            hk=hx*sigmax+hy*sigmay+hz*sigmaz;
            E=E+real(tr(hk*rho_out));
            #println(eigvals(C_out))
        end
    end
    E=E/N;

    return E

end



function C2_model1_correct_modified(Phx,Phz,Delta,Lx,Ly,P,M,kxs,kys,W)
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
    for ca=1:Lx
        for cb=1:Ly
            kx=kxs[ca];
            ky=kys[cb];
            C_in_1=[zeros(M,M) exp(im*kx)*Matrix(I, M, M); exp(-im*kx)*Matrix(I, M, M) zeros(M,M)];
            C_in_2=[zeros(M,M) exp(im*ky)*Matrix(I, M, M); exp(-im*ky)*Matrix(I, M, M) zeros(M,M)];
            C_in=[C_in_1 zeros(2*M,2*M); zeros(2*M,2*M) C_in_2];

            C_out=A-B*pinv(D+C_in)*B';
            rho_out=(C_out+Matrix(I, P, P))/2;

            hx=2*cos(kx)-2*cos(ky)+Phx*cos(2*ky);
            hy=-4*Delta*sin(kx)*sin(ky);
            hz=2*cos(kx)+2*cos(ky)+Phz*cos(2*ky);

            hk=hx*sigmax+hy*sigmay+hz*sigmaz;
            E=E+real(tr(hk*rho_out));
            #println(eigvals(C_out))
        end
    end
    E=E/N;

    return E

end


function C2_model2(R,Mz,Lx,Ly,P,M,kxs,kys,W)
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
    for ca=1:Lx
        for cb=1:Ly
            kx=kxs[ca];
            ky=kys[cb];
            C_in_1=[zeros(M,M) exp(im*kx)*Matrix(I, M, M); exp(-im*kx)*Matrix(I, M, M) zeros(M,M)];
            C_in_2=[zeros(M,M) exp(im*ky)*Matrix(I, M, M); exp(-im*ky)*Matrix(I, M, M) zeros(M,M)];
            C_in=[C_in_1 zeros(2*M,2*M); zeros(2*M,2*M) C_in_2];

            C_out=A-B*pinv(D+C_in)*B';
            rho_out=(C_out+Matrix(I, P, P))/2;

            hx=sin(kx)^2-sin(ky)^2-R;
            hy=2*sin(kx)*sin(ky);
            hz=Mz-cos(kx)-cos(ky);

            hk=hx*sigmax+hy*sigmay+hz*sigmaz;
            E=E+real(tr(hk*rho_out));
            #println(eigvals(C_out))
        end
    end
    E=E/N;

    return E

end



function C2_model1_correct_decoupled(Delta,Lx,Ly,P,M,kxs,kys,W)
    N=Lx*Ly;
    theta=0.25*pi;
    U_theta=[cos(theta/2) -sin(theta/2);sin(theta/2) cos(theta/2)]*(1+0*im);


    C_T=2*W*W'-Matrix(I, P+4*M, P+4*M);
    A=C_T[1:P,1:P];
    B=C_T[1:P,P+1:P+4*M];
    D=C_T[P+1:P+4*M,P+1:P+4*M];

    #define pauli matrix
    sigmax=[0 1;1 0];
    sigmay=[0 -im;im 0];
    sigmaz=[1 0;0 -1];

    E=0;
    for ca=1:Lx
        for cb=1:Ly
            kx=kxs[ca];
            ky=kys[cb];
            C_in_1=[zeros(M,M) exp(im*kx)*Matrix(I, M, M); exp(-im*kx)*Matrix(I, M, M) zeros(M,M)];
            C_in_2=[zeros(M,M) exp(im*ky)*Matrix(I, M, M); exp(-im*ky)*Matrix(I, M, M) zeros(M,M)];
            C_in=[C_in_1 zeros(2*M,2*M); zeros(2*M,2*M) C_in_2];

            C_out=A-B*pinv(D+C_in)*B';
            rho_out=(C_out+Matrix(I, P, P))/2;

            hx=2*cos(kx)-2*cos(ky);
            hy=-4*Delta*sin(kx)*sin(ky);
            hz=2*cos(kx)+2*cos(ky);

            hk=hx*sigmax+hy*sigmay+hz*sigmaz;

            hk=U_theta*hk*U_theta';

            E=E+real(tr(hk*rho_out));
            #println(eigvals(C_out))
        end
    end
    E=E/N;

    return E

end

function C2_model1_correct_decoupled_modified(Phx,Phz,Delta,Lx,Ly,P,M,kxs,kys,W)
    N=Lx*Ly;
    theta=0.25*pi;
    U_theta=[cos(theta/2) -sin(theta/2);sin(theta/2) cos(theta/2)]*(1+0*im);


    C_T=2*W*W'-Matrix(I, P+4*M, P+4*M);
    A=C_T[1:P,1:P];
    B=C_T[1:P,P+1:P+4*M];
    D=C_T[P+1:P+4*M,P+1:P+4*M];

    #define pauli matrix
    sigmax=[0 1;1 0];
    sigmay=[0 -im;im 0];
    sigmaz=[1 0;0 -1];

    E=0;
    for ca=1:Lx
        for cb=1:Ly
            kx=kxs[ca];
            ky=kys[cb];
            C_in_1=[zeros(M,M) exp(im*kx)*Matrix(I, M, M); exp(-im*kx)*Matrix(I, M, M) zeros(M,M)];
            C_in_2=[zeros(M,M) exp(im*ky)*Matrix(I, M, M); exp(-im*ky)*Matrix(I, M, M) zeros(M,M)];
            C_in=[C_in_1 zeros(2*M,2*M); zeros(2*M,2*M) C_in_2];

            C_out=A-B*pinv(D+C_in)*B';
            rho_out=(C_out+Matrix(I, P, P))/2;

            hx=2*cos(kx)-2*cos(ky)+Phx*cos(2*ky);
            hy=-4*Delta*sin(kx)*sin(ky);
            hz=2*cos(kx)+2*cos(ky)+Phz*cos(2*ky);

            hk=hx*sigmax+hy*sigmay+hz*sigmaz;

            hk=U_theta*hk*U_theta';

            E=E+real(tr(hk*rho_out));
            #println(eigvals(C_out))
        end
    end
    E=E/N;

    return E

end

function C2_model1_correct_theta(theta,Delta,Lx,Ly,P,M,kxs,kys,W)
    N=Lx*Ly;
    theta=theta*pi;
    U_theta=[cos(-theta/2) -sin(-theta/2);sin(-theta/2) cos(-theta/2)]*(1+0*im);
    theta0=0.25*pi;
    U_theta0=[cos(theta0/2) -sin(theta0/2);sin(theta0/2) cos(theta0/2)]*(1+0*im);


    C_T=2*W*W'-Matrix(I, P+4*M, P+4*M);
    A=C_T[1:P,1:P];
    B=C_T[1:P,P+1:P+4*M];
    D=C_T[P+1:P+4*M,P+1:P+4*M];

    #define pauli matrix
    sigmax=[0 1;1 0];
    sigmay=[0 -im;im 0];
    sigmaz=[1 0;0 -1];

    E=0;
    for ca=1:Lx
        for cb=1:Ly
            kx=kxs[ca];
            ky=kys[cb];
            C_in_1=[zeros(M,M) exp(im*kx)*Matrix(I, M, M); exp(-im*kx)*Matrix(I, M, M) zeros(M,M)];
            C_in_2=[zeros(M,M) exp(im*ky)*Matrix(I, M, M); exp(-im*ky)*Matrix(I, M, M) zeros(M,M)];
            C_in=[C_in_1 zeros(2*M,2*M); zeros(2*M,2*M) C_in_2];

            C_out=A-B*pinv(D+C_in)*B';
            rho_out=(C_out+Matrix(I, P, P))/2;

            hx=2*cos(kx)-2*cos(ky);
            hy=-4*Delta*sin(kx)*sin(ky);
            hz=2*cos(kx)+2*cos(ky);

            hk=hx*sigmax+hy*sigmay+hz*sigmaz;

            hk=U_theta0*U_theta*hk*U_theta'*U_theta0';

            E=E+real(tr(hk*rho_out));
            #println(eigvals(C_out))
        end
    end
    E=E/N;

    return E

end