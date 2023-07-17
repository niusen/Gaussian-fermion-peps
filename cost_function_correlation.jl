

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

    #energy
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




function Hofstadter_N2_correlation(tx,ty,t2,Lx,Ly,P,M,kxs,kys,W,correl_x,correl_y,correl_distance)
    N=Lx*Ly;

    C_T=2*W*W'-Matrix(I, P+4*M, P+4*M);
    A=C_T[1:P,1:P];
    B=C_T[1:P,P+1:P+4*M];
    D=C_T[P+1:P+4*M,P+1:P+4*M];

    #define pauli matrix
    sigmax=[0 1;1 0];
    sigmay=[0 -im;im 0];
    sigmaz=[1 0;0 -1];
    Pr1=[1;0];
    Pr2=[0;1];

    correl_uu_exact_left=correl_x[1];
    correl_ud_exact_left=correl_x[2];
    correl_du_exact_left=correl_x[3];
    correl_dd_exact_left=correl_x[4];
    correl_uu_exact_right=correl_x[5];
    correl_ud_exact_right=correl_x[6];
    correl_du_exact_right=correl_x[7];
    correl_dd_exact_right=correl_x[8];

    correl_uu_exact_up=correl_y[1];
    correl_dd_exact_up=correl_y[2];
    correl_uu_exact_down=correl_y[3];
    correl_dd_exact_down=correl_y[4];

    #energy
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


    ######################################
    #correlation functions
    #x direction correlations

    for dis_x=0:correl_distance
        correl_uu_right=0*(1+0*im);
        correl_ud_right=0*(1+0*im);
        correl_du_right=0*(1+0*im);
        correl_dd_right=0*(1+0*im);

        correl_uu_left=0*(1+0*im);
        correl_ud_left=0*(1+0*im);
        correl_du_left=0*(1+0*im);
        correl_dd_left=0*(1+0*im);
    
        for ca=1:Lx
            for cb=1:Ly

                kx=kxs[ca];
                ky=kys[cb];

                C_in_1=[zeros(M,M) exp(im*kx)*Matrix(I, M, M); exp(-im*kx)*Matrix(I, M, M) zeros(M,M)];
                C_in_2=[zeros(M,M) exp(im*ky)*Matrix(I, M, M); exp(-im*ky)*Matrix(I, M, M) zeros(M,M)];
                C_in=[C_in_1 zeros(2*M,2*M); zeros(2*M,2*M) C_in_2];

                C_out=A-B*pinv(D+C_in)*B';
                rho_out=(C_out+Matrix(I, P, P))/2;

                ob_uu_=Pr1'*rho_out*Pr1;
                ob_dd_=Pr2'*rho_out*Pr2;
                ob_ud_=Pr2'*rho_out*Pr1;
                ob_du_=Pr1'*rho_out*Pr2;


                correl_uu_right=correl_uu_right+exp(im*(dis_x)*kxs[ca])*ob_uu_/Lx/Ly;
                correl_dd_right=correl_dd_right+exp(im*(dis_x)*kxs[ca])*ob_dd_/Lx/Ly;
                correl_ud_right=correl_ud_right+exp(im*(dis_x)*kxs[ca])*ob_ud_/Lx/Ly;
                correl_du_right=correl_du_right+exp(im*(dis_x)*kxs[ca])*ob_du_/Lx/Ly;

                correl_uu_left=correl_uu_left+exp(-im*(dis_x)*kxs[ca])*ob_uu_/Lx/Ly;
                correl_dd_left=correl_dd_left+exp(-im*(dis_x)*kxs[ca])*ob_dd_/Lx/Ly;
                correl_ud_left=correl_ud_left+exp(-im*(dis_x)*kxs[ca])*ob_ud_/Lx/Ly;
                correl_du_left=correl_du_left+exp(-im*(dis_x)*kxs[ca])*ob_du_/Lx/Ly;
            end

        end
        #println("distance: "*string(dis_x))
        correl_GfPEPS=[correl_uu_right,correl_dd_right,correl_ud_right,correl_du_right,correl_uu_left,correl_dd_left,correl_ud_left,correl_du_left];

        correl_exact=[correl_uu_exact_right[dis_x+1],correl_dd_exact_right[dis_x+1],correl_ud_exact_right[dis_x+1],correl_du_exact_right[dis_x+1],correl_uu_exact_left[dis_x+1],correl_dd_exact_left[dis_x+1],correl_ud_exact_left[dis_x+1],correl_du_exact_left[dis_x+1]];
        #println(norm(correl_GfPEPS-correl_exact))

        E=E+100*real(dot(correl_GfPEPS-correl_exact,correl_GfPEPS-correl_exact));

    end
    

    return E
end




function Hofstadter_N2_single_correlation(tx,ty,t2,Lx,Ly,P,M,kxs,kys,W,correl_x,correl_y,correl_distance)
    N=Lx*Ly;

    C_T=2*W*W'-Matrix(I, P+4*M, P+4*M);
    A=C_T[1:P,1:P];
    B=C_T[1:P,P+1:P+4*M];
    D=C_T[P+1:P+4*M,P+1:P+4*M];

    #define pauli matrix
    sigmax=[0 1;1 0];
    sigmay=[0 -im;im 0];
    sigmaz=[1 0;0 -1];
    Pr1=[1;0];
    Pr2=[0;1];

    correl_uu_exact_left=correl_x[1];
    correl_ud_exact_left=correl_x[2];
    correl_du_exact_left=correl_x[3];
    correl_dd_exact_left=correl_x[4];
    correl_uu_exact_right=correl_x[5];
    correl_ud_exact_right=correl_x[6];
    correl_du_exact_right=correl_x[7];
    correl_dd_exact_right=correl_x[8];

    correl_uu_exact_up=correl_y[1];
    correl_dd_exact_up=correl_y[2];
    correl_uu_exact_down=correl_y[3];
    correl_dd_exact_down=correl_y[4];

    #energy
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


    ######################################
    #correlation functions
    #x direction correlations

    for dis_x=0:correl_distance
        correl_uu_right=0*(1+0*im);
        correl_ud_right=0*(1+0*im);
        correl_du_right=0*(1+0*im);
        correl_dd_right=0*(1+0*im);

        correl_uu_left=0*(1+0*im);
        correl_ud_left=0*(1+0*im);
        correl_du_left=0*(1+0*im);
        correl_dd_left=0*(1+0*im);
    
        for ca=1:Lx
            for cb=1:Ly

                kx=kxs[ca];
                ky=kys[cb];

                C_in_1=[zeros(M,M) exp(im*kx)*Matrix(I, M, M); exp(-im*kx)*Matrix(I, M, M) zeros(M,M)];
                C_in_2=[zeros(M,M) exp(im*ky)*Matrix(I, M, M); exp(-im*ky)*Matrix(I, M, M) zeros(M,M)];
                C_in=[C_in_1 zeros(2*M,2*M); zeros(2*M,2*M) C_in_2];

                C_out=A-B*pinv(D+C_in)*B';
                rho_out=(C_out+Matrix(I, P, P))/2;

                ob_uu_=Pr1'*rho_out*Pr1;
                ob_dd_=Pr2'*rho_out*Pr2;
                ob_ud_=Pr2'*rho_out*Pr1;
                ob_du_=Pr1'*rho_out*Pr2;


                correl_uu_right=correl_uu_right+exp(im*(dis_x)*kxs[ca])*ob_uu_/Lx/Ly;
                correl_dd_right=correl_dd_right+exp(im*(dis_x)*kxs[ca])*ob_dd_/Lx/Ly;
                correl_ud_right=correl_ud_right+exp(im*(dis_x)*kxs[ca])*ob_ud_/Lx/Ly;
                correl_du_right=correl_du_right+exp(im*(dis_x+1)*kxs[ca])*ob_du_/Lx/Ly;

                correl_uu_left=correl_uu_left+exp(-im*(dis_x)*kxs[ca])*ob_uu_/Lx/Ly;
                correl_dd_left=correl_dd_left+exp(-im*(dis_x)*kxs[ca])*ob_dd_/Lx/Ly;
                correl_ud_left=correl_ud_left+exp(-im*(dis_x+1)*kxs[ca])*ob_ud_/Lx/Ly;
                correl_du_left=correl_du_left+exp(-im*(dis_x)*kxs[ca])*ob_du_/Lx/Ly;
            end

        end
        #println("distance: "*string(dis_x))
        #correl_GfPEPS=[correl_uu_right,correl_dd_right,correl_ud_right,correl_du_right,correl_uu_left,correl_dd_left,correl_ud_left,correl_du_left];
        #println([correl_ud_right,correl_ud_exact_right[dis_x+1]])
        correl_exact=[correl_uu_exact_right[dis_x+1],correl_dd_exact_right[dis_x+1],correl_ud_exact_right[dis_x+1],correl_du_exact_right[dis_x+1],correl_uu_exact_left[dis_x+1],correl_dd_exact_left[dis_x+1],correl_ud_exact_left[dis_x+1],correl_du_exact_left[dis_x+1]];
        #println((correl_GfPEPS-correl_exact))
        #println(norm(correl_GfPEPS-correl_exact))
        #println([correl_ud_right-correl_ud_exact_right[dis_x+1]])

        E=E+100*real(dot(correl_ud_right-correl_ud_exact_right[dis_x+1],correl_ud_right-correl_ud_exact_right[dis_x+1]));

    end
    

    return E
end
