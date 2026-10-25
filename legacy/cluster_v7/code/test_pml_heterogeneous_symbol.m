function test_pml_heterogeneous_symbol
%TEST_PML_HETEROGENEOUS_SYMBOL Check the six-state target symbol against A_h.
    rng(43); nx=9; ny=8; hx=0.1; hy=0.12; omega=12; dt=0.01;
    cw=sqrt(1-(omega*dt/2)^2); d=-1i*omega;
    sx=[5;3;1;zeros(nx-4,1);1;3;5];
    sy=[5;2;zeros(ny-2,1);2;5];
    sxv=(sx(1:end-1)+sx(2:end))/2;
    syv=(sy(1:end-1)+sy(2:end))/2;
    pml=struct('s_x',1+1i*sx/omega,'s_y',1+1i*sy/omega, ...
        's_x_half',1+1i*sxv/omega,'s_y_half',1+1i*syv/omega);
    Gx=kron(speye(ny),sparse(diff([zeros(1,nx);eye(nx);zeros(1,nx)]))/hx);
    Gy=kron(sparse(diff([zeros(1,ny);eye(ny);zeros(1,ny)]))/hy,speye(nx));
    Bx=-Gx'; By=-Gy';
    ax=repmat(sx(2:end-1),ny,1); ay=kron(sy(2:end-1),ones(nx,1));
    avx=repmat(sxv,ny,1); avy=kron(syv,ones(nx,1));
    c=0.6+rand(nx*ny,1); U=randn(nx*ny,1)+1i*randn(nx*ny,1);
    A=pml_stiffness_2d(nx,ny,hx,hy,pml)+spdiags(omega^2./c.^2,0,nx*ny,nx*ny);
    f=A*U; Vx=(Gx*U)./(d+avx); Vy=(Gy*U)./(d+avy);
    psi1=ay.*(Bx*Vx)/d; psi2=ax.*(By*Vy)/d;
    theta=(ax/cw).*(ay/cw)*cw^2.*U./c.^2/d;
    stretch=(1+ax/d).*(1+ay/d); forcing=-stretch.*f/d;
    lhs=d*U./c.^2;
    rhs=Bx*Vx+By*Vy-(ax+ay).*U./c.^2-theta+psi1+psi2+forcing;
    rel=norm(lhs-rhs)/norm(lhs);
    assert(rel<1e-12,'Heterogeneous PML symbol does not match the assembled operator.');
    fprintf('Heterogeneous six-state PML symbol relative discrepancy %.3e\n',rel);
end
