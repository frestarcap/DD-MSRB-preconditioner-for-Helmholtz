function export_halfspace_frequency_figure
% Frozen DD algebraic field for the frequency-study illustration.
root=helm_output_path('outputs','reviewer_20260909','frequency_4');
out=helm_output_path('outputs','halfspace_figures_20260910');
s=load(fullfile(root,'problem.mat'),'ops','b','p');d=load(fullfile(root,'dd.mat'),'model');
mu=[1,.8761];A=s.ops{1}+(mu(2)^(-2)-1)*s.ops{2};
u=rom_replay_inverse_action(A,s.b,mu,d.model);rho=norm(s.b-A*u)/norm(s.b);assert(rho<=1e-10);
ref=A\s.b;relative_error=norm(u-ref)/norm(ref);assert(relative_error<1e-8);
x=s.p.x_int;y=s.p.y_int;U=reshape(u,s.p.nx,s.p.ny).';omega=s.p.omega;
weights=abs(reshape(s.b,s.p.nx,s.p.ny));total=sum(weights(:));
source=[x.'*sum(weights,2)/total, sum(weights,1)*y/total];
assert(norm(source-[.2,.5],Inf)<s.p.hx);
save(fullfile(out,'omega214_field.mat'),'U','x','y','omega','mu','rho','relative_error','source','-v7');
fprintf('FIELD_PASS omega=%g mu=%g rho=%.12g reference_error=%.12g\n',omega,mu(2),rho,relative_error);
end
