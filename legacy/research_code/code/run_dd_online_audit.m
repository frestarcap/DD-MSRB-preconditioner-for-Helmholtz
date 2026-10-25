function rows=run_dd_online_audit
% Frozen-model component timings; no retraining or changed accuracy targets.
root=helm_output_path('outputs','reviewer_20260909');
out=helm_output_path('outputs','dd_online_audit_20260910');
if ~isfolder(out),mkdir(out);end
addpath(out);cleanup=onCleanup(@()rmpath(out)); %#ok<NASGU>
rows=table();
for scale=[1 1.5 2 4]
 folder=fullfile(root,sprintf('frequency_%g',scale));
 p=load(fullfile(folder,'problem.mat'),'ops','b','c');
 models=cell(1,2);
 for v=1:2
  names={'global','dd'};d=load(fullfile(folder,[names{v} '.mat']),'model');models{v}=d.model;
 end
 for qi=1:size(p.c.queries,1)
  mu=p.c.queries(qi,:);alpha=mu(2)^(-2)-1;A=p.ops{1}+alpha*p.ops{2};
  for rep=0:8
   order=1:2;if mod(rep,2),order=2:-1:1;end
   for v=order
    for opt=0:1
     [x,t]=replay_profile(A,p.b,alpha,models{v},opt);
     if opt==0,xref=x;else,assert(norm(x-xref)/norm(xref)<1e-12);end
     rho=norm(p.b-A*x)/norm(p.b);
     if rep>0
      rows=[rows;table(scale,mu(2),string(names{v}),opt,rep,t(1),t(2),t(3),t(4),t(5),t(6),t(7),t(8),rho, ...
       'VariableNames',{'scale','c_right','variant','optimized','repetition','total_s','assemble_s','project_s','solve_s','lift_s','arnoldi_s','reconstruct_s','residual_s','post_ROM_residual'})]; %#ok<AGROW>
     end
    end
   end
  end
  fprintf('scale %g query %.4f complete\n',scale,mu(2));
  writetable(rows,fullfile(out,'components.csv'));
 end
end
end
function [x,t]=replay_profile(A,b,alpha,model,opt)
t=zeros(1,8);all=tic;n=numel(b);beta=norm(b);k=numel(model);
Q=complex(zeros(n,k+1));Q(:,1)=b/beta;Z=complex(zeros(n,k));H=complex(zeros(k+1,k));
for j=1:k
 l=model{j};tt=tic;B=l.operators{1}+alpha*l.operators{2};t(2)=t(2)+toc(tt);
 tt=tic;r=rom_reduced_space_utils_before('project',l.space,Q(:,j));t(3)=t(3)+toc(tt);
 tt=tic;a=B\r;t(4)=t(4)+toc(tt);
 tt=tic;
 if opt && strcmp(l.space.type,'blockn')
  sp=l.space;z=zeros(sp.n,1,'like',a);cur=0;
  for ib=1:numel(sp.blocks)
   bl=sp.blocks(ib);kb=bl.k;
   if kb>0,z(bl.idx)=bl.basis*a(cur+(1:kb));cur=cur+kb;end
  end
 else,z=rom_reduced_space_utils_before('lift',l.space,a);end
 t(5)=t(5)+toc(tt);
 Z(:,j)=z;tt=tic;
 [Q(:,j+1),H(1:j,j),H(j+1,j),stop]=rom_arnoldi_step(A*z,Q(:,1:j));t(6)=t(6)+toc(tt);
 if ~opt || j==k || stop
  tt=tic;g=zeros(j+1,1);g(1)=beta;x=Z(:,1:j)*(H(1:j+1,1:j)\g);t(7)=t(7)+toc(tt);
 end
 if stop,break;end
end
tt=tic;rho=norm(b-A*x)/beta;t(8)=toc(tt);assert(isfinite(rho));t(1)=toc(all);
end
