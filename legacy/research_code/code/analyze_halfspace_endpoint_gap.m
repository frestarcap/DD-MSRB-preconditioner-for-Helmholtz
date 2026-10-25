function analyze_halfspace_endpoint_gap
% Diagnose frozen Global/DD spaces without changing training or paper evidence.
maxNumCompThreads(1);
s=load(helm_output_path('outputs','jcp_shifted_20260914','frozen_cohorts.mat'),'cohorts'); c=s.cohorts{1};
out=helm_output_path('outputs','halfspace_endpoint_gap_20260922'); if ~exist(out,'dir'),mkdir(out);end
layers=table(); actions=table(); regions=table(); hybrids=table(); numerical=table();
for qi=1:numel(c.parameters)
 mu=c.parameters(qi); A=c.A_base+(mu^(-2)-1)*c.A_right; b=c.b; nb=norm(b); F=decomposition(A,'lu');
 states=cell(1,2);
 for mi=1:2
  [u,h,Q,Z]=trajectory(A,b,mu,c.models{mi},false); states{mi}=struct('u',u,'h',h,'Q',Q,'Z',Z);
  assert(norm(u-c.expected{mi}(:,qi))/max(norm(u),eps)<1e-9);
  for j=1:3
   method=string(c.names{mi}); rho=h(j+1); contraction=h(j+1)/h(j);
   layers=[layers;table(qi,mu,method,j,rho,contraction)]; %#ok<AGROW>
  end
 end
 % Compare both spaces on each identical source Arnoldi vector.
 for source=1:2
  for j=1:3
   q=states{source}.Q(:,j); exact=F\q;
   for target=1:2
    l=c.models{target}.layers{j}; sp=l.space;
    [z,Ar,br,coef]=inverse(A,q,mu,l,false);
    proj=rom_reduced_space_utils('lift',sp,rom_reduced_space_utils('project',sp,exact));
    projection_error=norm(exact-proj)/norm(exact); action_error=norm(exact-z)/norm(exact);
    action_residual=norm(q-A*z); cond_reduced=cond(Ar);
    backward=norm(Ar*coef-br)/(norm(Ar)*norm(coef)+norm(br));
    source_name=string(c.names{source}); target_name=string(c.names{target});
    actions=[actions;table(qi,mu,source_name,target_name,j,projection_error,action_error,action_residual,cond_reduced,backward)]; %#ok<AGROW>
    if target==2 && ismember(qi,[1,2,10,22,23])
     e=exact-proj; r=q-A*z;
     for bi=1:numel(sp.blocks)
      ix=sp.blocks(bi).idx; label=string(sp.blocks(bi).label); rank=sp.blocks(bi).k;
      proj_energy=sum(abs(e(ix)).^2)/norm(exact)^2; residual_energy=sum(abs(r(ix)).^2)/norm(r)^2;
      regions=[regions;table(qi,source_name,j,bi,label,rank,proj_energy,residual_energy)]; %#ok<AGROW>
     end
    end
   end
  end
 end
 if ismember(qi,[1,2,10,22,23])
  for mask=0:7
   m=c.models{1}; bits=bitget(mask,3:-1:1); label='GGG';
   for j=1:3,if bits(j),m.layers{j}=c.models{2}.layers{j};label(j)='D';end,end
   [~,h]=trajectory(A,b,mu,m,false); pattern=string(label); rho1=h(2);rho2=h(3);rho3=h(4);
   hybrids=[hybrids;table(qi,mu,pattern,rho1,rho2,rho3)]; %#ok<AGROW>
  end
  for mi=1:2
   st=states{mi}; [ufresh,~]=trajectory(A,b,mu,c.models{mi},true);
   W=A*st.Z; y=W\b; uls=st.Z*y;
   correction=W\(b-A*st.u); urefine=st.u+st.Z*correction;
   method=string(c.names{mi}); original=st.h(end); fresh=norm(b-A*ufresh)/nb;
   full_ls=norm(b-A*uls)/nb; refine_ls=norm(b-A*urefine)/nb;
   ref_residual=norm(b-A*c.references(:,qi))/nb;
   numerical=[numerical;table(qi,method,original,fresh,full_ls,refine_ls,ref_residual)]; %#ok<AGROW>
  end
 end
 fprintf('GAP query=%d/23 complete\n',qi);
end
writetable(layers,fullfile(out,'layers.csv'));writetable(actions,fullfile(out,'actions.csv'));
writetable(regions,fullfile(out,'regions.csv'));writetable(hybrids,fullfile(out,'hybrids.csv'));writetable(numerical,fullfile(out,'numerical.csv'));
save(fullfile(out,'diagnostics.mat'),'layers','actions','regions','hybrids','numerical');
end
function [u,h,Q,Z]=trajectory(A,b,mu,model,fresh)
n=numel(b); nb=norm(b); Q=zeros(n,4);Z=zeros(n,3);H=zeros(4,3);Q(:,1)=b/nb;h=1;
for j=1:3
 Z(:,j)=inverse(A,Q(:,j),mu,model.layers{j},fresh);
 [Q(:,j+1),H(1:j,j),H(j+1,j)]=rom_arnoldi_step(A*Z(:,j),Q(:,1:j));
 g=zeros(j+1,1);g(1)=nb;y=H(1:j+1,1:j)\g;u=Z(:,1:j)*y;h(end+1)=norm(b-A*u)/nb;
end
end
function [z,Ar,br,coef]=inverse(A,q,mu,l,fresh)
sp=l.space;br=rom_reduced_space_utils('project',sp,q);
if fresh,Ar=full(rom_reduced_space_utils('assemble_reduced_matrix',A,sp));else,Ar=l.Ar_base+(mu^(-2)-1)*l.Ar_right;end
coef=Ar\br;z=rom_reduced_space_utils('lift',sp,coef);
end
