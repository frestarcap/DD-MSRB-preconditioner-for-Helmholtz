function analyze_halfspace_endpoint_geometry
maxNumCompThreads(1);
s=load(helm_output_path('outputs','jcp_shifted_20260914','frozen_cohorts.mat'),'cohorts');c=s.cohorts{1};
out=helm_output_path('outputs','halfspace_endpoint_gap_20260922');
a=load(helm_output_path('outputs','halfspace_dd_refinement','review_repaired_20260905','global.mat'),'artifact');g=a.artifact;
a=load(helm_output_path('outputs','halfspace_dd_refinement','review_repaired_20260905','slabs1-aligned-sym8.mat'),'artifact');d=a.artifact;
f=fopen(fullfile(out,'metadata.txt'),'w');
for j=1:3
 fprintf(f,'layer %d global rank %d DD rank %d; snapshot sizes G=%s D=%s\n',j,c.models{1}.layers{j}.space.k,c.models{2}.layers{j}.space.k,mat2str(size(g.layers{j}.eta_snapshots)),mat2str(size(d.layers{j}.eta_snapshots)));
 for bi=1:10
  v=d.layers{j}.basis_space.blocks(bi);m=v.meta;
  fprintf(f,'%s rank %d localtail %.5e globalweight %.5e globaltail %.5e\n',v.label,v.k,m.local_relative_tail,m.global_energy_weight,m.global_relative_tail);
 end
end
if isequal(size(g.layers{1}.eta_snapshots),size(d.layers{1}.eta_snapshots))
 fprintf(f,'first snapshot relative difference %.5e\n',norm(g.layers{1}.eta_snapshots-d.layers{1}.eta_snapshots,'fro')/norm(g.layers{1}.eta_snapshots,'fro'));
end
fclose(f);
regions=table(); n=numel(c.b); sp=c.models{2}.layers{1}.space;labels=zeros(n,1);
for bi=1:10,labels(sp.blocks(bi).idx)=bi;end
for qi=[1,2,10,22,23]
 mu=c.parameters(qi); A=c.A_base+(mu^(-2)-1)*c.A_right;
 [ii,jj]=find(A);cross=labels(ii)~=labels(jj);boundary=false(n,1);boundary(unique([ii(cross);jj(cross)]))=true;
 for mi=1:2
  model=c.models{mi};[u,h,Q,Z]=trajectory(A,c.b,mu,model,false); %#ok<ASGLU>
  for j=[1,3]
   z=inverse(A,Q(:,j),mu,model.layers{j},false);r=Q(:,j)-A*z;
   if j==1
    z1=Z(:,1); az=A*z1; outer=c.b-az*((az'*c.b)/(az'*az));
   else,outer=c.b-A*u;end
   outer_boundary_energy=norm(outer(boundary))^2/norm(outer)^2;
   method=string(c.names{mi});boundary_fraction=mean(boundary);residual_boundary_energy=norm(r(boundary))^2/norm(r)^2;
   regions=[regions;table(qi,method,j,boundary_fraction,residual_boundary_energy,outer_boundary_energy)]; %#ok<AGROW>
  end
 end
end
writetable(regions,fullfile(out,'boundary.csv'));
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
