function verify_dd_field_response(outroot,case_ids)
% Reuse all saved models/references; measure field-error, not just residual,
% contraction after the same 20-step MG-GMRES direction experiment.
if nargin<2,case_ids=1:6;end
maxNumCompThreads(2); rows=struct([]);outfile=fullfile(outroot,'field_response.csv');
if isfile(outfile),rows=table2struct(readtable(outfile));rows=rows(~ismember([rows.case_id],case_ids));end
for cid=case_ids
 folder=fullfile(outroot,'results',sprintf('case_%02d',cid));
 protocol=jsondecode(fileread(fullfile(folder,'protocol.json')));
 original=jsondecode(fileread(fullfile(folder,'mechanisms.json')));
 p=helmholtz_3d_problem_interface(protocol.cfg);b=p.rhs(.4);
 saved=load(fullfile(folder,'models_narrow5.mat'),'models','c');
 for qi=1:2
  ref=load(fullfile(folder,sprintf('reference_narrow5_q%d.mat',qi)));mu=ref.mu;
  A=p.K_pml+spdiags(p.affine.mass_base+mu*p.affine.mass_parameter_diagonal,0,p.N,p.N);
  mass=p.affine.mass_base+mu*p.affine.mass_parameter_diagonal;
  [M,~]=build_shifted_laplacian_preconditioner(A+2i*spdiags(mass,0,p.N,p.N),protocol.mg_options);
  opts=protocol.gmres_options;opts.restart=20;opts.maxit=1;opts.tol=1e-14;
  for mi=1:numel(saved.models)
   m=saved.models{mi};
   if ~((m.fixed&&m.tail==0)||(~m.fixed&&m.tail==.003)),continue;end
   x=initial(p,b,m,mu);r=b-A*x;e=ref.uref-x;rn=norm(r);
   [delta,~]=shifted_laplacian_gmres(@(v)A*v,M,r/rn,[],opts);
   postr=norm(r/rn-A*delta);postfield=norm(e-rn*delta)/norm(e);
   z=M(r);az=A*z;alpha=(az'*r)/(az'*az);
   onefield=norm(e-alpha*z)/norm(e);
   stationaryfield=norm(e-z)/norm(e);
   idx=find(strcmp({original.method},m.name)&abs([original.mu]-mu)<1e-12,1);
   assert(~isempty(idx)&&abs(postr-original(idx).normalized20)<1e-8);
   row=struct('case_id',cid,'mu',mu,'method',m.name,'field_contraction1',onefield, ...
    'field_contraction20',postfield,'residual_contraction20',postr, ...
    'stationary_MG_field_contraction',stationaryfield, ...
    'reference_residual_over_handoff',norm(b-A*ref.uref)/rn);
   if isempty(rows),rows=row;else,rows(end+1)=row;end %#ok<AGROW>
  end
 end
 writetable(struct2table(rows),outfile);
end
for cid=case_ids,assert(nnz([rows.case_id]==cid)==8);end
fprintf('FIELD_RESPONSE_PASS rows=%d\n',numel(rows));
end
function [z,c]=act(l,mu,q)
c=(l.Ar0+mu*l.Ar1)\rom_reduced_space_utils('project',l.space,q);
z=rom_reduced_space_utils('lift',l.space,c);
end
function x=initial(p,b,m,mu)
if m.fixed,x=act(m.layers{1},mu,b);return;end
n=p.N;nb=norm(b);Q=complex(zeros(n,4));Z=complex(zeros(n,3));H=complex(zeros(4,3));Q(:,1)=b/nb;
for j=1:3
 z=act(m.layers{j},mu,Q(:,j));Z(:,j)=z;v=p.K_pml*z+(p.affine.mass_base+mu*p.affine.mass_parameter_diagonal).*z;
 for pass=1:2,for i=1:j,a=Q(:,i)'*v;H(i,j)=H(i,j)+a;v=v-Q(:,i)*a;end,end
 H(j+1,j)=norm(v);g=zeros(j+1,1);g(1)=nb;x=Z(:,1:j)*(H(1:j+1,1:j)\g);
 if H(j+1,j)<1e-13,break;end
 Q(:,j+1)=v/H(j+1,j);
end
end
