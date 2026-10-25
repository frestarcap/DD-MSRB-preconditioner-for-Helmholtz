function run_dd_parameter_mechanism_pilot(case_id,outroot)
% Independent small 3D study. Original solvers/data/manuscript are untouched.
% Training inverse actions use the exact Arnoldi recurrence, validated by A.
if nargin<1,case_id=1;end
if nargin<2,outroot=helm_output_path('outputs','dd_parameter_mechanism_20260915','results');end
outdir=fullfile(outroot,sprintf('case_%02d',case_id));
if ~isfolder(outdir),mkdir(outdir);end
assert(~isfile(fullfile(outdir,'complete.json')),'Completed case exists.');
maxNumCompThreads(4); rng(915,'twister'); timer=tic;
pcfg=struct('material_model','gaussian_waveguide','domain_lengths',[3 1 1], ...
 'Nphys',[71 23 23],'npml',[6 6 6],'omega',12,'sigma_max',80, ...
 'waveguide_axis','x','waveguide_center',[1.5 .5 .5],'waveguide_width',[.14 .14], ...
 'source_model','gaussian','source_location',[.2 .5 .5],'source_sigma',.0625);
switch case_id
 case 0,pcfg.Nphys=[35 11 11];pcfg.npml=[3 3 3];pcfg.omega=6;
 case {1,101,102}
 case 2,pcfg.omega=6;
 case 3,pcfg.omega=9;
 case 4,pcfg.domain_lengths=[6 1 1];pcfg.Nphys=[143 23 23];pcfg.waveguide_center=[3 .5 .5];
 case 5,pcfg.material_model='gaussian_lens';pcfg.domain_lengths=[1 1 1];pcfg.Nphys=[23 23 23];pcfg.gaussian_center=[.55 .5 .5];pcfg.gaussian_width=[.15 .18 .18];pcfg.omega=12;
 case 6,pcfg.Nphys=[95 31 31];pcfg.npml=[8 8 8];
 otherwise,error('Unknown case');
end
p=helmholtz_3d_problem_interface(pcfg);b=p.rhs(.4);nb=norm(b);
cfg=struct('dd_partition_mode','signed_depth_pml','physical_partition_mode','profile_bands', ...
 'profile_partition_thresholds',[.001 .01 .1 .5],'physical_axial_sections',5,'pml_depth_fraction',.5);
part=pilot_make_rom_partition(p,.4,cfg);
mgopts=struct('solver','mg','grid_size',p.n,'mg_max_levels',6,'mg_coarse_max_dofs',1000, ...
 'mg_pre_sweeps',2,'mg_post_sweeps',2,'mg_jacobi_weight',.6,'mg_cycles_per_apply',1);
gopts=struct('restart',40,'maxit',25,'tol',1e-6,'reorthogonalize',true, ...
 'orthogonalization','cgs2','store_preconditioned_basis',false,'fixed_linear_preconditioner',true,'verbose',false);
cohorts=struct('name','narrow5','train',.2:.1:.6,'queries',[.314 .494]);
if ismember(case_id,[1 101 102])
 cohorts(2)=struct('name','medium9','train',0:.1:.8,'queries',[.228 .588]);
 cohorts(3)=struct('name','wide15','train',0:.1:1.4,'queries',[.399 1.029]);
 cohorts(4)=struct('name','wide5','train',linspace(0,1.4,5),'queries',[.399 1.029]);
end
if case_id==102
 cohorts(1).queries=[.25 .55];cohorts(2).queries=[.25 .65];
 cohorts(3).queries=[.45 1.15];cohorts(4).queries=[.45 1.15];
end
pool=uniquetol([cohorts.train],1e-12); S=complex(zeros(p.N,numel(pool)));refmeta=struct([]);new_training_solves=0;
writejson(fullfile(outdir,'protocol.json'),struct('case_id',case_id,'cfg',pcfg,'N',p.N,'blocks',numel(part.blocks), ...
 'cohorts',cohorts,'mg_options',mgopts,'gmres_options',gopts,'threads',4,'matlab',version, ...
 'timing_scope','Single-run diagnostics; shared reference solves reported separately. Model online includes ROM, MG setup, completion and verification; common operator assembly excluded.'));
for k=1:numel(pool)
 fn=fullfile(outdir,sprintf('reference_train_%02d.mat',k)); mu=pool(k);
 if ismember(case_id,[101 102]),fn=fullfile(outroot,'case_01',sprintf('reference_train_%02d.mat',k));assert(isfile(fn));end
 if isfile(fn),s=load(fn);assert(abs(s.mu-mu)<1e-12);u=s.u;row=s.row;
 else
  fprintf('REFERENCE case=%d %d/%d mu=%.5f\n',case_id,k,numel(pool),mu);
  [u,row]=reference(p,b,mu,mgopts,gopts);save(fn,'u','mu','row','-v7.3');new_training_solves=new_training_solves+1;
 end
 assert(norm(b-applyA(p,mu,u))/nb<2e-11);S(:,k)=u/nb;refmeta=append(refmeta,row);
end
writejson(fullfile(outdir,'reference_training.json'),refmeta);
rows=struct([]);modelrows=struct([]);mechanisms=struct([]);histories=struct([]);trainchecks=struct([]);
for ci=1:numel(cohorts)
 c=cohorts(ci);idx=arrayfun(@(mu)find(abs(pool-mu)<1e-12,1),c.train);T=S(:,idx);
 if case_id==1,tails=[.03 .003 .0003 0];tat=[.03 .003 .0003];elseif ismember(case_id,[101 102]),tails=[1e-5 1e-7];tat=[];else,tails=0;tat=.003;end
 models={};
 if case_id==102
  previous=load(fullfile(outroot,'case_01',['models_' c.name '.mat']),'models');
  for pi=1:numel(previous.models)
   pm=previous.models{pi};
   if strcmp(pm.style,'dd') && ((pm.fixed&&pm.tail==0)||(~pm.fixed&&ismember(pm.tail,[.003 .0003])))
    models{end+1}=pm;
   end
  end
 end
 for tol=tails
  models{end+1}=fixedmodel(p,T,part,'dd',tol); %#ok<AGROW>
 end
 for tol=tat
  [models{end+1},check]=trainmodel(p,T,c.train,b,part,'dd',tol,mgopts,gopts); %#ok<AGROW>
  check.cohort=c.name;trainchecks=append(trainchecks,check);
 end
 if ci==1 && ~ismember(case_id,[101 102])
  models{end+1}=fixedmodel(p,T,part,'global',0);
  [models{end+1},check]=trainmodel(p,T,c.train,b,part,'global',.003,mgopts,gopts);
  check.cohort=c.name;trainchecks=append(trainchecks,check);
 end
 for mi=1:numel(models)
  model=models{mi};modelrows=append(modelrows,struct('case_id',case_id,'cohort',c.name,'method',model.name, ...
   'samples',numel(c.train),'payload_MiB',model.payload/2^20,'basis_MiB',model.basisbytes/2^20, ...
   'operator_MiB',model.opbytes/2^20,'layer_ranks',model.ranks,'build_s',model.build_s, ...
   'tail',model.tail,'snapshot_projection_error',model.capture));
 end
 writejson(fullfile(outdir,'models.json'),modelrows);writejson(fullfile(outdir,'training_checks.json'),trainchecks);
 if case_id~=102,save(fullfile(outdir,['models_' c.name '.mat']),'models','c','part','-v7.3');end
 for qi=1:numel(c.queries)
  mu=c.queries(qi);A=matrixA(p,mu);mass=p.affine.mass_base+mu*p.affine.mass_parameter_diagonal;
  tt=tic;[M,probe,mginfo]=build_dd_mechanism_mg_probe(A+2i*spdiags(mass,0,p.N,p.N),p.n,mgopts);setup_s=toc(tt);
  if case_id==101
   rf=load(fullfile(outroot,'case_01',sprintf('reference_%s_q%d.mat',c.name,qi)));assert(abs(rf.mu-mu)<1e-12);uref=rf.uref;refrow=rf.refrow;
  else
   [uref,refrow]=reference(p,b,mu,mgopts,gopts);save(fullfile(outdir,sprintf('reference_%s_q%d.mat',c.name,qi)),'uref','mu','refrow','-v7.3');
  end
  order=circshift(1:numel(models),[0 qi-1]);
  for pos=1:numel(order)
   mi=order(pos);model=models{mi};tt=tic;[x,trace,coeffs]=online(p,b,model,mu);rom_s=toc(tt);
   r=b-A*x;rho=norm(r)/nb;e=uref-x;
   [done,infog]=shifted_laplacian_gmres(@(v)A*v,M,b,x,gopts);
   tt=tic;tr=norm(b-A*done)/nb;verify_s=toc(tt);assert(tr<=1.001e-6,'Completion failed');
   row=struct('case_id',case_id,'cohort',c.name,'mu',mu,'method',model.name,'payload_MiB',model.payload/2^20, ...
    'pre_residual',rho,'field_error',norm(e)/norm(uref),'iterations',infog.iter_count,'true_residual',tr, ...
    'rom_s',rom_s,'setup_s',setup_s,'completion_s',infog.total_time,'online_s',rom_s+setup_s+infog.total_time+verify_s);
   rows=append(rows,row);histories=append(histories,struct('cohort',c.name,'mu',mu,'method',model.name,'rom',trace,'completion',infog.relres_hist));
   fprintf('ONLINE c%d %s mu=%.4f %s MB=%.3f r=%.3e it=%d\n',case_id,c.name,mu,model.name,model.payload/2^20,rho,infog.iter_count);
   % Direction diagnostics for the same four families; fixed bases are complete.
   if ci==1 && case_id~=102 && ((model.fixed && model.tail==0)||(~model.fixed && model.tail==.003))
    opts=gopts;opts.restart=20;opts.maxit=1;opts.tol=1e-14;
    [d,di]=shifted_laplacian_gmres(@(v)A*v,M,r/norm(r),[],opts);
    rr=r/norm(r)-A*d;ps=probe(r,@(v)A*v);assert(ps.relative_sum_vs_Msolve_error<1e-10);
    metrics=fieldmetrics(r,p,part); emetrics=fieldmetrics(e,p,part);
    rec=struct('case_id',case_id,'mu',mu,'method',model.name,'rho',rho,'field_error',norm(e)/norm(uref), ...
     'reference_over_handoff',norm(b-A*uref)/norm(r),'normalized20',norm(rr),'history',di.relres_hist, ...
     'residual_metrics',metrics,'error_metrics',emetrics,'mg',ps,'decomposition',struct(),'interventions',struct([]));
    if strcmp(model.style,'dd') && qi==1
     rec.decomposition=commutator_probe(p,A,b,model,coeffs,x,r,M,opts,part);
    end
    if qi==1 && case_id~=0
     % Isolate smoother sensitivity and shift sensitivity; same r,A,20 steps.
     variants=[.5 2;1 2;2 0;2 4];
     for vi=1:size(variants,1)
      mo=mgopts;mo.mg_pre_sweeps=variants(vi,2);mo.mg_post_sweeps=variants(vi,2);
      [Mv,~,~]=build_dd_mechanism_mg_probe(A+1i*variants(vi,1)*spdiags(mass,0,p.N,p.N),p.n,mo);
      [dv,iv]=shifted_laplacian_gmres(@(v)A*v,Mv,r/norm(r),[],opts);
      rec.interventions=append(rec.interventions,struct('beta',variants(vi,1),'sweeps',variants(vi,2),'rho20',norm(r/norm(r)-A*dv),'iterations',iv.iter_count,'history',iv.relres_hist));
     end
    end
    mechanisms=append(mechanisms,rec);writejson(fullfile(outdir,'mechanisms.json'),mechanisms);
   end
   writetable(struct2table(rows),fullfile(outdir,'online.csv'));writejson(fullfile(outdir,'histories.json'),histories);
  end
 end
 clear models T
end
result=struct('passed',true,'case_id',case_id,'N',p.N,'training_solves',numel(pool),'new_training_solves',new_training_solves,'formal_queries',sum(arrayfun(@(z)numel(z.queries),cohorts)), ...
 'online_solves',numel(rows),'elapsed_s',toc(timer),'matlab',version,'threads',4,'max_true_residual',max([rows.true_residual]));
writejson(fullfile(outdir,'complete.json'),result);fprintf('DD_PARAMETER_MECHANISM_PASS case=%d time=%.3f\n',case_id,result.elapsed_s);
end

function [u,row]=reference(p,b,mu,mo,go)
t=tic;A=matrixA(p,mu);mass=p.affine.mass_base+mu*p.affine.mass_parameter_diagonal;
[M,~]=build_shifted_laplacian_preconditioner(A+2i*spdiags(mass,0,p.N,p.N),mo);
go.tol=5e-12;go.maxit=50;[u,infog]=shifted_laplacian_gmres(@(v)A*v,M,b,[],go);
rho=norm(b-A*u)/norm(b);assert(rho<2e-11,'Reference solve failed');
row=struct('mu',mu,'true_residual',rho,'iterations',infog.iter_count,'seconds',toc(t));
end

function model=fixedmodel(p,S,part,style,tol)
t=tic;l=layer(p,S,part,style,tol);model=pack({l},style,true,tol,toc(t));
end

function [model,check]=trainmodel(p,S,mus,b,part,style,tol,mo,go)
t=tic;ns=numel(mus);n=p.N;nb=norm(b);Q=complex(zeros(n,4,ns));E=complex(zeros(n,3,ns));Q(:,1,:)=repmat(b/nb,1,1,ns);
snap=S;layers={};checks=[];fallbacks=0;
for j=1:3
 for k=1:ns
  q=Q(:,j,k);if norm(q)<1e-12,snap(:,k)=0;continue;end
  defect=norm(q-applyA(p,mus(k),snap(:,k)))/norm(q);
  if defect>1e-7
   [snap(:,k),~]=reference(p,q,mus(k),mo,go);fallbacks=fallbacks+1;
   defect=norm(q-applyA(p,mus(k),snap(:,k)))/norm(q);
  end
  assert(defect<=1e-7);checks(end+1)=defect; %#ok<AGROW>
 end
 E(:,j,:)=reshape(snap,n,1,ns);factors=[1 8/3 20/3];l=layer(p,snap,part,style,tol*factors(j));
 layers{j}=l;next=complex(zeros(n,ns));
 if j<3
  for k=1:ns
   q=Q(:,j,k);if norm(q)<1e-12,continue;end
   z=action(l,mus(k),q);v=applyA(p,mus(k),z);h=complex(zeros(j,1));
   for pass=1:2,for i=1:j,a=Q(:,i,k)'*v;h(i)=h(i)+a;v=v-Q(:,i,k)*a;end,end
   hn=norm(v);
   if hn>1e-10,Q(:,j+1,k)=v/hn;next(:,k)=(z-E(:,1:j,k)*h)/hn;end
  end
 end
 snap=next;
end
model=pack(layers,style,false,tol,toc(t));
check=struct('style',style,'tail',tol,'max_inverse_defect',max(checks),'fallback_solves',fallbacks,'cohort','');
end

function l=layer(p,S,part,style,tol)
if strcmp(style,'global')
 [U,D,~]=svd(S,'econ');s=diag(D);k=selectrank(s,tol);sp=rom_reduced_space_utils('make_dense_space',U(:,1:k));bb=16*numel(sp.basis);
else
 cfg=struct('rank_rtol',1e-13);
 if tol>0,cfg.rank_global_tail_tol=tol;cfg.rank_global_tail_strategy='marginal_storage';end
 sp=rom_reduced_space_utils('build_blockn_space',S,part,cfg);bb=16*sum(arrayfun(@(z)numel(z.basis),sp.blocks));
end
[a0,a1]=assemble_3d_rom_affine_sparse(p,sp);
v=complex(sin((1:sp.k).'),cos((1:sp.k).'));u=rom_reduced_space_utils('lift',sp,v);
proj=rom_reduced_space_utils('project',sp,applyA(p,.431,u));assert(norm(proj-(a0+.431*a1)*v)/max(norm(proj),eps)<1e-9);
err=0;for k=1:size(S,2),r=S(:,k)-rom_reduced_space_utils('lift',sp,rom_reduced_space_utils('project',sp,S(:,k)));err=err+norm(r)^2;end
l=struct('space',sp,'Ar0',a0,'Ar1',a1,'rank',sp.k,'bb',bb,'S',S,'capture',sqrt(err)/max(norm(S,'fro'),realmin));
end

function k=selectrank(s,tol)
if isempty(s)||s(1)==0,k=0;return;end
if tol==0,k=sum(s>1e-13*s(1));return;end
after=[flipud(cumsum(flipud(s(2:end).^2)));0];k=find(sqrt(after)/norm(s)<=tol,1);if isempty(k),k=numel(s);end
end
function m=pack(layers,style,fixed,tol,secs)
bb=sum(cellfun(@(z)z.bb,layers));op=16*sum(cellfun(@(z)numel(z.Ar0)+numel(z.Ar1),layers));idx=0;
if strcmp(style,'dd'),idx=8*layers{1}.space.n;end
if fixed,kind='fixed';else,kind='TA3';end
m=struct('layers',{layers},'style',style,'fixed',fixed,'tail',tol,'name',sprintf('%s_%s_t%.0e',style,kind,tol), ...
 'payload',bb+op+idx,'basisbytes',bb,'opbytes',op,'ranks',cellfun(@(z)z.rank,layers),'build_s',secs,'capture',cellfun(@(z)z.capture,layers));
end
function [z,c]=action(l,mu,q)
c=(l.Ar0+mu*l.Ar1)\rom_reduced_space_utils('project',l.space,q);z=rom_reduced_space_utils('lift',l.space,c);
end
function [x,hist,coeffs]=online(p,b,m,mu)
if m.fixed,[x,c]=action(m.layers{1},mu,b);hist=[1;norm(b-applyA(p,mu,x))/norm(b)];coeffs={c};return;end
n=p.N;nb=norm(b);Q=complex(zeros(n,4));Z=complex(zeros(n,3));H=complex(zeros(4,3));Q(:,1)=b/nb;hist=1;raw=cell(1,3);
for j=1:3
 [z,raw{j}]=action(m.layers{j},mu,Q(:,j));Z(:,j)=z;v=applyA(p,mu,z);
 for pass=1:2,for i=1:j,a=Q(:,i)'*v;H(i,j)=H(i,j)+a;v=v-Q(:,i)*a;end,end
 H(j+1,j)=norm(v);g=zeros(j+1,1);g(1)=nb;y=H(1:j+1,1:j)\g;x=Z(:,1:j)*y;hist(end+1,1)=norm(b-applyA(p,mu,x))/nb;
 if H(j+1,j)<1e-13,break;end
 Q(:,j+1)=v/H(j+1,j);
end
coeffs=cell(1,j);for i=1:j,coeffs{i}=raw{i}*y(i);end
end

function o=fieldmetrics(v,p,part)
lab=zeros(p.N,1);for k=1:numel(part.blocks),lab(part.blocks(k).idx)=k;end
L=reshape(lab,p.n);mask=false(p.n);rough=0;F=reshape(v,p.n);
for d=1:3
 dif=diff(F,1,d);rough=rough+sum(abs(dif(:)).^2)/p.spacing(d)^2;
 bd=diff(L,1,d)~=0;a={':',':',':'};z=a;a{d}=1:p.n(d)-1;z{d}=2:p.n(d);mask(a{:})=mask(a{:})|bd;mask(z{:})=mask(z{:})|bd;
end
o=struct('interface_fraction',norm(v(mask(:)))^2/max(norm(v)^2,realmin),'interface_volume',mean(mask(:)), ...
 'roughness_over_omega2',rough/max(norm(v)^2,realmin)/p.cfg.omega^2);
end

function o=commutator_probe(p,A,b,m,coeffs,x,r,M,go,part)
bulk=complex(zeros(p.N,1));recon=bulk;
for j=1:numel(coeffs)
 l=m.layers{j};AS=A*l.S;off=0;
 for k=1:numel(l.space.blocks)
  z=l.space.blocks(k);c=coeffs{j}(off+(1:z.k));off=off+z.k;local=z.basis*c;
  a=pinv(l.S(z.idx,:),1e-12*norm(l.S(z.idx,:),2))*local;
  recon(z.idx)=recon(z.idx)+l.S(z.idx,:)*a;bulk(z.idx)=bulk(z.idx)+AS(z.idx,:)*a;
 end
end
re=norm(recon-x)/max(norm(x),realmin);assert(re<1e-8,'Snapshot extension reconstruction failed');
rb=b-bulk;ri=bulk-A*x;assert(norm(r-rb-ri)/norm(r)<1e-8);
[db,ib]=shifted_laplacian_gmres(@(v)A*v,M,rb/norm(rb),[],go);
[di,ii]=shifted_laplacian_gmres(@(v)A*v,M,ri/norm(ri),[],go);
o=struct('reconstruction_error',re,'bulk_over_total',norm(rb)/norm(r),'commutator_over_total',norm(ri)/norm(r), ...
 'cross_cosine',real(rb'*ri)/norm(rb)/norm(ri),'bulk_rho20',norm(rb/norm(rb)-A*db), ...
 'commutator_rho20',norm(ri/norm(ri)-A*di),'bulk_history',ib.relres_hist,'commutator_history',ii.relres_hist, ...
 'bulk_metrics',fieldmetrics(rb,p,part),'commutator_metrics',fieldmetrics(ri,p,part));
end
function A=matrixA(p,mu),A=p.K_pml+spdiags(p.affine.mass_base+mu*p.affine.mass_parameter_diagonal,0,p.N,p.N);end
function y=applyA(p,mu,x),y=p.K_pml*x+(p.affine.mass_base+mu*p.affine.mass_parameter_diagonal).*x;end
function a=append(a,b),if isempty(a),a=b;else,a(end+1)=b;end,end
function writejson(fn,x),f=fopen([fn '.tmp'],'w');assert(f>=0);fprintf(f,'%s\n',jsonencode(safejson(x)));fclose(f);movefile([fn '.tmp'],fn,'f');end
function x=safejson(x)
if isnumeric(x)&&~isreal(x),x=struct('real',real(x),'imag',imag(x));
elseif isstruct(x),ff=fieldnames(x);for k=1:numel(x),for j=1:numel(ff),x(k).(ff{j})=safejson(x(k).(ff{j}));end,end
elseif iscell(x),x=cellfun(@safejson,x,'UniformOutput',false);end
end

% Partition helpers copied unchanged from the existing 3D driver.
function partition = pilot_make_rom_partition(problem, mu, cfg)
    raw = problem.partition(mu);
    blocks = make_physical_blocks(problem, raw, cfg);

    pml_idx = find(raw.labels(:) == 3);
    switch cfg.dd_partition_mode
        case 'active_far_pml'
            blocks(end + 1) = make_partition_block('pml', 3, pml_idx);
        otherwise
            pml_blocks = make_refined_pml_blocks( ...
                problem, pml_idx, cfg.dd_partition_mode, cfg.pml_depth_fraction);
            blocks = [blocks, pml_blocks];
    end
    blocks = blocks([blocks.count] > 0);
    covered = sort(vertcat(blocks.idx));
    if ~isequal(covered, (1:problem.N).')
        error('The DD partition does not cover every unknown exactly once.');
    end
    partition = struct('n', problem.N, 'blocks', blocks, ...
        'block_names', {{blocks.name}}, ...
        'mode', cfg.dd_partition_mode, ...
        'physical_mode', cfg.physical_partition_mode);
end

function blocks = make_physical_blocks(problem, raw, cfg)
    switch cfg.physical_partition_mode
        case 'active_far'
            blocks = struct('name', {}, 'label', {}, 'idx', {}, 'count', {});
            blocks(end + 1) = make_partition_block( ...
                'active', 2, find(raw.labels(:) == 2));
            blocks(end + 1) = make_partition_block( ...
                'far', 1, find(raw.labels(:) == 1));
        case 'profile_bands'
            if ~isfield(problem, 'affine') || ...
                    ~isfield(problem.affine, 'mass_parameter_diagonal')
                error('profile_bands requires affine material-profile metadata.');
            end
            profile = abs(problem.affine.mass_parameter_diagonal(:)) / ...
                (problem.cfg.omega ^ 2);
            physical_idx = find(raw.labels(:) ~= 3);
            edges = [0; cfg.profile_partition_thresholds(:); inf];
            blocks = struct('name', {}, 'label', {}, 'idx', {}, 'count', {});
            for band = 1:numel(edges) - 1
                in_band = profile(physical_idx) >= edges(band) & ...
                    profile(physical_idx) < edges(band + 1);
                band_idx = physical_idx(in_band);
                blocks = append_axial_sections(blocks, problem, band_idx, ...
                    sprintf('profile_band_%d', band), 10 * band, ...
                    cfg.physical_axial_sections);
            end
    end
    blocks = blocks([blocks.count] > 0);
end

function blocks = append_axial_sections( ...
        blocks, problem, idx, base_name, base_label, section_count)
    if section_count == 1 || isempty(idx)
        blocks(end + 1) = make_partition_block(base_name, base_label, idx);
        return;
    end
    axis_dim = find(strcmp({'x', 'y', 'z'}, problem.cfg.waveguide_axis), 1);
    if isempty(axis_dim)
        axis_dim = 1;
    end
    [ix, iy, iz] = ind2sub(problem.n, idx);
    coords = {ix, iy, iz};
    physical_coordinate = coords{axis_dim} - problem.npml(axis_dim);
    section = min(floor((physical_coordinate - 1) * section_count / ...
        problem.nphys(axis_dim)) + 1, section_count);
    for k = 1:section_count
        blocks(end + 1) = make_partition_block( ...
            sprintf('%s_s%d', base_name, k), base_label + k, ...
            idx(section == k)); %#ok<AGROW>
    end
end

function blocks = make_refined_pml_blocks(problem, pml_idx, mode, depth_fraction)
    [ix, iy, iz] = ind2sub(problem.n, pml_idx);
    coords = {ix, iy, iz};
    axis_names = {'x', 'y', 'z'};
    axis_mask = false(numel(pml_idx), 3);
    low_mask = false(numel(pml_idx), 3);
    high_mask = false(numel(pml_idx), 3);
    depth = zeros(numel(pml_idx), 3);
    for d = 1:3
        low_mask(:, d) = coords{d} <= problem.npml(d);
        high_mask(:, d) = coords{d} > problem.n(d) - problem.npml(d);
        axis_mask(:, d) = low_mask(:, d) | high_mask(:, d);
        depth(low_mask(:, d), d) = problem.npml(d) - coords{d}(low_mask(:, d)) + 1;
        depth(high_mask(:, d), d) = ...
            coords{d}(high_mask(:, d)) - (problem.n(d) - problem.npml(d));
    end

    groups = struct('name', {}, 'idx', {});
    switch mode
        case {'directional_pml', 'signed_directional_pml', ...
                'directional_depth_pml', 'signed_depth_pml'}
            assigned = false(numel(pml_idx), 1);
            directional_mask = false(numel(pml_idx), 3);
            for d = 1:3
                directional_mask(:, d) = axis_mask(:, d) & ~assigned;
                assigned = assigned | directional_mask(:, d);
            end
            for d = 1:3
                mask = directional_mask(:, d);
                switch mode
                    case 'directional_pml'
                        groups(end + 1) = make_pml_group( ...
                            ['pml_' axis_names{d}], pml_idx(mask)); %#ok<AGROW>
                    case 'signed_directional_pml'
                        groups(end + 1) = make_pml_group( ...
                            ['pml_' axis_names{d} '_low'], ...
                            pml_idx(mask & low_mask(:, d))); %#ok<AGROW>
                        groups(end + 1) = make_pml_group( ...
                            ['pml_' axis_names{d} '_high'], ...
                            pml_idx(mask & high_mask(:, d))); %#ok<AGROW>
                    case 'directional_depth_pml'
                        inner_cells = max(1, ceil(depth_fraction * problem.npml(d)));
                        groups(end + 1) = make_pml_group( ...
                            ['pml_' axis_names{d} '_inner'], ...
                            pml_idx(mask & depth(:, d) <= inner_cells)); %#ok<AGROW>
                        groups(end + 1) = make_pml_group( ...
                            ['pml_' axis_names{d} '_outer'], ...
                            pml_idx(mask & depth(:, d) > inner_cells)); %#ok<AGROW>
                    case 'signed_depth_pml'
                        inner_cells = max(1, ceil(depth_fraction * problem.npml(d)));
                        groups = append_signed_depth_groups(groups, pml_idx, mask, ...
                            low_mask(:, d), high_mask(:, d), depth(:, d), ...
                            inner_cells, axis_names{d});
                end
            end
        case 'topological_pml'
            face_count = sum(axis_mask, 2);
            topology_names = {'x', 'y', 'z', 'xy', 'xz', 'yz', 'xyz'};
            topology_masks = { ...
                face_count == 1 & axis_mask(:, 1), ...
                face_count == 1 & axis_mask(:, 2), ...
                face_count == 1 & axis_mask(:, 3), ...
                face_count == 2 & axis_mask(:, 1) & axis_mask(:, 2), ...
                face_count == 2 & axis_mask(:, 1) & axis_mask(:, 3), ...
                face_count == 2 & axis_mask(:, 2) & axis_mask(:, 3), ...
                face_count == 3};
            for k = 1:numel(topology_names)
                groups(end + 1) = make_pml_group( ...
                    ['pml_' topology_names{k}], ...
                    pml_idx(topology_masks{k})); %#ok<AGROW>
            end
        otherwise
            error('Unsupported refined PML partition mode "%s".', mode);
    end

    groups = groups(arrayfun(@(group) ~isempty(group.idx), groups));
    blocks = struct('name', {}, 'label', {}, 'idx', {}, 'count', {});
    for k = 1:numel(groups)
        blocks(end + 1) = make_partition_block( ...
            groups(k).name, 30 + k, groups(k).idx); %#ok<AGROW>
    end
end

function groups = append_signed_depth_groups( ...
        groups, pml_idx, mask, low, high, depth, inner_cells, axis_name)
    side_names = {'low', 'high'};
    side_masks = {low, high};
    for side = 1:2
        groups(end + 1) = make_pml_group( ...
            sprintf('pml_%s_%s_inner', axis_name, side_names{side}), ...
            pml_idx(mask & side_masks{side} & depth <= inner_cells)); %#ok<AGROW>
        groups(end + 1) = make_pml_group( ...
            sprintf('pml_%s_%s_outer', axis_name, side_names{side}), ...
            pml_idx(mask & side_masks{side} & depth > inner_cells)); %#ok<AGROW>
    end
end

function group = make_pml_group(name, idx)
    group = struct('name', name, 'idx', idx(:));
end

function block = make_partition_block(name, label, idx)
    block = struct('name', name, 'label', label, 'idx', idx(:), ...
        'count', numel(idx));
end
