function run_dd_error_spectrum_saved(base,outdir)
% Read-only full-volume Fourier audit of frozen computed errors and residuals.
assert(~isfolder(outdir),'Refuse overwrite');mkdir(outdir);
summary=struct('version','dd_error_spectrum_v1','matlab',version,'job',getenv('SLURM_JOB_ID'),'records',struct([]));
for omega=[24,48]
 if omega==24,rd='refined_w24';vd='w24';else,rd='refined_fast_w48';vd='fast_w48';end
 meta=jsondecode(fileread(fullfile(base,'results',rd,'solver_summary.json')));
 t=load(meta.source_training,'geometry');g=t.geometry;
 n=double(g.grid_size(:).');np=double(g.physical_grid_size(:).');p=double(g.pml_cells(:).');h=double(g.spacing(:).');
 for mc={'global','dd'}
  method=mc{1};if strcmp(method,'global'),label='GGG';else,label='DDD';end
  file=fullfile(base,'results',rd,[method '_refined_error.mat']);s=load(file,'new_error','old_error');
  for fc={'new_error','old_error','reference_change','residual'}
   field=fc{1};
   if strcmp(field,'reference_change'),v=s.new_error-s.old_error;
   elseif strcmp(field,'residual')
    f=dir(fullfile(base,'results',vd,['*_' label '_vectors.mat']));assert(numel(f)==1);r=load(fullfile(f.folder,f.name),'r0');v=r.r0;clear r
   else,v=s.(field);end
   assert(numel(v)==prod(n)&&all(isfinite(v)));
   V=reshape(v,n);V=V(p(1)+(1:np(1)),p(2)+(1:np(2)),p(3)+(1:np(3)));clear v
   for wc={'rectangular','hann_xyz','hann_x'}
    row=analyze(V,h,omega,wc{1});row.omega=omega;row.method=method;row.field=field;row.window=wc{1};row.reference_rho=meta.new_true_relative_residual;
    if isempty(summary.records),summary.records=row;else,summary.records(end+1)=row;end
   end
   fprintf('DONE omega=%g %s %s\n',omega,method,field);
  end
 end
end
f=fopen(fullfile(outdir,'spectra.json'),'w');fprintf(f,'%s\n',jsonencode(summary));fclose(f);
save(fullfile(outdir,'spectra.mat'),'summary','-v7');
end
function row=analyze(V,h,omega,window)
n=size(V);W=V;
if ~strcmp(window,'rectangular')
 dims=1;if strcmp(window,'hann_xyz'),dims=1:3;end
 for d=dims
  a=.5-.5*cos(2*pi*(0:n(d)-1)/(n(d)-1));sz=ones(1,3);sz(d)=n(d);W=W.*reshape(a,sz);
 end
end
E=sum(abs(W(:)).^2);F=fftn(W);P=abs(F).^2/prod(n);clear F W
parseval=abs(sum(P(:))-E)/max(E,realmin);assert(parseval<1e-10);
k=cell(1,3);theta=cell(1,3);
for d=1:3
 sz=ones(1,3);sz(d)=n(d);theta{d}=reshape(2*pi*[0:floor(n(d)/2),-ceil(n(d)/2)+1:-1]/n(d),sz);k{d}=theta{d}/h(d);
end
radial=sqrt(k{1}.^2+k{2}.^2+k{3}.^2)/omega;
tmax=max(max(abs(theta{1}),abs(theta{2})),abs(theta{3}))/pi;
edges=[0:.05:8,Inf];bins=discretize(radial(:),edges);power=accumarray(bins,P(:),[numel(edges)-1,1]);
tedges=0:.025:1.00001;tbins=discretize(tmax(:),tedges);tpower=accumarray(tbins,P(:),[numel(tedges)-1,1]);
row=struct('physical_energy',sum(abs(V(:)).^2),'windowed_energy',E,'parseval_relative_error',parseval,...
 'radial_lower_edges',edges(1:end-1),'radial_fraction',power.'/E,'theta_lower_edges',tedges(1:end-1),'theta_fraction',tpower.'/E,...
 'radial_ge1p5',sum(P(radial>=1.5))/E,'grid_high_any',sum(P(tmax>=.5))/E,'axis',struct());
for d=1:3
 other=setdiff(1:3,d);marg=P;for o=other,marg=sum(marg,o);end;marg=marg(:);
 kval=k{d}(:);tv=theta{d}(:);[ka,order]=sort(kval/omega);
 row.axis(d).k_over_omega=ka;row.axis(d).fraction=marg(order)/E;
 row.axis(d).ge1p5=sum(marg(abs(kval)/omega>=1.5))/E;
 row.axis(d).grid_high=sum(marg(abs(tv)>=pi/2))/E;
end
end
