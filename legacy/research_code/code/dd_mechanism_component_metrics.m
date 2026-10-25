function out = dd_mechanism_component_metrics(v, problem, partition)
%DD_MECHANISM_COMPONENT_METRICS Spatial diagnostics of a complex 3D field.
% Works for residuals, approximate errors, and solutions. No inverse solve is
% performed here. This helper is intended for the frozen x-directed Gaussian
% waveguide cohort; it does not assume that Fourier components are eigenmodes.
%
% Energy means sum(abs(v).^2), with no Helmholtz energy norm implied. On this
% uniform mesh, multiplication by cell volume cancels from every fraction.
% Physical / inner PML / outer PML are disjoint. PML depth is the largest
% normalized outward grid depth over the three axes; inner means <= 1/2.
% Profile and axial fractions have PHYSICAL energy as their denominator.
% Source slab is |x-source_x| <= 3*source_sigma, restricted to the physical
% domain; upstream/downstream are the remaining axial slabs.
% Roughness is the sum of squared forward differences divided by spacing^2
% and physical field energy. Only edges inside the physical domain are used.
% Fourier power uses a Hann window in physical x, FFT along x independently
% for every physical (y,z), and sums the powers. k_x/omega is a reference
% wavenumber ratio (background speed is one in the paper), not a dispersion
% relation in this heterogeneous, nonnormal PML problem. Fourier fractions
% have WINDOWED Fourier energy as denominator, distinct from raw energy.
% All work is linear in N, with bounded column chunks rather than a dense
% full-grid Fourier transform. No model, operator, or input field is edited.

    if nargin < 3, partition = []; end
    n = double(problem.n(:).');
    np = double(problem.nphys(:).');
    pml = double(problem.npml(:).');
    h = double(problem.spacing(:).');
    assert(numel(n)==3 && numel(v)==prod(n), 'Field/grid size mismatch.');
    assert(strcmp(problem.cfg.waveguide_axis,'x'), ...
        'These diagnostics currently support x-directed waveguides.');
    assert(all(np>=2), 'Physical grid must contain at least two points per axis.');
    assert(isnumeric(v) && isvector(v), 'Expected one numeric field vector.');
    omega = problem.cfg.omega;
    assert(isscalar(omega) && omega>0);
    V = reshape(v,n(1),n(2)*n(3));
    ix = (pml(1)+(1:np(1))).';
    iy = pml(2)+(1:np(2));
    iz = pml(3)+(1:np(3));
    px = double(problem.coordinates{1}(ix));
    chunk = 128;

    out = struct();
    out.definition_version = 'dd_component_metrics_v1';
    out.N = prod(n);
    out.grid_size = n;
    out.omega = omega;
    out.energy_total = 0;
    out.region_names = {'physical','pml_inner','pml_outer'};
    out.region_energy = zeros(1,3);
    out.profile_band_edges = [0,1e-3,1e-2,1e-1,.5,Inf];
    out.profile_band_energy = zeros(1,5);
    out.axial_section_energy = zeros(1,5);
    out.axial_section_coordinate_edges = linspace(0,problem.lengths(1),6);
    out.source_region_names = {'upstream','source_slab','downstream'};
    out.source_region_energy = zeros(1,3);
    sigma = problem.cfg.source_sigma;
    if isempty(sigma), sigma=1.5*min(h); end
    source_bounds = problem.cfg.source_location(1)+[-3,3]*sigma;
    out.source_slab_x_bounds = source_bounds;
    roughness = zeros(1,3);
    fft_edges = [0,.5,.9,1.1,1.5,Inf];
    out.axial_fft_abs_k_over_omega_edges = fft_edges;
    out.axial_fft_power = zeros(1,numel(fft_edges)-1);
    out.axial_fft_signed_power = zeros(1,3); % negative, zero, positive
    out.axial_fft_signed_names = {'negative_k','zero_k','positive_k'};
    fft_total = 0;
    kfreq = [0:floor(np(1)/2), -ceil(np(1)/2)+1:-1].' ...
        *(2*pi/(np(1)*h(1)));
    kratio = abs(kfreq)/omega;
    hann = .5-.5*cos(2*pi*(0:np(1)-1).'/(np(1)-1));
    axial_section = min(floor((0:np(1)-1).'*5/np(1))+1,5);
    source_region = 2*ones(np(1),1);
    source_region(px<source_bounds(1))=1;
    source_region(px>source_bounds(2))=3;

    % Spatial regions include every extended-grid node exactly once.
    xphysical = false(n(1),1); xphysical(ix)=true;
    xdepth = outward_depth((1:n(1)).',n(1),pml(1));
    for c0 = 1:chunk:n(2)*n(3)
        cols = c0:min(c0+chunk-1,n(2)*n(3));
        [yy,zz] = ind2sub(n(2:3),cols);
        field = V(:,cols);
        assert(all(isfinite(field(:))), 'Field contains nonfinite values.');
        power = abs(field).^2;
        out.energy_total=out.energy_total+sum(power(:));
        yzphysical = yy>pml(2) & yy<=pml(2)+np(2) & ...
            zz>pml(3) & zz<=pml(3)+np(3);
        physical = xphysical & yzphysical;
        yzdepth = max(outward_depth(yy,n(2),pml(2)), ...
            outward_depth(zz,n(3),pml(3)));
        depth = max(xdepth,yzdepth);
        inner = ~physical & depth<=.5;
        outer = ~physical & depth>.5;
        out.region_energy=out.region_energy+ ...
            [sum(power(physical)),sum(power(inner)),sum(power(outer))];
    end

    widths = problem.cfg.waveguide_width;
    if isscalar(widths), widths=[widths,widths]; end
    assert(numel(widths)==2 && all(widths>0));
    [YY,ZZ] = ndgrid(iy,iz);
    physical_cols = sub2ind(n(2:3),YY(:),ZZ(:));
    % At most n_y*n_z indices are formed; no full 3D coordinate arrays.
    for c0 = 1:chunk:numel(physical_cols)
        selection = c0:min(c0+chunk-1,numel(physical_cols));
        cols = physical_cols(selection);
        [yy,zz] = ind2sub(n(2:3),cols);
        field = V(ix,cols);
        power = abs(field).^2;
        per_x = sum(power,2);
        per_yz = sum(power,1);
        ycoord=problem.coordinates{2}(yy);
        zcoord=problem.coordinates{3}(zz);
        profile=exp(-.5*((ycoord(:)-problem.cfg.waveguide_center(2))/widths(1)).^2 ...
            -.5*((zcoord(:)-problem.cfg.waveguide_center(3))/widths(2)).^2);
        for b = 1:5
            keep = profile>=out.profile_band_edges(b) & ...
                profile<out.profile_band_edges(b+1);
            out.profile_band_energy(b)=out.profile_band_energy(b)+sum(per_yz(keep));
            out.axial_section_energy(b)=out.axial_section_energy(b)+sum(per_x(axial_section==b));
        end
        for b = 1:3
            out.source_region_energy(b)=out.source_region_energy(b)+sum(per_x(source_region==b));
        end
        d = diff(field,1,1)/h(1);
        roughness(1)=roughness(1)+sum(abs(d(:)).^2);
        keep=yy<pml(2)+np(2);
        d=(V(ix,cols(keep)+1)-field(:,keep))/h(2);
        roughness(2)=roughness(2)+sum(abs(d(:)).^2);
        keep=zz<pml(3)+np(3);
        d=(V(ix,cols(keep)+n(2))-field(:,keep))/h(3);
        roughness(3)=roughness(3)+sum(abs(d(:)).^2);
        transform=fft(hann.*field,[],1);
        spectral=sum(abs(transform).^2,2)/np(1); % Parseval-normalized
        fft_total=fft_total+sum(spectral);
        for b=1:numel(fft_edges)-1
            keep=kratio>=fft_edges(b) & kratio<fft_edges(b+1);
            out.axial_fft_power(b)=out.axial_fft_power(b)+sum(spectral(keep));
        end
        out.axial_fft_signed_power=out.axial_fft_signed_power+ ...
            [sum(spectral(kfreq<0)),sum(spectral(kfreq==0)),sum(spectral(kfreq>0))];
    end
    physical_energy=out.region_energy(1);
    out.region_fraction=normalize(out.region_energy,out.energy_total);
    out.profile_band_fraction=normalize(out.profile_band_energy,physical_energy);
    out.axial_section_fraction=normalize(out.axial_section_energy,physical_energy);
    out.source_region_fraction=normalize(out.source_region_energy,physical_energy);
    out.roughness_axis_names={'x','y','z'};
    out.roughness_numerator=roughness;
    out.roughness_over_energy=normalize(roughness,physical_energy);
    out.roughness_over_omega_squared=out.roughness_over_energy/omega^2;
    out.roughness_axial_over_omega_squared=out.roughness_over_omega_squared(1);
    out.roughness_transverse_over_omega_squared=sum(out.roughness_over_omega_squared(2:3));
    out.axial_fft_windowed_energy=fft_total;
    out.axial_fft_fraction=normalize(out.axial_fft_power,fft_total);
    out.axial_fft_signed_fraction=normalize(out.axial_fft_signed_power,fft_total);
    assert_conservation(out.region_energy,out.energy_total,'spatial regions');
    assert_conservation(out.profile_band_energy,physical_energy,'profile bands');
    assert_conservation(out.axial_section_energy,physical_energy,'axial sections');
    assert_conservation(out.source_region_energy,physical_energy,'source slabs');
    assert_conservation(out.axial_fft_power,fft_total,'Fourier bins');

    out.partition_names={};
    out.partition_energy=[];
    out.partition_fraction=[];
    if ~isempty(partition)
        assert(isfield(partition,'blocks'), 'Expected saved ROM partition.blocks.');
        nb=numel(partition.blocks);
        out.partition_names=cell(1,nb);
        out.partition_energy=zeros(1,nb);
        coverage=zeros(prod(n),1,'uint8');
        for b=1:nb
            idx=partition.blocks(b).idx;
            assert(all(idx>=1 & idx<=prod(n)) && numel(unique(idx))==numel(idx), ...
                'Invalid or repeated block indices.');
            assert(~any(coverage(idx)), 'Partition blocks overlap.');
            coverage(idx)=1;
            out.partition_names{b}=partition.blocks(b).name;
            for j=1:65536:numel(idx)
                z=v(idx(j:min(j+65535,numel(idx))));
                out.partition_energy(b)=out.partition_energy(b)+sum(abs(z(:)).^2);
            end
        end
        assert(all(coverage), 'Partition does not cover the full grid.');
        out.partition_fraction=normalize(out.partition_energy,out.energy_total);
        assert_conservation(out.partition_energy,out.energy_total,'DD blocks');
    end
end

function d = outward_depth(i,n,npml)
    if npml==0, d=zeros(size(i)); return; end
    d=max(max(npml-i+1,i-(n-npml)),0)/npml;
end

function f = normalize(a,total)
    if total>0, f=a/total; else, f=zeros(size(a)); end
end

function assert_conservation(parts,total,label)
    assert(abs(sum(parts)-total)<=2e-11*max(total,realmin), ...
        'Energy conservation failed for %s.',label);
end
