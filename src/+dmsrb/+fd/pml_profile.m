function pml = pml_profile(n, omega, npml, sigma_max)
%PML_PROFILE Quadratic PML damping and complex stretching on n+2 grid nodes.
%   pml = dmsrb.fd.pml_profile([nx ny], omega, npml, sigma_max) returns, per
%   direction d, sigma{d} = sigma_max*((npml-i+1)/npml)^2 on the first and last
%   npml nodes (boundary nodes included), s{d} = 1 + 1i*sigma{d}/omega and the
%   half-node averages s_half{d}.
    pml = struct('sigma', {cell(1, numel(n))}, 's', {cell(1, numel(n))}, ...
        's_half', {cell(1, numel(n))});
    for d = 1:numel(n)
        sigma = zeros(n(d) + 2, 1);
        for i = 1:npml
            value = sigma_max * ((npml - (i - 1)) / npml)^2;
            sigma(i) = value;
            sigma(end + 1 - i) = value;
        end
        s = 1.0 + 1i * sigma / omega;
        pml.sigma{d} = sigma;
        pml.s{d} = s;
        pml.s_half{d} = 0.5 * (s(1:end-1) + s(2:end));
    end
end
