function results = waveholtz_correction_rom(cfg)
%WAVEHOLTZ_CORRECTION_ROM
% Unified entrypoint for correction-ROM workflows.
%
% cfg.problem_type:
%   - 'helm' | 'helm_interface'             -> helm_waveholtz_correction_rom
%   - 'ecc'  | 'scattering' | 'scattering_ecc' -> waveholtz_2d_pml_scattering_ecc_correction_rom
%   - 'pc' | 'photonic_crystal' | 'double_layer' -> photonic_crystal_waveguide_correction_rom

    if nargin < 1 || isempty(cfg)
        cfg = struct();
    end

    if ~isfield(cfg, 'problem_type') || isempty(cfg.problem_type)
        cfg.problem_type = 'ecc';
    end

    mode = lower(strtrim(cfg.problem_type));
    if isfield(cfg, 'problem_type')
        cfg = rmfield(cfg, 'problem_type');
    end

    switch mode
        case {'helm', 'helm_interface'}
            results = helm_waveholtz_correction_rom(cfg);

        case {'ecc', 'scattering', 'scattering_ecc'}
            results = waveholtz_2d_pml_scattering_ecc_correction_rom(cfg);

        case {'pc', 'photonic_crystal', 'pc_waveguide', 'double_layer', 'two_layer'}
            results = photonic_crystal_waveguide_correction_rom(cfg);

        otherwise
            error(['Unsupported problem_type="%s". Use one of: ' ...
                   'helm, helm_interface, ecc, scattering, scattering_ecc, ' ...
                   'pc, photonic_crystal, pc_waveguide, double_layer, two_layer.'], mode);
    end
end
