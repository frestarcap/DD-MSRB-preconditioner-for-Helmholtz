function p = exp2_params()
%EXP2_PARAMS Frozen aspect ratios of Experiment 2: 96 training, 22 test geometries.
    p = jsondecode(fileread(fullfile(fileparts(mfilename('fullpath')), 'exp2_params.json')));
    p.train = p.train(:).';
    p.test = p.test(:).';
end
