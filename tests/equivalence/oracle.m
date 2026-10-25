function root = oracle()
%ORACLE Path of the authors' original code, or '' when it is not available.
%   Equivalence tests compare the refactored code with the code that produced
%   the paper.  Set DMSRB_ORACLE to a checkout containing code/ to run them.
    root = getenv('DMSRB_ORACLE');
    if ~isempty(root) && isfolder(fullfile(root, 'code'))
        addpath(fullfile(root, 'code'));
    else
        root = '';
    end
end
