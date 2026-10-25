function root = helm_project_root()
%HELM_PROJECT_ROOT Return the repository root containing code/ and outputs/.
    code_dir = fileparts(mfilename('fullpath'));
    root = fileparts(code_dir);
end
