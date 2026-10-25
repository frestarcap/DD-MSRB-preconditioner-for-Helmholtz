function results = run_helm_waveholtz_interface_experiment()
    % Example: keep right-side speed fixed and train/test interface position.
    cfg = struct();
    cfg.fixed_c_right = 0.85;
    cfg.interface_x = 0.50;

    move_span = 0.04;
    train_raw = linspace(cfg.interface_x - move_span, cfg.interface_x + move_span, 24);
    cfg.train_params = min(max(train_raw, 0.02), 0.98);
    cfg.test_params = 0.5 * (cfg.train_params(1:end-1) + cfg.train_params(2:end));

    cfg.do_plots = true;
    results = helm_waveholtz_correction_rom_interface(cfg);
end
