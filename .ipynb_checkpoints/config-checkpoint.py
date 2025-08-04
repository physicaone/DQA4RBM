def get_config():
    # === 공통 설정 ===
    mode = 'pcd'  # 'pcd' 또는 'dwave' 중 택 1
    dataset = 'MNIST'
    n_visible = 784
    n_hidden = 1400
    epochs = 20
    batch_size = 50000
    initial_lr = 0.01
    beta_rescale = 1.0
    num_reads = 3000

    base_config = {
        'mode': mode,
        'dataset': dataset,
        'n_visible': n_visible,
        'n_hidden': n_hidden,
        'epochs': epochs,
        'batch_size': batch_size,
        'initial_lr': initial_lr,
        'save_dir': f'./results/{dataset}{n_hidden}',
        'graph_path': f'./results/dwave_graph_{n_hidden}.pkl'
    }

    # === PCD 모드일 경우 ===
    if mode == 'pcd':
        k = 100
        base_config.update({
            'k': k,
            'num_reads': num_reads,
            'save_dir': f"{base_config['save_dir']}/pcd{k}",
        })

    # === D-Wave 모드일 경우 ===
    elif mode == 'dwave':
        anneal_schedule = [(0.0, 0.0), (0.006, 1.0)]
        base_config.update({
            'beta_rescale': beta_rescale,
            'anneal_schedule': anneal_schedule,
            'num_reads': num_reads,
            'save_dir': f"{base_config['save_dir']}/dwave{beta_rescale}",
        })

    else:
        raise ValueError(f"Unknown mode '{mode}'")

    return base_config
