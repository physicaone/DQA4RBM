# samplers/sampler.py
import torch
import numpy as np
from dimod import BinaryQuadraticModel
from dwave.system import DWaveSampler, EmbeddingComposite

def sample_negative_particles(mode, rbm, config, mask_tensor, graph_info=None, persistent_v=None):
    """
    mode: 'pcd' or 'dwave'
    returns: v_model, h_model (torch.Tensor)
    """
    device = torch.device('cpu')

    if mode == 'pcd':
        all_samples = []
        for _ in range(config['num_reads']):
            with torch.no_grad():
                for _ in range(config['k']):
                    h_persistent = rbm.sample_h(persistent_v)
                    persistent_v = rbm.sample_v(h_persistent)
            all_samples.append(persistent_v[0].detach().clone())

        v_model = torch.stack(all_samples).to(device)

        # Z2 symmetry fix
        for i in range(v_model.size(0)):
            if v_model[i, 0] == -1:
                v_model[i] *= -1

        h_model = torch.tanh(v_model @ (rbm.W * mask_tensor))
        h_model = torch.sign(h_model + torch.randn_like(h_model) * 0.01)

        return v_model, h_model, persistent_v

    elif mode == 'dwave':
        W_np = (rbm.W * mask_tensor).detach().cpu().numpy()
        h = {i: 0.0 for i in range(config['n_visible'] + config['n_hidden'])}
        J = {(i, config['n_visible'] + j): -W_np[i, j]/config['beta_rescale']
             for i in range(config['n_visible'])
             for j in range(config['n_hidden']) if graph_info['mask'][i, j] != 0}
        bqm = BinaryQuadraticModel.from_ising(h, J)

        sampler = EmbeddingComposite(DWaveSampler(token=config.get('token')))
        sampleset = sampler.sample(
            bqm,
            num_reads=config['num_reads'],
            auto_scale=False,
            fast_anneal=True,
            anneal_schedule=config['anneal_schedule'],
            label="RBM training"
        )
        samples = sampleset.record.sample.astype(np.int8)
        samples = 2 * samples - 1

        for i in range(samples.shape[0]):
            if samples[i, 0] == -1:
                samples[i] *= -1

        v_model = samples[:, :config['n_visible']]
        h_model = samples[:, config['n_visible']:]

        v_model = torch.tensor(v_model, dtype=torch.float32, device=device)
        h_model = torch.tensor(h_model, dtype=torch.float32, device=device)

        return v_model, h_model, None

    else:
        raise ValueError(f"Unknown sampling mode: {mode}")
