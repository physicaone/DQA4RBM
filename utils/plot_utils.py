# utils/plot_utils.py
import torch
import matplotlib.pyplot as plt
import torchvision.utils as vutils

def save_reconstruction_images(rbm, data_loader, n_visible, save_path, device='cpu'):
    with torch.no_grad():
        for data, _ in data_loader:
            v_data = data.view(-1, 784)[:, :n_visible].to(device)
            v_data = torch.sign(v_data * 2 - 1)
            h_sample = rbm.sample_h(v_data)
            v_recon = rbm.sample_v(h_sample)
            break

    v_recon = (v_recon + 1) / 2.0
    v_recon_images = v_recon.view(-1, 1, 28, 28).cpu()
    v_recon_subset = v_recon_images[:100]
    grid = vutils.make_grid(v_recon_subset, nrow=10, padding=2, normalize=True)
    plt.figure(figsize=(10, 10))
    plt.axis('off')
    plt.title('Final Reconstruction Samples (Best Model)')
    plt.imshow(grid.permute(1, 2, 0).numpy())
    plt.savefig(save_path)
    plt.close()
    print(f"✅ Reconstruction image saved to {save_path}")
