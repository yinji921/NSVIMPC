import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

def f(x_k, u_k):
    A = np.eye(8)
    B = np.random.rand(8, 2)
    return A @ x_k + B @ u_k

def generate_data(num_samples, noise_std=0.1):
    x_k = np.random.randn(8)
    noise_samples = []

    for i in range(num_samples):
        u_k = np.random.randn(2)
        # Generate multivariate Gaussian noise
        mean = np.zeros(8)  # Mean vector with 8 zeros
        cov = np.diag([0.1, 0.3, 0.5, 0.7, 0.9, 1.1, 1.3, 1.5])  # Covariance matrix with different variances on the diagonal
        w_k = np.random.multivariate_normal(mean, cov)
        # w_k = np.random.normal(0, noise_std, 8)
        x_k1 = f(x_k, u_k) + w_k
        noise_samples.append(w_k)
        x_k = x_k1

    return noise_samples

class VAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(VAE, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        self.mu = nn.Linear(hidden_dim, latent_dim)
        self.logvar = nn.Linear(hidden_dim, latent_dim)

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        h = self.encoder(x)
        mu, logvar = self.mu(h), self.logvar(h)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decoder(z)
        return x_recon, mu, logvar

def vae_loss(x, x_recon, mu, logvar, beta=0.0002):
    recon_loss = nn.MSELoss()(x_recon, x)
    kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    loss = recon_loss + beta*kl_div
    return loss, recon_loss, kl_div

num_samples = 1000
data = generate_data(num_samples)

def normalize_data(data):
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    return (data - mean) / std, mean, std

normalized_noise_data, noise_mean, noise_std = normalize_data(data)

batch_size = 64
dataset = TensorDataset(torch.Tensor(normalized_noise_data))
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

input_dim = 8
hidden_dim = 32
latent_dim = 8
epochs = 500

vae = VAE(input_dim, hidden_dim, latent_dim)
optimizer = optim.Adam(vae.parameters(), lr=1e-3)

# Add lists to store loss history
recon_loss_history = []
kl_loss_history = []

for epoch in range(epochs):
    vae.train()
    train_loss = 0
    recon_loss_epoch = 0
    kl_loss_epoch = 0
    for batch in dataloader:
        x = batch[0]
        optimizer.zero_grad()
        x_recon, mu, logvar = vae(x)
        loss, recon_loss, kl_loss = vae_loss(x, x_recon, mu, logvar)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        recon_loss_epoch += recon_loss.item()
        kl_loss_epoch += kl_loss.item()

    recon_loss_epoch /= len(dataloader)
    kl_loss_epoch /= len(dataloader)
    recon_loss_history.append(recon_loss_epoch)
    kl_loss_history.append(kl_loss_epoch)
    print(f"Epoch {epoch + 1}, Total Loss: {train_loss / len(dataloader)}, "
          f"Reconstruction Loss: {recon_loss_epoch}, "
          f"KL Divergence Loss: {kl_loss_epoch}")

# ... (rest of the code)

# Plot loss history
plt.figure()
plt.plot(recon_loss_history, label='Reconstruction Loss')
# plt.plot(kl_loss_history, label='KL Divergence Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

num_generated_samples = 1000

with torch.no_grad():
    z_sampled = torch.randn(num_generated_samples, latent_dim)

vae.eval()
with torch.no_grad():
    generated_noise_normalized = vae.decoder(z_sampled)

generated_noise_np = generated_noise_normalized.numpy() * noise_std + noise_mean
data_np = np.array(data)

plt.figure(figsize=(12, 6))
for i in range(8):
    plt.subplot(2, 4, i + 1)
    plt.hist(generated_noise_np[:, i], bins=30, alpha=0.75, label="Generated Noise", color='blue')
    plt.hist(data_np[:, i], bins=30, alpha=0.75, label="Original Noise", color='red', histtype='step')
    # print(len(generated_noise_np), len(data_np))
    plt.title(f'Noise Dimension {i + 1}')
    plt.legend()

plt.tight_layout()
plt.show()
