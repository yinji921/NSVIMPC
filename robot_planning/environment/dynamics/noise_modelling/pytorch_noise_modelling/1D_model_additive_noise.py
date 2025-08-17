import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

##-------------------------------------------------------------------------------------------------------------
#Sample data consisting of sine wave + Gaussian noise
noise_mean = 0.0
noise_std = 0.4
def generate_data(num_samples, noise_mean=noise_mean, noise_std=noise_std):
    x = np.linspace(0, 4 * np.pi, num_samples)
    data_noise = np.random.normal(noise_mean, noise_std, num_samples)
    y = np.sin(x) + data_noise
    return x, y, data_noise

##-------------------------------------------------------------------------------------------------------------
# # Sample data consisting of only Gaussian noise
# noise_mean = 0.0
# noise_std = 0.4
# def generate_data(num_samples, noise_mean=noise_mean, noise_std=noise_std):
#     x = np.linspace(0, 4 * np.pi, num_samples)
#     data_noise = np.random.normal(noise_mean, noise_std, num_samples)
#     y = data_noise
#     return x, y, data_noise

##-------------------------------------------------------------------------------------------------------------

num_samples = 1000
x, y, data_noise = generate_data(num_samples)

# Define the VAE architecture
class VAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(VAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2 * latent_dim)  # Split into mu and logvar
        )
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
        mu, logvar = torch.split(h, h.size(1) // 2, dim=1)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decoder(z)
        return x_recon, mu, logvar

# Hyperparameters
input_dim = 1
hidden_dim = 64
latent_dim = 2
batch_size = 64
epochs = 1000
lr = 0.001

# Create PyTorch dataset and dataloader
dataset = TensorDataset(torch.Tensor(y).unsqueeze(-1))
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Initialize VAE and optimizer
vae = VAE(input_dim, hidden_dim, latent_dim)
optimizer = optim.Adam(vae.parameters(), lr=lr)

# Loss function
def vae_loss(x, x_recon, mu, logvar):
    recon_loss = nn.MSELoss()(x_recon, x)
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    kl_loss /= batch_size * input_dim
    return recon_loss + 0.1*kl_loss

# Training loop
vae.train()
# for epoch in range(epochs):
#     for batch in dataloader:
#         x_batch = batch[0]
#         x_recon, mu, logvar = vae(x_batch)
#         loss = vae_loss(x_batch, x_recon, mu, logvar)
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#     if epoch % 100 == 0:
#         print(f'Epoch [{epoch}/{epochs}], Loss: {loss.item():.4f}')

reconstruction_losses = []
kl_divergence_losses = []
for epoch in range(epochs):
    epoch_recon_losses = []
    epoch_kl_losses = []
    for batch in dataloader:
        x_batch = batch[0]
        x_recon, mu, logvar = vae(x_batch)
        loss = vae_loss(x_batch, x_recon, mu, logvar)
        recon_loss = nn.MSELoss()(x_recon, x_batch)  # Reconstruction loss
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())  # KL divergence loss
        kl_loss /= batch_size * input_dim

        epoch_recon_losses.append(recon_loss.item())
        epoch_kl_losses.append(kl_loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    reconstruction_losses.append(np.mean(epoch_recon_losses))
    kl_divergence_losses.append(np.mean(epoch_kl_losses))

    if epoch % 100 == 0:
        print(
            f'Epoch [{epoch}/{epochs}], Loss: {loss.item():.4f}, Reconstruction Loss: {recon_loss.item():.4f}, KL Divergence Loss: {kl_loss.item():.4f}')

plt.plot(reconstruction_losses, label='Reconstruction Loss')
plt.plot(kl_divergence_losses, label='KL Divergence Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Losses vs. Epoch')
plt.legend()
plt.show()

# Generate denoised data
vae.eval()
with torch.no_grad():
    y_recon, _, _ = vae(torch.Tensor(y).unsqueeze(-1))

num_generated_samples = 1000
with torch.no_grad():
    z_sampled = torch.randn(num_generated_samples, latent_dim)
vae.eval()
with torch.no_grad():
    generated_noise = vae.decoder(z_sampled)

print("mean: ", np.mean(generated_noise.numpy()),"std: ", np.std(generated_noise.numpy()))
# # Plot noisy data, and generated data
plt.scatter(x, data_noise, alpha=0.3, label='Original Noise', color='red')
plt.scatter(x, generated_noise[:, 0].numpy(), alpha=0.3, label='Generated Noise', color='blue')
plt.legend()
plt.show()

# Plot histograms for the noisy data and generated data
plt.hist(generated_noise[:, 0].numpy(), bins=30, alpha=0.75, label="Generated Noise", color='blue')
plt.hist(data_noise, bins=30, alpha=0.75, label="Original Noise", color='red', histtype='step')
plt.legend()
plt.show()


# # Plot original data, noisy data, and denoised data
# plt.plot(x, np.sin(x), label='Original Data')
# plt.scatter(x, y, alpha=0.3, label='Noisy Data', color='red')
# plt.scatter(x, np.sin(x)+y_recon.numpy().squeeze(), alpha=0.3, label='Denoised Data', color='green')
# plt.legend()
# plt.show()

# #Sample data with uniform noise
# noise_mean = 0.5
# noise_std = 0.4
# def generate_data(num_samples, noise_mean=noise_mean, noise_std=noise_std):
#     x = np.linspace(0, 4 * np.pi, num_samples)
#     data_noise = np.random.uniform(low=noise_mean-np.sqrt(3)*noise_std, high=noise_mean+np.sqrt(3)*noise_std, size=num_samples)
#     # y = np.sin(x) + data_noise
#     y = data_noise
#     return x, y, data_noise