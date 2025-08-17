import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

##-------------------------------------------------------------------------------------------------------------
# Sample data consisting of changing variance
noise_mean = 0.0
noise_std = 0.4
def generate_data(num_samples, noise_mean=noise_mean, noise_std=noise_std):
    x = np.linspace(0, 4 * np.pi, num_samples)
    y = np.random.normal(noise_mean, 0.1*x, num_samples)
    return x, y
##-------------------------------------------------------------------------------------------------------------
# # Sample data consisting using fixed variance
# noise_mean = 0.0
# noise_std = 0.4
# def generate_data(num_samples, noise_mean=noise_mean, noise_std=noise_std):
#     x = np.linspace(0, 4 * np.pi, num_samples)
#     y = np.random.normal(noise_mean, noise_std, num_samples)
#     return x, y
##-------------------------------------------------------------------------------------------------------------

num_samples = 1000
x, y = generate_data(num_samples)

# Combine x and y values into input data
input_data = np.column_stack((x, y))

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
input_dim = 2
hidden_dim = 64
latent_dim = 2
batch_size = 64
epochs = 1000
lr = 0.001

# Create PyTorch dataset and dataloader
dataset = TensorDataset(torch.Tensor(input_data))
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
for epoch in range(epochs):
    for batch in dataloader:
        x_batch = batch[0]
        x_recon, mu, logvar = vae(x_batch)
        loss = vae_loss(x_batch, x_recon, mu, logvar)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if epoch % 100 == 0:
        print(f'Epoch [{epoch}/{epochs}], Loss: {loss.item():.4f}')

# Generate denoised data
vae.eval()
with torch.no_grad():
    input_data_recon, _, _ = vae(torch.Tensor(input_data))

# Separate the denoised x and y values
x_recon, y_recon = input_data_recon[:, 0], input_data_recon[:, 1]

# # Plot original data, noisy data, and denoised data
# plt.scatter(x, y, alpha=0.3, label='Noisy Data', color='red')
# plt.scatter(x_recon, y_recon, alpha=0.3, label='Denoised Data', color='green')
# plt.legend()
# plt.show()

num_generated_samples = 1000
with torch.no_grad():
    z_sampled = torch.randn(num_generated_samples, latent_dim)
vae.eval()
with torch.no_grad():
    generated_data = vae.decoder(z_sampled)

x_generated, y_generated = generated_data[:, 0], generated_data[:, 1]
plt.scatter(x, y, alpha=0.3, label='Noisy Data', color='red')
plt.scatter(x_generated, y_generated, alpha=0.3, label='Generated Data', color='blue')
plt.legend()
plt.show()

