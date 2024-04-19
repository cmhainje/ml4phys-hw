import torch
from torch import nn


def softplus(x, beta=1, threshold=20):
    z = beta * x
    mask = z < threshold
    x[mask] = torch.log(1 + torch.exp(z[mask])) / beta
    return x


class Encoder(nn.Module):
    """super simple 1D CNN. (performed well in hw3.ipynb.)"""
    def __init__(self, latent_dim=8, n_features=8575):
        super().__init__()
        
        channels = [1, 4, 4, 4]
        m = n_features
        kw = dict(padding='same', padding_mode='circular')
        
        layers = []
        for i in range(len(channels) - 1):
            layers.append(nn.Conv1d(channels[i], channels[i + 1], 3, **kw))
            layers.append(nn.ReLU())
        layers.append(nn.Flatten())
        layers.append(nn.Linear(channels[-1] * m, 2 * latent_dim))
        
        self.cnn = nn.Sequential(*layers)
        # self.softplus = nn.Softplus()
        
    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.cnn(x) # (N, 2 * latent_dim)
        
        # return means and variances separately, ensuring variances are positive
        means, variances = torch.tensor_split(x, 2, dim=-1)
        return means, softplus(variances)
    
    
class Decoder(nn.Module):
    """basically just the encoder reversed."""
    def __init__(self, latent_dim=8, n_features=8575):
        super().__init__()
        
        channels = [4, 4, 4, 1]
        m = n_features
        kw = dict(padding='same', padding_mode='circular')
        
        layers = []
        layers.append(nn.Linear(latent_dim, channels[0] * m))
        layers.append(nn.Unflatten(1, (channels[0], m)))
        for i in range(len(channels) - 1):
            layers.append(nn.Conv1d(channels[i], channels[i + 1], 3, **kw))
            if i != len(channels) - 2:
                layers.append(nn.ReLU())
        self.cnn = nn.Sequential(*layers)
        
    def forward(self, z):
        return self.cnn(z).squeeze()


class VAE(nn.Module):
    """coupled encoder and decoder for training"""
    def __init__(self, latent_dim=8, n_features=8575):
        super().__init__()
        self.encoder = Encoder(latent_dim, n_features)
        self.decoder = Decoder(latent_dim, n_features)
        
        # priors on latents
        self.register_buffer("p_z_mu", torch.zeros(latent_dim))
        self.register_buffer("p_z_sigma", torch.ones(latent_dim))
        
        # reconstruction loss function
        self.reco = nn.MSELoss(reduction='none')
        
        
    def log_prob(self, mu, sigma, z):
        var = torch.pow(sigma, 2)
        return -0.5 * torch.log(2 * torch.pi * var) - torch.pow(z - mu, 2) / (2 * var)
        
    def forward(self, x):
        # *** Run the model ***
        mu, sigma = self.encoder(x)
        eps = torch.randn(mu.shape, device=mu.device)
        z = mu + sigma * eps
        x_hat = self.decoder(z)
        
        if self.training:
            # *** Compute losses ***
            log_q_z = self.log_prob(mu, sigma, z).sum(-1)  # p(z|mu,sigma)
            log_p_z = self.log_prob(self.p_z_mu, self.p_z_sigma, z).sum(-1)  # p(z|priors)
            reco = -self.reco(x_hat, x).mean(-1)

            elbo = reco + log_p_z - log_q_z
            loss = -elbo.sum() / len(elbo)

            return x_hat, loss
        
        return x_hat