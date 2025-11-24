import torch
import torch.nn as nn
import torch.nn.functional as F

class TemporalProbeEncoder(nn.Module):
    def __init__(self, num_probes, num_variables, seq_len, latent_dim, hidden_dim=512):
        super().__init__()
        self.num_probes = num_probes
        self.num_variables = num_variables
        self.seq_len = seq_len
        
        # Input dimension per time step: num_probes * num_variables
        self.input_dim = num_probes * num_variables
        
        # Feature extractor for each time step
        self.feature_extractor = nn.Sequential(
            nn.Linear(self.input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Temporal aggregator
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        
        # Latent projection
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        
    def forward(self, x):
        # x: (Batch, SeqLen, NumProbes, Variables)
        b, t, p, v = x.size()
        x = x.view(b, t, -1) # Flatten probes and variables
        
        # Extract features for each time step
        # (B*T, InputDim) -> (B*T, Hidden)
        features = self.feature_extractor(x.view(b*t, -1))
        features = features.view(b, t, -1)
        
        # Temporal aggregation
        # output: (B, T, Hidden), (h_n, c_n)
        _, (h_n, _) = self.lstm(features)
        
        # Use the final hidden state
        h_final = h_n[-1] # (B, Hidden)
        
        mu = self.fc_mu(h_final)
        logvar = self.fc_logvar(h_final)
        
        return mu, logvar

class ProbeVAE(nn.Module):
    def __init__(self, vae_model, probe_encoder):
        super().__init__()
        self.vae_model = vae_model
        self.probe_encoder = probe_encoder
        
        # Freeze the VAE model (both encoder and decoder, though we might only use decoder)
        for param in self.vae_model.parameters():
            param.requires_grad = False
            
    def forward(self, probes):
        # probes: (Batch, SeqLen, NumProbes, Variables)
        mu, logvar = self.probe_encoder(probes)
        z = self.reparameterize(mu, logvar)
        
        # Decode using the frozen VAE decoder
        # The VAE decoder expects z
        # MultiScaleVAE.decode(z) -> dict of outputs
        outputs = self.vae_model.decode(z)
        
        return outputs, mu, logvar
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def get_image_encoding(self, images):
        # Helper to get the "target" latent distribution from the full image
        # images: (Batch, Channels, H, W)
        return self.vae_model.encode(images)

