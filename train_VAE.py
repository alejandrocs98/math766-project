#!/bin/python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler
import joblib

# === Set hyperparameters ===
epochs = 100
batch_size = 32
learning_rate = 1e-4
dropout_prob = 0.3
latent_dim = 10
hidden_dim = 512
use_batchnorm = False
kl_anneal_epochs = epochs // 10 if epochs <= 1000 else 100
beta_max = 1.0
patience = 20
grad_clip_value = 1.0
# === Set random seed for reproducibility ===
torch.manual_seed(2025)
np.random.seed(2025)

# === Set device ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# === Load and robust scale your data ===
gene_df = pd.read_table("ko_table.tsv", index_col=0)
gene_df.columns.name = "KO"

scaler = RobustScaler()
gene_df_scaled = scaler.fit_transform(gene_df)
data_tensor = torch.tensor(gene_df_scaled, dtype=torch.float32).to(device)
dataset = TensorDataset(data_tensor)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# === VAE with Dropout and Normalization ===
class VAE(nn.Module):
    def __init__(self, input_dim, hidden_dim=512, latent_dim=10, dropout_prob=0.2, use_batchnorm=True):
        super(VAE, self).__init__()
        self.use_batchnorm = use_batchnorm

        # Encoder
        self.encoder_fc = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout_prob)
        self.relu = nn.ReLU()

        self.mu_layer = nn.Linear(hidden_dim, latent_dim)
        self.logvar_layer = nn.Linear(hidden_dim, latent_dim)

        # Decoder
        self.decoder_fc1 = nn.Linear(latent_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)
        self.decoder_fc2 = nn.Linear(hidden_dim, input_dim)

    def encode(self, x):
        x = self.encoder_fc(x)
        x = self.bn1(x) if self.use_batchnorm else self.ln1(x)
        x = self.relu(x)
        x = self.dropout(x)
        return self.mu_layer(x), self.logvar_layer(x)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        z = self.decoder_fc1(z)
        z = self.bn2(z) if self.use_batchnorm else self.ln2(z)
        z = self.relu(z)
        z = self.dropout(z)
        return self.decoder_fc2(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar

# === Loss Function with KL Annealing ===
def loss_function(recon_x, x, mu, logvar, kl_weight):
    recon_loss = nn.MSELoss()(recon_x, x)
    kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + kl_weight * kl_loss, recon_loss.item(), kl_loss.item()

# === Train the model ===
def train_vae(model, dataloader, optimizer, device, epochs=100, beta_max=1.0, kl_anneal_epochs=50, grad_clip_value=1.0, patience=20):
    model.train()
    loss_history = []
    recon_history = []
    kl_history = []
    best_loss = float("inf")
    counter = 0

    for epoch in range(epochs):
        total_loss = 0
        total_recon = 0
        total_kl = 0

        # Linearly anneal beta from 0 to beta_max
        beta = beta_max * min(1.0, epoch / kl_anneal_epochs)

        for batch in dataloader:
            batch = batch[0].to(device)
            optimizer.zero_grad()
            x_recon, mu, logvar = model(batch)

            recon_loss = F.mse_loss(x_recon, batch, reduction="sum") / batch.size(0)
            kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / batch.size(0)

            # Optional: Add small latent regularization to prevent collapse
            latent_l1 = torch.mean(torch.abs(mu))
            latent_penalty = 1e-3 * latent_l1

            loss = recon_loss + beta * kl_div + latent_penalty
            loss.backward()

            # Gradient clipping
            if grad_clip_value is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_value)

            optimizer.step()

            total_loss += loss.item()
            total_recon += recon_loss.item()
            total_kl += kl_div.item()

        avg_loss = total_loss / len(dataloader)
        avg_recon = total_recon / len(dataloader)
        avg_kl = total_kl / len(dataloader)

        loss_history.append(avg_loss)
        recon_history.append(avg_recon)
        kl_history.append(avg_kl)

        print(
            f"Epoch [{epoch+1}/{epochs}] Loss: {avg_loss:.4f} | Recon: {avg_recon:.4f} | KL: {avg_kl:.4f} | Beta: {beta:.4f}"
        )

        # Early stopping
        if avg_loss < best_loss:
            best_loss = avg_loss
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                print("Early stopping triggered.")
                break

    history = {
        "epoch": list(range(1, len(loss_history) + 1)),
        "loss": loss_history,
        "reconstruction_loss": recon_history,
        "kl_divergence": kl_history,
    }

    return model, history

# === Model Setup ===
input_dim = gene_df.shape[1]
vae = VAE(input_dim, hidden_dim=hidden_dim, latent_dim=latent_dim, dropout_prob=dropout_prob, use_batchnorm=use_batchnorm).to(device)
optimizer = optim.Adam(vae.parameters(), lr=learning_rate)

# === Train the model ===
vae, history = train_vae(vae, dataloader, optimizer, device, epochs=epochs, kl_anneal_epochs=kl_anneal_epochs, grad_clip_value=grad_clip_value, patience=patience)

# === Save trained model ===
torch.save(vae.state_dict(), f"vae_model_{epochs}.pth")

# === Save the scaler for inverse transform ===
joblib.dump(scaler, "robust_scaler.pkl")

# === Save training history ===
history_df = pd.DataFrame(history)
history_df.to_csv("training_history.csv", index=False)

# === Save latent variables ===
vae.eval()
with torch.no_grad():
    latent_vars = []
    for batch in dataloader:
        batch = batch[0].to(device)
        mu, _ = vae.encode(batch)
        latent_vars.append(mu.cpu().numpy())
    latent_vars = np.concatenate(latent_vars, axis=0)
    latent_df = pd.DataFrame(latent_vars, columns=[f"latent_{i}" for i in range(latent_vars.shape[1])])
    latent_df.to_csv("latent_variables.csv", index=False)

# === Save reconstructed data ===
vae.eval()
with torch.no_grad():
    recon_data = []
    for batch in dataloader:
        batch = batch[0].to(device)
        recon_batch, _, _ = vae(batch)
        recon_data.append(recon_batch.cpu().numpy())
    recon_data = np.concatenate(recon_data, axis=0)
    recon_df = pd.DataFrame(recon_data, columns=gene_df.columns)
    recon_df.to_csv("reconstructed_data.csv", index=False)

# === Save model architecture and hyperparameters ===
model_info = {
    "input_dim": input_dim,
    "hidden_dim": 512,
    "latent_dim": 10,
    "dropout_prob": 0.3,
    "use_batchnorm": False,
    "epochs": epochs
}
# === Save model weights ===
vae_weights = vae.state_dict()
torch.save(vae_weights, "vae_weights.pth")
with open("model_info.txt", "w") as f:
    for key, value in model_info.items():
        f.write(f"{key}: {value}\n")