import torch
import torch.nn as nn


class ConcatFusion(nn.Module):
    def forward(self, img, proprio):
        if proprio is None:
            return img
        if img is None:
            return proprio
        return torch.cat([img, proprio], dim=-1) 



class FiLM(nn.Module):
    def __init__(
        self, cond_dim, output_dim, append=False, hidden_dim=64, activation=nn.Mish()
    ):
        super().__init__()
        self.cond_dim = cond_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.append = append
        # Define MLP to map proprio to gamma and beta
        self.mlp = nn.Sequential(
            nn.Linear(cond_dim, hidden_dim),  # First hidden layer
            activation,  # Activation function
            nn.Linear(hidden_dim, 2 * output_dim),  # Output layer for gamma and beta
        )

    def forward(self, img, proprio):
        gamma_beta = self.mlp(proprio)  # Compute gamma and beta
        gamma, beta = torch.chunk(gamma_beta, chunks=2, dim=-1)  # Split output

        img = gamma * img + beta  # Apply FiLM transformation
        return torch.cat([img, proprio], dim=-1) if self.append else img
