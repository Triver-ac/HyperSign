import torch
import torch.nn as nn
import torch.nn.functional as F

class CIG(nn.Module):
    def __init__(self, dim, dropout_rate=0.5):
        super(CIG, self).__init__()
        self.dim = dim
        # Hypernetworks for generating weights and biases to apply crosswise
        self.gamma_hypernet_rgb = nn.Sequential(
            nn.Linear(dim, dim),  # To generate weights for local features
            nn.ReLU(),
        )
        self.beta_hypernet_rgb = nn.Sequential(
            nn.Linear(dim, dim),  # To generate biases for local features
            nn.ReLU(),
        )
        self.gamma_hypernet_local = nn.Sequential(
            nn.Linear(dim, dim),  # To generate weights for rgb features
            nn.ReLU(),
        )
        self.beta_hypernet_local = nn.Sequential(
            nn.Linear(dim, dim),  # To generate biases for rgb features
            nn.ReLU(),
        )
        self.layernorm = nn.LayerNorm(dim)

    def forward(self, rgb_features, local_features):
        # Generate weights and biases for local features using rgb hypernetworks
        gamma_local = self.gamma_hypernet_rgb(rgb_features)
        beta_local = self.beta_hypernet_rgb(rgb_features)
        # Adjust local features
        adjusted_local_features = gamma_local * local_features + beta_local

        # Generate weights and biases for rgb features using local hypernetworks
        gamma_rgb = self.gamma_hypernet_local(local_features)
        beta_rgb = self.beta_hypernet_local(local_features)
        # Adjust rgb features
        adjusted_rgb_features = gamma_rgb * rgb_features + beta_rgb

        # Combine adjusted features and apply LayerNorm
        combined_features = adjusted_rgb_features + adjusted_local_features
        output = self.layernorm(combined_features)
        return output

if __name__ == '__main__':
    # Example usage
    dim = 1024  # Shared dimension for rgb and local features
    rgb_input = torch.rand(2, 33, dim)  # Example rgb features
    local_input = torch.rand(2, 33, dim)   # Example local features
    block = CIG(dim=dim)
    output = block(rgb_input, local_input)
    print(rgb_input.size(), local_input.size(), output.size())
