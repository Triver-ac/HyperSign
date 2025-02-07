import torch
import torch.nn as nn
import torch.nn.functional as F

class DShG(nn.Module):
    def __init__(self, dim, dropout_rate=0.5):
        super(DShG, self).__init__()
        self.dim = dim
        # Single hypernetwork to generate both gamma and beta
        self.hypernet = nn.Sequential(
            nn.Linear(2 * dim, dim),  # First layer to maintain dimension
            nn.ReLU(),                    # Non-linearity
            nn.Dropout(dropout_rate),     # Dropout layer after ReLU
            nn.Linear(dim, 2 * dim)   # Output layer that matches twice the dimension of the features
        )
        self.layernorm = nn.LayerNorm(dim)

    def forward(self, rgb_features, local_features):
        # Combine rgb and local features
        combined_features = torch.cat([rgb_features, local_features], dim=-1)

        # Use a single hypernetwork to generate extended features
        extended_features = self.hypernet(combined_features)
        # Split the extended features into gamma and beta
        gamma, beta = extended_features.chunk(2, dim=-1)

        # Element-wise multiplication of rgb and local features
        Multi_features = rgb_features * local_features
        Multi_features = self.layernorm(Multi_features)

        # Adjust rgb features using gamma and beta
        output = gamma * Multi_features + beta
        return output

if __name__ == '__main__':
    # Example usage
    dim = 1024  # Shared dimension for rgb and local features
    rgb_input = torch.rand(2, 33, dim)  # Example rgb features
    local_input = torch.rand(2, 33, dim)   # Example local features
    block = DShG(dim=dim)
    output = block(rgb_input, local_input)
    print(rgb_input.size(), local_input.size(), output.size())
