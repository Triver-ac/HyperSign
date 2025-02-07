import torch
import torch.nn as nn
import torch.nn.functional as F

class PIG(nn.Module):
    def __init__(self, dim, dropout_rate=0.5):
        super(PIG, self).__init__()
        self.dim = dim
        # Hypernetworks for gamma (weights) and beta (biases) for rgb features
        # self.gamma_hypernet_rgb = nn.Sequential(
        #     nn.Linear(dim, dim),
        #     nn.ReLU(),
        #     nn.Dropout(dropout_rate),
        #     nn.Linear(dim, dim)
        # )
        self.gamma_hypernet_rgb = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
        )
        self.beta_hypernet_rgb = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
        )
        # Hypernetworks for gamma (weights) and beta (biases) for local features
        self.gamma_hypernet_local = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
        )
        self.beta_hypernet_local = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
        )
        self.layernorm = nn.LayerNorm(dim)

    def forward(self, rgb_features, local_features):
        # Generate gamma and beta for rgb features
        gamma_rgb = self.gamma_hypernet_rgb(rgb_features)
        beta_rgb = self.beta_hypernet_rgb(rgb_features)
        # Adjust rgb features
        adjusted_rgb_features = gamma_rgb * rgb_features + beta_rgb

        # Generate gamma and beta for local features
        gamma_local = self.gamma_hypernet_local(local_features)
        beta_local = self.beta_hypernet_local(local_features)
        # Adjust local features
        adjusted_local_features = gamma_local * local_features + beta_local

        # Combine adjusted features and apply LayerNorm
        combined_features = adjusted_rgb_features + adjusted_local_features
        output = self.layernorm(combined_features)
        return output

if __name__ == '__main__':
    # Example usage
    dim = 1024  # Shared dimension for rgb and local features
    rgb_input = torch.rand(2, 33, dim)  # Example rgb features
    local_input = torch.rand(2, 33, dim)   # Example local features
    block = PIG(dim=dim)
    output = block(rgb_input, local_input)
    print(rgb_input.size(), local_input.size(), output.size())
