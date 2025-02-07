import torch
import torch.nn as nn
import torch.nn.functional as F
class DSG(nn.Module):
    def __init__(self, dim, dropout_rate=0.5):
        print("dropout rate",dropout_rate)
        super(DSG, self).__init__()
        self.dim = dim
        # Hypernetworks for gamma (weights) and beta (biases)
        self.gamma_hypernet = nn.Sequential(
            nn.Linear(2 * dim, dim),  # First layer to reduce dimension
            nn.ReLU(),                # Non-linearity
            nn.Dropout(dropout_rate), # Dropout layer after ReLU
            nn.Linear(dim, dim)       # Output layer that matches the dimension of the features
        )
        self.beta_hypernet = nn.Sequential(
            nn.Linear(2 * dim, dim),  # Similar structure for beta
            nn.ReLU(),
            nn.Dropout(dropout_rate), # Dropout layer after ReLU
            nn.Linear(dim, dim)
        )
        self.layernorm = nn.LayerNorm(dim)

    def forward(self, global_features, local_features):
        # Combine global and local features
        combined_features = torch.cat([global_features, local_features], dim=-1)

        # Use hypernetworks to generate gamma and beta
        gamma = self.gamma_hypernet(combined_features)
        beta = self.beta_hypernet(combined_features)
        Multi_features = global_features * local_features
        Multi_features = self.layernorm(Multi_features)
        # Adjust global features using gamma and beta
        output = gamma * Multi_features + beta  # Modified to directly use gamma without adding 1
        return output
    
if __name__ == '__main__':
    # Example usage
    dim = 1024  # Shared dimension for global and local features
    global_input = torch.rand(2, 33, dim)  # Example global features
    local_input = torch.rand(2, 33, dim)   # Example local features
    block = DSG(dim=dim)
    output = block(global_input, local_input)
    print(global_input.size(), local_input.size(), output.size())
