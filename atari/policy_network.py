import torch
import torch.nn as nn

class GridPolicyNetwork(nn.Module):
    """
    MLP 500 -> 32 -> 32 -> 5, tanh
    Matches the Weight-Space Programming paper exactly (17,253 weights).
    """
    def __init__(self):
        super().__init__()
        
        # Input: 5 channels * 10 * 10 = 500
        self.fc1 = nn.Linear(500, 32)
        self.fc2 = nn.Linear(32, 32)
        
        # Output: 5 action logits (NOOP, UP, DOWN, LEFT, RIGHT)
        self.fc3 = nn.Linear(32, 5)

    def forward(self, x):
        # Ensure the input is flattened from (batch, 5, 10, 10) to (batch, 500)
        x = x.view(x.size(0), -1)
        
        # Apply layers with tanh activation
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        
        # Raw logits out (no activation on the final layer)
        logits = self.fc3(x)
        
        return logits

# --- Verification ---
if __name__ == "__main__":
    model = GridPolicyNetwork()
    
    # Calculate total trainable parameters
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    
    # Dummy input representing a batch of 1 game state: (batch_size=1, channels=5, H=10, W=10)
    dummy_input = torch.zeros(1, 5, 10, 10)
    logits = model(dummy_input)
    
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {logits.shape}")
    print(f"Output logits: {logits.detach().numpy()}")