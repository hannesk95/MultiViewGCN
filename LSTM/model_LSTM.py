import torch
import torch.nn as nn

class LSTMClassifier(nn.Module):
    """
    A simple LSTM-based classifier for sequence data.
    
    Args:
        input_dim (int): The dimension of the input features (e.g., DINOv2's feature dimension).
        hidden_dim (int): The number of features in the hidden state of the LSTM.
        num_layers (int): The number of recurrent layers.
        output_dim (int): The dimension of the output (e.g., 1 for binary classification).
        
    """
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(LSTMClassifier, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # LSTM layer
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        
        # Fully connected layer for classification
        self.fc = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        """
        Forward pass for the LSTM classifier.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, sequence_length, input_dim).
        
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, output_dim).
        """
        # Initialize hidden and cell states
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        
        # Pass input through the LSTM layer
        # `out` contains the output for each time step, `(h_n, c_n)` is the final hidden and cell state
        out, (h_n, c_n) = self.lstm(x, (h0, c0))
        
        # Use the final hidden state from the last time step for classification
        # h_n shape: (num_layers, batch_size, hidden_dim)
        # We take the hidden state from the last layer (h_n[-1, :, :])
        
        # The input to the linear layer is the hidden state of the last time step from the last layer.
        # This is where the sequence-level information is aggregated.
        final_hidden_state = h_n[-1, :, :]
        
        # Pass through the linear layer
        out = self.fc(final_hidden_state)
        
        return out