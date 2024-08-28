import torch
import torch.nn as nn
from .block import xLSTMBlock

class xLSTM(nn.Module):
    """
    xLSTM model for malware detection.

    Args:
        input_size (int): Size of the input embeddings.
        hidden_size (int): Size of the hidden state in LSTM blocks.
        num_layers (int): Number of LSTM layers in each block.
        num_blocks (int): Number of xLSTM blocks.
        dropout (float, optional): Dropout probability. Default: 0.0.
        lstm_type (str, optional): Type of LSTM to use ('slstm' or 'mlstm'). Default: 'slstm'.
    """

    def __init__(self, input_size, hidden_size, num_layers, num_blocks,
                 dropout=0.0, lstm_type="slstm"):
        super(xLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_blocks = num_blocks
        self.dropout = dropout
        self.lstm_type = lstm_type

        self.blocks = nn.ModuleList([
            xLSTMBlock(input_size, hidden_size, num_layers, dropout, lstm_type)
            for _ in range(num_blocks)
        ])
        self.output_layer = nn.Linear(input_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_seq, hidden_states=None):
        """
        Forward pass of the xLSTM model for malware detection.

        Args:
            input_seq (Tensor): Already embedded input sequence.
            hidden_states (list of tuples, optional): Initial hidden states for each block. Default: None.

        Returns:
            tuple: Output probability and final hidden states.
        """
        # Check for NaN values in output_seq
        if torch.isnan(input_seq).any():
            print("NaN detected in input_seq!")
        output_seq = input_seq
        if hidden_states is None:
            hidden_states = [None] * self.num_blocks
        
        for i, block in enumerate(self.blocks):
            output_seq, hidden_states[i] = block(output_seq, hidden_states[i])
            # Check for NaN values after each block
            if torch.isnan(output_seq).any():
                print(f"NaN detected after block {i+1}!")
        
        output_seq = self.output_layer(output_seq[:, -1, :])  # Taking the output of the last time step
        if torch.isnan(output_seq).any():
            print("NaN detected in output_seq!")
        output_prob = self.sigmoid(output_seq)
        
        return output_prob, hidden_states


class CNN_xLSTM(nn.Module):
    """
    CNN-xLSTM model for malware detection with three convolutional layers.

    Args:
        input_size (int): Size of the input embeddings.
        hidden_size (int): Size of the hidden state in LSTM blocks.
        num_layers (int): Number of LSTM layers in each block.
        num_blocks (int): Number of xLSTM blocks.
        dropout (float, optional): Dropout probability. Default: 0.0.
        lstm_type (str, optional): Type of LSTM to use ('slstm' or 'mlstm'). Default: 'slstm'.
        num_channels (int, optional): Number of channels in the convolutional layers. Default: 32.
    """

    def __init__(self, input_size, hidden_size, num_layers, num_blocks,
                 dropout=0.0, lstm_type="slstm", num_channels=32):
        super(CNN_xLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_blocks = num_blocks
        self.dropout = dropout
        self.lstm_type = lstm_type

        # Three convolutional layers with kernel sizes 2, 3, and 4
        self.conv1 = nn.Conv1d(in_channels=input_size, out_channels=num_channels, kernel_size=2, padding=1)
        self.conv2 = nn.Conv1d(in_channels=input_size, out_channels=num_channels, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(in_channels=input_size, out_channels=num_channels, kernel_size=4, padding=1)

        self.blocks = nn.ModuleList([
            xLSTMBlock(num_channels * 3, hidden_size, num_layers, dropout, lstm_type)
            for _ in range(num_blocks)
        ])
        self.output_layer = nn.Linear(num_channels * 3, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_seq, hidden_states=None):
        """
        Forward pass of the CNN-xLSTM model for malware detection.

        Args:
            input_seq (Tensor): Already embedded input sequence.
            hidden_states (list of tuples, optional): Initial hidden states for each block. Default: None.

        Returns:
            tuple: Output probability and final hidden states.
        """
        # Check for NaN values in input_seq
        if torch.isnan(input_seq).any():
            print("NaN detected in input_seq!")
        
        # Apply convolutional layers
        conv1_out = self.conv1(input_seq.permute(0, 2, 1)).permute(0, 2, 1)
        conv2_out = self.conv2(input_seq.permute(0, 2, 1)).permute(0, 2, 1)
        conv3_out = self.conv3(input_seq.permute(0, 2, 1)).permute(0, 2, 1)
        
        # Concatenate the outputs of the three convolutional layers
        output_seq = torch.cat([conv1_out, conv2_out, conv3_out])
        
        if hidden_states is None:
            hidden_states = [None] * self.num_blocks
        
        for i, block in enumerate(self.blocks):
            output_seq, hidden_states[i] = block(output_seq, hidden_states[i])
            # Check for NaN values after each block
            if torch.isnan(output_seq).any():
                print(f"NaN detected after block {i+1}!")
        
        output_seq = self.output_layer(output_seq[:, -1, :])  # Taking the output of the last time step
        if torch.isnan(output_seq).any():
            print("NaN detected in output_seq!")
        output_prob = self.sigmoid(output_seq)
        
        return output_prob, hidden_states
