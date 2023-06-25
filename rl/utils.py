import torch
import torch.nn as nn
import ipdb

class FixedNormal(torch.distributions.Normal):
    def log_prob(self, actions):
        return super().log_prob(actions).sum(dim=-1)

    def entropy(self):
        return super().entropy().sum(dim=-1)

    def mode(self):
        return self.mean


class ScaleParameterizedNormal(nn.Module):
    def __init__(self, n_actions, init_scale=1.0):
        super().__init__()
        self.logstd = nn.Parameter(torch.zeros(n_actions))
        self.init_scale = init_scale

    def forward(self, logits):
        return FixedNormal(logits, self.logstd.expand_as(logits).exp() * self.init_scale)


class FourierEncoding(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        # (output_dim - input_dim) must be an integer multiple of input_dim * 2
        assert (output_dim - input_dim) % (input_dim * 2) == 0
        self.input_dim = input_dim
        self.output_dim = output_dim
        n_freqs = (output_dim - input_dim) // (input_dim * 2)
        self.freq_bands = 2. ** torch.linspace(0., 2 * torch.pi, steps=n_freqs)
        
    def forward(self, input_data):
        # Perform the Fourier encodings
        self.freq_bands = self.freq_bands.to(input_data.device)
        encodings = torch.cat((torch.cos(torch.matmul(input_data.unsqueeze(-1), self.freq_bands.unsqueeze(0))),
                                torch.sin(torch.matmul(input_data.unsqueeze(-1), self.freq_bands.unsqueeze(0)))), 
                                dim=-1)
        encodings = encodings.reshape(*encodings.shape[:-2], -1)
        encodings = torch.cat((input_data, encodings), dim=-1)
        return encodings


if __name__ == "__main__":
    # Example usage
    input_dim = 2
    output_dim = 64
    
    embedding = FourierEncoding(input_dim, output_dim)
    input_data = torch.randn(64, input_dim)  # Input data of shape (batch_size, input_dim)
    
    output_embeddings = embedding(input_data)
    print(output_embeddings.shape) 
    ipdb.set_trace()
    pass
