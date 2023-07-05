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


def concat_dict_tensors(dict_tensors, dim=0):
    assert len(dict_tensors) > 0
    # Get keys from the first dictionary in the list
    keys = list(dict_tensors[0].keys())
    
    # Initialize the output dictionary
    output_dict = {}
    
    # Concatenate tensors for each key
    for key in keys:
        # Get the tensors for the current key from all dictionaries
        tensors = [dict_tensor[key] for dict_tensor in dict_tensors]

        # Concatenate the tensors along the specified dimension
        concat_tensor = torch.cat(tensors, dim=dim)
        
        # Assign the concatenated tensor to the output dictionary
        output_dict[key] = concat_tensor
    
    return output_dict


def zeros_like_dict_tensor(dict_tensor):
    # Initialize the output dictionary
    output_dict = {}
    
    # Iterate over the input dictionary
    for key, tensor in dict_tensor.items():
        # Create a new tensor of zeros with the same shape as the input tensor
        zeros_tensor = torch.zeros_like(tensor)
        
        # Assign the zeros tensor to the output dictionary
        output_dict[key] = zeros_tensor
    
    return output_dict

def unsqueeze_dict_tensor(dict_tensor, dim):
    # Initialize the output dictionary
    output_dict = {}

    # Iterate over the input dictionary
    for key, tensor in dict_tensor.items():
        unsqueezed_tensor = tensor.unsqueeze(dim)
        output_dict[key] = unsqueezed_tensor
    
    return output_dict


def clone_dict_tensor(dict_tensor):
    # Initialize the output dictionary
    output_dict = {}

    # Iterate over the input dictionary
    for key, tensor in dict_tensor.items():
        cloned_tensor = tensor.detach().clone()
        output_dict[key] = cloned_tensor
    
    return output_dict


def chunk_dict_tensor(dict_tensor, num_chunks, dim=0):
    # Create an empty dictionary to store the chunked tensors
    chunked_dict_tensors = {}

    # Iterate over the keys and values in the input dictionary
    for key, tensor in dict_tensor.items():
        # Chunk the tensor
        chunks = torch.chunk(tensor, num_chunks, dim=0)

        # Store the list of chunked tensors in the output dictionary
        for i, chunk in enumerate(chunks):
            if not i in chunked_dict_tensors.keys():
                chunked_dict_tensors[i] = {}
            chunked_dict_tensors[i][key] = chunk

    return list(chunked_dict_tensors.values())

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
