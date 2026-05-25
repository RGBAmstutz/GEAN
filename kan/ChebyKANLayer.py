import torch
import torch.nn as nn


# This is inspired by Kolmogorov-Arnold Networks but using Chebyshev polynomials instead of splines coefficients
class ChebyKANLayer(nn.Module):
    def __init__(self, input_dim, output_dim, degree, embed_dim=None):
        super(ChebyKANLayer, self).__init__()
        self.degree = degree
        self.embed_dim = embed_dim
        self.cheby_coeffs = nn.Parameter(torch.empty(input_dim, output_dim, degree + 1))
        nn.init.normal_(self.cheby_coeffs, mean=0.0, std=1 / (input_dim * (degree + 1)))
        self.register_buffer("arange", torch.arange(0, degree + 1, 1))

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.cheby_coeffs, gain=1e-9)
    # END KAN

    def forward(self, x):
        # normalize x to [-1, 1] using tanh
        x = torch.tanh(x)
        x = x.unsqueeze(-1)
        if self.embed_dim:
            x = x.expand(-1, -1, -1, -1, self.degree + 1)
        else:
            x = x.expand(-1, -1, -1, self.degree + 1)
        x = x.acos()
        # mult by arange
        x = x * self.arange
        x = x.cos()
        # apply learned chebyshev coeffs
        if self.embed_dim:
            y = torch.einsum("btsed,sod->btoe", x, self.cheby_coeffs)
        else:
            y = torch.einsum("btsd,sod->bto", x, self.cheby_coeffs)  # adapted for larger dim

        return y
