import torch
import torch.nn as nn

class BKANLayer(nn.Module):
    def __init__(self, input_dim, output_dim, degree, embed_dim=None):
        """
        Bernstein-Kolmogorov-Arnold Network layer.

        Args:
            input_dim (int): Number of input features per location.
            output_dim (int): Number of output features per location.
            degree (int): Degree of the Bernstein polynomial.
            embed_dim (int, optional): Additional embedding dim (for higher-order feature interactions).
        """
        super(BKANLayer, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.degree = degree
        self.embed_dim = embed_dim

        # Learnable Bernstein coefficients: (input_dim, output_dim, degree+1)
        self.bern_coeffs = nn.Parameter(torch.empty(input_dim, output_dim, degree + 1))
        nn.init.normal_(self.bern_coeffs, mean=0.0, std=1 / (input_dim * (degree + 1)))
        self.register_buffer("arange", torch.arange(0, degree + 1, dtype=torch.float32))

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.bern_coeffs, gain=1e-9)

    def forward(self, x):
        """
        Forward pass of BKANLayer.

        Args:
            x (Tensor): Input tensor of shape (..., input_dim)

        Returns:
            Tensor: Output tensor of shape (..., output_dim)
        """
        # Normalize input to [0, 1] for Bernstein basis
        x = torch.sigmoid(x)  # shape (..., input_dim)

        # Get original shape for later restoration
        orig_shape = x.shape[:-1]  # everything except input_dim

        # Add degree dimension: (..., input_dim, degree+1)
        x = x.unsqueeze(-1)

        # Compute Bernstein basis functions
        powers = self.arange.to(x.device)  # shape (degree+1,)
        # (..., input_dim, degree+1)
        bern_basis = (
            torch.combinations(self.arange, self.degree + 1).to(x.device)
            * (x ** powers) * ((1 - x) ** (self.degree - powers))
        )

        # Contract over input_dim and degree
        # bern_basis (..., input_dim, degree+1)
        # bern_coeffs (input_dim, output_dim, degree+1)
        # Result: (..., output_dim)
        y = torch.einsum("...ied,ijd->...je", bern_basis, self.bern_coeffs)
        #y = torch.einsum("...id,ijd->...j", bern_basis, self.bern_coeffs) # use if smaller size input

        return y
