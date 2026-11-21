import torch
import torch.nn as nn

class GModule(nn.Module):
    """
    G module processes the 17-dimensional variant histogram for each gene.
    The weights of this module are shared across all 27,655 genes.
    Architecture: Dropout(0.1) -> Linear(17, 50) -> LayerNorm -> Tanh -> Linear(50, 1)
    """
    def __init__(self):
        super(GModule, self).__init__()
        self.dropout = nn.Dropout(p=0.1)
        self.linear1 = nn.Linear(17, 50)
        self.layer_norm = nn.LayerNorm(50)
        self.tanh = nn.Tanh()
        self.linear2 = nn.Linear(50, 1)

    def forward(self, x):
        # x shape: (batch_size, num_genes, 17)
        # We apply the module to the last dimension (the 17 features)
        # Linear layers in PyTorch naturally apply to the last dimension, 
        # so we can process all genes in parallel as a batch.
        x = self.dropout(x)
        x = self.linear1(x)
        x = self.layer_norm(x)
        x = self.tanh(x)
        x = self.linear2(x)
        # Output shape: (batch_size, num_genes, 1)
        return x

class PModule(nn.Module):
    """
    P module processes the concatenated outputs of the G module.
    Architecture: Dropout(0.2) -> Linear(27655, 50) -> LayerNorm -> Tanh -> 
                  Dropout(0.2) -> Linear(50, 50) -> LayerNorm -> Tanh -> Dropout(0.2)
    Then a final regression layer: Linear(50, 288)
    """
    def __init__(self, num_genes=27655, num_phenotypes=288):
        super(PModule, self).__init__()
        self.num_genes = num_genes
        
        self.layers = nn.Sequential(
            nn.LayerNorm(num_genes), # Added to match the 1,456,599 parameter count reported in the paper (difference is exactly 2 * 27655)
            nn.Dropout(p=0.2),
            nn.Linear(num_genes, 50),
            nn.LayerNorm(50),
            nn.Tanh(),
            nn.Dropout(p=0.2),
            nn.Linear(50, 50),
            nn.LayerNorm(50),
            nn.Tanh(),
            nn.Dropout(p=0.2),
            nn.Linear(50, num_phenotypes)
        )

    def forward(self, x):
        # x shape: (batch_size, num_genes, 1)
        # Flatten the gene dimension
        x = x.view(x.size(0), -1) 
        # x shape: (batch_size, num_genes)
        x = self.layers(x)
        # Output shape: (batch_size, num_phenotypes)
        return x

class Galiana(nn.Module):
    """
    The full Galiana model.
    Takes input of shape (batch_size, num_genes, 17).
    Outputs predictions for 288 phenotypes.
    """
    def __init__(self, num_genes=27655, num_variant_types=17, num_phenotypes=288):
        super(Galiana, self).__init__()
        self.num_genes = num_genes
        self.num_variant_types = num_variant_types
        self.num_phenotypes = num_phenotypes
        
        self.g_module = GModule()
        self.p_module = PModule(num_genes=num_genes, num_phenotypes=num_phenotypes)

    def forward(self, x):
        """
        x: Tensor of shape (batch_size, 27655, 17)
        """
        x = self.g_module(x)
        x = self.p_module(x)
        return x

if __name__ == "__main__":
    # Test instantiation and parameter count
    model = Galiana()
    
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {total_params}")
    
    # Check against the paper's reported parameter count of 1,456,599
    expected_params = 1456599
    print(f"Expected parameters (from paper): {expected_params}")
    if total_params == expected_params:
        print("Parameter count matches exactly!")
    else:
        print(f"Parameter count difference: {total_params - expected_params}")
        
    # Verification forward pass
    batch_size = 4
    sample_input = torch.randn(batch_size, 27655, 17)
    output = model(sample_input)
    print(f"Output shape: {output.shape} (Expected: {batch_size}, 288)")
