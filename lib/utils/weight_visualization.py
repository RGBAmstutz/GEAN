import matplotlib
import matplotlib.pyplot as plt
import torch
import seaborn as sns
import re
#matplotlib.use('Agg')

def save_plot(filename="weights.png"):
    plt.savefig(filename, bbox_inches="tight")
    plt.close()  # close the figure to avoid memory issues

import os

def save_plot_unique(basename="weights", out_dir="plots"):
    os.makedirs(out_dir, exist_ok=True)  # create folder if it doesn't exist
    stripped = re.sub(r'\s+', '', basename)
    i = 1
    while True:
        filename = f"{out_dir}/{stripped}_{i}.png"
        if not os.path.exists(filename):
            break
        i += 1
    plt.title(f"Weights for {basename} #{i}", fontsize=14)

    plt.savefig(filename, bbox_inches="tight")
    plt.close()
    print(f"Saved: {filename}")


import matplotlib.pyplot as plt
import os

def visualize_conv_weights_descriptive(weights, layer_name="conv_layer", out_dir="plots"):
    """
    Visualizes Conv2D weights as grids of kernels with labeled axes and colorbar.
    """
    os.makedirs(out_dir, exist_ok=True)
    filename = f"{out_dir}/{layer_name.replace('.', '_')}.png"

    # Normalize weights for visualization
    weights = weights.detach().cpu()
    weights_min, weights_max = weights.min(), weights.max()
    weights = (weights - weights_min) / (weights_max - weights_min)

    num_kernels = weights.shape[0]  # out_channels
    num_cols = 8  # number of kernels per row
    num_rows = (num_kernels + num_cols - 1) // num_cols

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(num_cols * 1.5, num_rows * 1.5))
    axes = axes.flatten()

    for i in range(len(axes)):
        ax = axes[i]
        if i < num_kernels:
            # Display the first channel of each kernel
            kernel = weights[i, 0].numpy()
            img = ax.imshow(kernel, cmap="viridis", interpolation="nearest")
            ax.set_title(f"Kernel {i}", fontsize=8)
        ax.axis("off")  # Hide ticks and labels for subplots

    # Add a colorbar for magnitude
    fig.subplots_adjust(right=0.85)  # make space for colorbar
    cbar_ax = fig.add_axes([0.88, 0.15, 0.03, 0.7])
    cbar = fig.colorbar(img, cax=cbar_ax)
    cbar.set_label("Normalized Weight Magnitude", fontsize=10)

    # Add a main title
    #fig.suptitle(f"Weights for {layer_name}", fontsize=14)
    save_plot_unique(basename=layer_name)


def visualize_conv_weights(weights, num_cols=8):
    # Normalize weights to [0,1] for visualization
    weights = weights.clone().detach()
    weights -= weights.min()
    weights /= weights.max()

    num_kernels = weights.shape[0]
    num_rows = (num_kernels + num_cols - 1) // num_cols

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(num_cols, num_rows))
    for i in range(num_rows * num_cols):
        ax = axes[i // num_cols, i % num_cols]
        if i < num_kernels:
            # Select the first channel for RGB/Grayscale filters
            kernel = weights[i, 0].cpu().numpy()
            ax.imshow(kernel, cmap='viridis')
        ax.axis('off')
    #plt.show()
    save_plot_unique(basename='conv_weights')

def visualize_fc_weights(weights):
    weights = weights.clone().detach().cpu().numpy()
    plt.figure(figsize=(10, 8))
    sns.heatmap(weights, cmap='coolwarm', center=0)
    #plt.show()
    save_plot_unique(basename='fc_weights')
    
def visualize_fc_weights_descriptive(weights, layer_name="fc_layer", out_dir="plots"):
    """
    Visualizes Linear layer weights as a heatmap with descriptive labels.
    """
    os.makedirs(out_dir, exist_ok=True)
    #filename = f"{out_dir}/{layer_name.replace('.', '_')}.png"

    weights = weights.detach().cpu().numpy()

    plt.figure(figsize=(8, 6))
    img = plt.imshow(weights, cmap="coolwarm", aspect="auto", interpolation="nearest")
    plt.xlabel("Input Features")
    plt.ylabel("Output Neurons")

    # Colorbar with label
    cbar = plt.colorbar(img)
    cbar.set_label("Weight Magnitude", fontsize=10)

    save_plot_unique(basename=layer_name)

def visualize_layer_weights(layer):
    weights = layer.weight.data
    if len(weights.shape) == 4:  # Conv2d: [out_channels, in_channels, H, W]
        visualize_conv_weights(weights)
    elif len(weights.shape) == 2:  # Linear: [out_features, in_features]
        visualize_fc_weights(weights)
    else:
        print(f"Unsupported layer shape: {weights.shape}")
        
def visualize_subnetwork_weights(subnet):
    for name, module in subnet.named_children():
        if hasattr(module, "weight"):
            print(f"Visualizing weights of {name}: {module.__class__.__name__}")
            weights = module.weight.data
            if len(weights.shape) == 4:  # Conv2d: [out_channels, in_channels, H, W]
                visualize_conv_weights(weights)
            elif len(weights.shape) == 2:  # Linear: [out_features, in_features]
                visualize_fc_weights(weights)
            else:
                print(f"Unsupported shape {weights.shape} for {name}")


def visualize_weights_recursive(module, parent_name=""):
    for name, child in module.named_children():
        full_name = f"{parent_name}.{name}" if parent_name else name

        # If this child has weights, visualize them
        if hasattr(child, "weight") and child.weight is not None:
            print(f"Visualizing weights for: {full_name} | Shape: {tuple(child.weight.shape)}")
            if len(child.weight.shape) == 4:  # Conv2d
                visualize_conv_weights_descriptive(child.weight)
            elif len(child.weight.shape) == 2:  # Linear
                visualize_fc_weights_descriptive(child.weight, layer_name='Frequency Network')
            else:
                print(f"Skipping {full_name}: unsupported shape {child.weight.shape}")

        # Recurse into this child
        visualize_weights_recursive(child, parent_name=full_name)


class WeightVisualizer:
    def __init__(self, device="cpu"):
        self.device = device

    def visualize(self, module):
        if isinstance(module, torch.nn.Sequential) or hasattr(module, "children"):
            for name, layer in module.named_children():
                self._visualize_layer(layer, name)
        else:
            self._visualize_layer(module, "Layer")

    def _visualize_layer(self, layer, name):
        if hasattr(layer, "weight"):
            weights = layer.weight.data.to(self.device)
            print(f"Visualizing {name}: {layer.__class__.__name__} | Shape: {tuple(weights.shape)}")
            if len(weights.shape) == 4:  # Conv2d
                visualize_conv_weights(weights)
            elif len(weights.shape) == 2:  # Linear
                visualize_fc_weights(weights)
            else:
                print(f"Unsupported shape {weights.shape} for {name}")
