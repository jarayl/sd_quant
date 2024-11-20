import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
import torch
import torch.nn.functional as F


class LayerDataCollector():
    def __init__(self, name):
        self.name = name
        self.layer_data = []

    def __call__(self, module, input, output):
        # Collect the output activations
        self.layer_data.append(output.detach().cpu())

def register_data_collector_hooks(model, args=None):
    layer_hooks = []
    layer_num = 0

    for name, m in model.named_modules():
        if args.plot_layer is None or \
           args.plot_layer == layer_num or \
           (isinstance(args.plot_layer, str) and args.plot_layer in name) or \
           (isinstance(args.plot_layer, list) and layer_num in args.plot_layer):
            m.plot = True
            m.plot_this_batch = True

            # Initialize the data collector with the layer name
            hook = LayerDataCollector(name)
            handle = m.register_forward_hook(hook)
            layer_hooks.append((hook, handle))
        layer_num += 1

    return layer_hooks


def plot_activations(model, layer_hooks, i=0, timestep=None, save_dir=None, args=None):
    import matplotlib.pyplot as plt
    import os
    import torch
    import numpy as np

    # Collect data and layer names
    data = []
    layer_names = []
    for (hook, handle) in layer_hooks:
        handle.remove()
        data.append(hook.layer_data)
        layer_names.append(hook.name)

    assert data != []

    # Number of layers to plot
    N = len(data)

    # Set up the figure with one subplot per layer
    fig, axs = plt.subplots(nrows=N, ncols=1, figsize=(10, N * 3))

    # Ensure axs is iterable
    if N == 1:
        axs = [axs]

    for idx, ax in enumerate(axs):
        activations_list = data[idx]
        layer_name = layer_names[idx]

        # Combine all collected activations for the layer (list of tensors)
        activations_tensor = torch.cat(activations_list, dim=0)

        # Convert activations to numpy array
        activations_np = activations_tensor.numpy().flatten()

        # Calculate statistical metrics
        mean_val = np.mean(activations_np)
        std_val = np.std(activations_np)
        min_val = np.min(activations_np)
        max_val = np.max(activations_np)

        # Calculate weights for frequency normalization
        weights = np.ones_like(activations_np) / len(activations_np)

        # Plot histogram of activations with normalized frequency
        bins = 256
        ax.hist(activations_np, bins=bins, weights=weights, log=args.log,
                color='lightgray', alpha=0.7)

        # Add vertical lines for min and max values
        ax.axvline(min_val, color='red', linestyle='dashed', linewidth=1, label=f"Min: {min_val:.2e}")
        ax.axvline(max_val, color='green', linestyle='dashed', linewidth=1, label=f"Max: {max_val:.2e}")

        # Add vertical lines for mean and standard deviations
        ax.axvline(mean_val, color='orange', linestyle='solid', linewidth=1.5, label=f"Mean: {mean_val:.2e}")
        ax.axvline(mean_val - std_val, color='purple', linestyle='dashed', linewidth=1, label=f"-1σ: {(mean_val - std_val):.2e}")
        ax.axvline(mean_val + std_val, color='purple', linestyle='dashed', linewidth=1, label=f"+1σ: {(mean_val + std_val):.2e}")
        ax.axvline(mean_val - 2*std_val, color='gray', linestyle='dashed', linewidth=1, label=f"-2σ: {(mean_val - 2*std_val):.2e}")
        ax.axvline(mean_val + 2*std_val, color='gray', linestyle='dashed', linewidth=1, label=f"+2σ: {(mean_val + 2*std_val):.2e}")

        # Set y-axis label to 'Frequency'
        ax.set_ylabel('Frequency')

        # Set title including layer name, batch number, and timestep
        ax.set_title(f"Layer: {layer_name}, Batch: {i}, Timestep: {timestep}")

        # Add legend to show statistical lines
        ax.legend(loc='upper right', fontsize='small')

    plt.tight_layout()

    # Save or show the plot
    if save_dir is not None:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        filename = f"activations_batch_{i}_timestep_{timestep}.png"
        filepath = os.path.join(save_dir, filename)
        plt.savefig(filepath, dpi=args.plot_dpi if args and hasattr(args, 'plot_dpi') else 100, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()
        plt.close(fig)
