import sys
sys.path.append('./')

import argparse, os, sys, glob, importlib, csv
from datetime import datetime
import shutil
import yaml
import copy
import numpy as np

from packaging import version
from omegaconf import OmegaConf
from torch.utils.data import random_split, DataLoader, Dataset, Subset
from functools import partial
from PIL import Image
from einops import rearrange

import torch
import torch.backends.cudnn as cudnn
import torchvision
from torch.utils.tensorboard import SummaryWriter
import pytorch_lightning as pl
import torch.nn.functional as F

from ldm.data.base import Txt2ImgIterableBaseDataset
from ldm.data.coco import CocoImagesAndCaptionsTrain2017, CocoImagesAndCaptionsValidation2017
from ldm.util import instantiate_from_config, str_or_int
from ldm.eval.clip_score import CLIPScore
from ldm.eval.fid_score import FrechetInceptionDistance

parser = argparse.ArgumentParser()

parser.add_argument("--config", nargs="*", metavar="base_config.yaml", help="configs path", default=list())
parser.add_argument('--checkpoint', type=str, default=None, help='checkpoint path')

parser.add_argument('--evaluate', action='store_true', help='evaluate model on testing set')
parser.add_argument('--batch_size', default=None, type=int, help='batch size per GPU (overrides configs in yaml file)')
parser.add_argument("--train_samples", type=int, help="training set size", default=None)
parser.add_argument("--val_samples", type=int, help="validation set size", default=500)
parser.add_argument("--eval_samples", type=int, help="evaluation set size", default=None)
parser.add_argument("--sample_steps", type=int, default=50, help="number of DDIM sampling steps for generating images during validation and testing")
parser.add_argument("--cfg_scale", type=float, default=9.0, help="unconditional guidance scale for validation and testing")
parser.add_argument('--monitor_fid', action='store_true', help='monitor FID for accuracy signal instead of CLIP score')

parser.add_argument('--disable_xformers', action='store_true', help='disable xformers and use vanilla attention implementation instead')
parser.add_argument('--plot', action='store_true', help='visualize model during evaluation loop')
parser.add_argument('--plot_only', action='store_true', help='exit after generating the first plot')
parser.add_argument("--plot_layer", default=None, type=str_or_int, help="when plotting, plot specified layers; provide layer number (e.g., 4), comma-separated list (e.g., 1,2,3,4), or a range (e.g., 1-4)")
parser.add_argument("--plot_log", action='store_true', help="use log-scale when plotting")
parser.add_argument("--save_images", action='store_true', help="save images during evaluation")
parser.add_argument('--print_freq', default=50, type=int, help='print every k batches')
parser.add_argument('--print_model', action='store_true', help='print model modules')
parser.add_argument("--tag", default="", type=str, help="experiment identifier (string to prepend to save directory names")
parser.add_argument('--activations', action='store_true', help='collect activation statistics')
parser.add_argument('--kmean_quantize', action='store_true', help='use kmeans clusting to quantize weights of model')
parser.add_argument('--gaussian_quantize', action='store_true', help='use gaussian specific quantization for weights of model')
parser.add_argument('--norm_before_quantize', action='store_true', help='normalize before quantizing')
parser.add_argument('--dynamic_quantize', action='store_true', help='dynamic quantization of model')
parser.add_argument('--static_quantize', action='store_true', help='static quantization of model')
parser.add_argument('--weight_quantize_bitwidth', default=8, type=int, help='number of bits to quantize weights to (not activations)')
parser.add_argument('--activation_quantize_bitwidth', default=8, type=int, help='number of bits to quantize activations to (not weights)')
parser.add_argument('--rht_quant', action='store_true', help='Quantize with Random Hadamard Transform')

def get_experiment_str(args):
    experiment_str = args.tag + '_' + datetime.now().strftime('%Y-%m-%d_%H-%M/')
    return experiment_str

class WrappedDataset(Dataset):
    """Wraps an arbitrary object with __len__ and __getitem__ into a pytorch dataset"""

    def __init__(self, dataset):
        self.data = dataset

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

def worker_init_fn(_):
    worker_info = torch.utils.data.get_worker_info()

    dataset = worker_info.dataset
    worker_id = worker_info.id

    if isinstance(dataset, Txt2ImgIterableBaseDataset):
        split_size = dataset.num_records // worker_info.num_workers
        # reset num_records to the true number to retain reliable length information
        dataset.sample_ids = dataset.valid_ids[worker_id * split_size:(worker_id + 1) * split_size]
        current_id = np.random.choice(len(np.random.get_state()[1]), 1)
        return np.random.seed(np.random.get_state()[1][current_id] + worker_id)
    else:
        return np.random.seed(np.random.get_state()[1][0] + worker_id)

class DataModuleFromConfig(pl.LightningDataModule):
    def __init__(self, batch_size, train=None, validation=None, test=None, predict=None,
                 wrap=False, num_workers=None, shuffle_test_loader=False, use_worker_init_fn=False,
                 shuffle_val_dataloader=False):
        super().__init__()
        self.batch_size = batch_size
        self.dataset_configs = dict()
        self.num_workers = num_workers if num_workers is not None else batch_size * 2
        self.use_worker_init_fn = use_worker_init_fn
        if train is not None:
            self.dataset_configs["train"] = train
            self.train_dataloader = self._train_dataloader
        if validation is not None:
            self.dataset_configs["validation"] = validation
            self.val_dataloader = partial(self._val_dataloader, shuffle=shuffle_val_dataloader)
        if test is not None:
            self.dataset_configs["test"] = test
            self.test_dataloader = partial(self._test_dataloader, shuffle=shuffle_test_loader)
        if predict is not None:
            self.dataset_configs["predict"] = predict
            self.predict_dataloader = self._predict_dataloader
        self.wrap = wrap

    def prepare_data(self):
        for data_cfg in self.dataset_configs.values():
            instantiate_from_config(data_cfg)

    def setup(self, stage=None):
        self.datasets = dict(
            (k, instantiate_from_config(self.dataset_configs[k]))
            for k in self.dataset_configs)
        if self.wrap:
            for k in self.datasets:
                self.datasets[k] = WrappedDataset(self.datasets[k])

    def _train_dataloader(self):
        is_iterable_dataset = isinstance(self.datasets['train'], Txt2ImgIterableBaseDataset)
        if is_iterable_dataset or self.use_worker_init_fn:
            init_fn = worker_init_fn
        else:
            init_fn = None
        return DataLoader(self.datasets["train"], batch_size=self.batch_size,
                          num_workers=self.num_workers, shuffle=False if is_iterable_dataset else True,
                          worker_init_fn=init_fn)

    def _val_dataloader(self, shuffle=False):
        if isinstance(self.datasets['validation'], Txt2ImgIterableBaseDataset) or self.use_worker_init_fn:
            init_fn = worker_init_fn
        else:
            init_fn = None
        return DataLoader(self.datasets["validation"],
                          batch_size=self.batch_size,
                          num_workers=self.num_workers,
                          worker_init_fn=init_fn,
                          shuffle=shuffle)

    def _test_dataloader(self, shuffle=False):
        is_iterable_dataset = isinstance(self.datasets['train'], Txt2ImgIterableBaseDataset)
        if is_iterable_dataset or self.use_worker_init_fn:
            init_fn = worker_init_fn
        else:
            init_fn = None

        # do not shuffle dataloader for iterable dataset
        shuffle = shuffle and (not is_iterable_dataset)

        return DataLoader(self.datasets["test"], batch_size=self.batch_size,
                          num_workers=self.num_workers, worker_init_fn=init_fn, shuffle=shuffle)

    def _predict_dataloader(self, shuffle=False):
        if isinstance(self.datasets['predict'], Txt2ImgIterableBaseDataset) or self.use_worker_init_fn:
            init_fn = worker_init_fn
        else:
            init_fn = None
        return DataLoader(self.datasets["predict"], batch_size=self.batch_size,
                          num_workers=self.num_workers, worker_init_fn=init_fn)

def evaluate_accuracy(model, dataloader=None, args=None):
    model.eval()
    save_dir = f'./results/{args.experiment_str}/'

    if args.monitor_fid:
        acc_metric = FrechetInceptionDistance().set_dtype(torch.float64).to(args.device)
    else:
        acc_metric = CLIPScore(model_name_or_path="laion/CLIP-ViT-g-14-laion2B-s12B-b42K").to(args.device)

    if args.evaluate and args.save_images and (args.accelerator is None or args.accelerator.is_main_process):
        images_dir = save_dir + 'images/'
        os.makedirs(images_dir, exist_ok=True)

    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            print(f"batch {i} \n")
            # check if we want to plot this batch
            plot = args.plot and (i < 20)
            activations = args.activations

            batch[model.first_stage_key] = batch[model.first_stage_key].to(args.device)
            samples, images, prompts = model.sample_images(batch, batch_idx=i, epoch=None, save_dir=save_dir, plot=plot, activations = activations)

            if (plot and args.plot_only) or activations:
                exit()

            # save the first 10 images, and then save one image every 100 iterations
            if args.evaluate and args.save_images and (args.accelerator is None or args.accelerator.is_main_process) and (i < 10 or i % 100 == 0):
                print("saving image")
                filename = f"{i:05}.png"
                path = os.path.join(images_dir, filename)
                sample = rearrange(samples[0].detach().cpu().numpy(), 'c h w -> h w c')
                Image.fromarray(sample.astype(np.uint8)).save(path)

            # if args.accelerator is not None:
            #     samples = args.accelerator.gather_for_metrics(samples).to(args.device)
            #     prompts = args.accelerator.gather_for_metrics(prompts)
            if args.monitor_fid:
                acc_metric.update(images, real=True)
                acc_metric.update(samples, real=False)
            else:
                acc_metric.update(samples, prompts)

            if args.print_freq != 0 and i != 0 and i % args.print_freq == 0:
                acc_so_far = acc_metric.compute().detach().cpu().numpy()
                print(f'\tbatch {i}/{len(dataloader)}: test acc {acc_so_far:.2f}')

    test_acc = acc_metric.compute().detach().cpu().numpy()

    return test_acc

########################
# Additional Quantization: k-means codebook and gaussian codebook 
########################

def kmeans_clustering(values: torch.Tensor, K: int, num_iters=10):
    """
    Perform k-means clustering on 1D values to find K cluster centers.
    values: [N] tensor
    K: number of clusters
    num_iters: iterations of k-means
    
    Returns:
        centers: [K] cluster centers
    """
    values = values.view(-1)
    # If all values are equal or K=1, shortcut
    if values.numel() == 0:
        return values

    unique_vals = values.unique()
    if unique_vals.numel() <= K:
        # Already less unique values than K clusters
        return unique_vals

    # Initialize centers by random sampling
    indices = torch.randperm(values.numel())[:K]
    centers = values[indices].clone()

    for _ in range(num_iters):
        # Compute distances and assign clusters
        # [N, K]
        dist = (values.unsqueeze(1) - centers.unsqueeze(0))**2
        cluster_ids = dist.argmin(dim=1)

        # Update centers
        for c in range(K):
            mask = (cluster_ids == c)
            if mask.any():
                centers[c] = values[mask].mean()
            else:
                # If cluster empty, re-init from random point
                idx = torch.randint(0, values.numel(), (1,))
                centers[c] = values[idx]

    # Sort centers for consistency
    centers = centers.sort()[0]
    return centers

def kmeans_quantize(tensor: torch.Tensor, num_bits: int):
    """
    Quantize tensor using k-means codebook of size 2^num_bits.
    """
    K = 2**num_bits
    flat = tensor.view(-1)
    centers = kmeans_clustering(flat, K)
    # Assign each value to nearest center
    dist = (flat.unsqueeze(1) - centers.unsqueeze(0))**2
    cluster_ids = dist.argmin(dim=1)
    quantized_flat = centers[cluster_ids]
    return quantized_flat.view_as(tensor)


def gaussian_quantize(tensor: torch.Tensor, num_bits: int):
    """
    Quantize `tensor` using a Gaussian-based codebook approach.
    1. Compute mean and std of the data.
    2. Determine codebook by partitioning the Gaussian distribution.
    3. Map each value to the nearest centroid.
    """
    K = 2 ** num_bits
    flat = tensor.view(-1)
    mean = flat.mean()
    std = flat.std() + 1e-8  # avoid div-by-zero

    # If all values are nearly identical
    if std < 1e-12:
        return torch.full_like(tensor, mean), mean.unsqueeze(0)

    # Compute codebook by quantile
    # The inverse CDF of standard normal can be approximated by torch.special.erfinv or define your own
    # For simplicity, use a numerical approximation:
    def normal_icdf(p):
        # approximate inverse CDF for standard normal using torch.erfinv
        # CDF(x) = 0.5*(1+erf(x/sqrt(2)))
        # p = 0.5*(1+erf(x/sqrt(2))) => erf(x/sqrt(2)) = 2p-1
        # x = sqrt(2)*erfinv(2p-1)
        return np.sqrt(2) * torch.erfinv(torch.tensor(2*p-1, device=flat.device))

    # Compute codebook centers
    # We want equally spaced quantiles in [0,1], so:
    # q_k = (k - 0.5)/K for k=1,...,K
    codebook = []
    for k in range(1, K+1):
        q = (k - 0.5)/K
        z = normal_icdf(q)
        c = mean + std * z
        codebook.append(c)
    codebook = torch.stack(codebook)  # [K]

    # Assign each value to the nearest codebook center
    # One approach: sorting codebook and binary searching, 
    # but we can do directly by distance since K might not be huge
    # For large K, consider a more efficient search
    dist = (flat.unsqueeze(1) - codebook.unsqueeze(0))**2
    cluster_ids = dist.argmin(dim=1)
    quantized_flat = codebook[cluster_ids]
    return quantized_flat.view_as(tensor)


########################
# Main Quantization Functions
########################
def quantize_weights(model, args):
    """
    Main weight quantization function that handles different quantization modes.
    """
    if args.rht_quant:
        print(f"Applying RHT quantization (W{args.weight_quantize_bitwidth}A{args.activation_quantize_bitwidth} bits) to the model's Linear layers...")
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Linear):
                with torch.no_grad():
                    # Apply random hadamard rotation and quantize weights
                    N = module.in_features
                    
                    sign_vector = generate_random_sign_vector(N, module.weight.device, module.weight.dtype)
                    
                    weight = module.weight.data  # Shape: [out_features, in_features]

                    # Pad if necessary
                    N_padded = 2 ** int(np.ceil(np.log2(N))) if (N & (N - 1)) != 0 else N
                    N_padded = N_padded
                    pad_size = N_padded - N

                    if pad_size > 0:
                        weight = F.pad(weight, (0, pad_size))  # Pad last dimension (input dimension)
                        sign_vector = F.pad(sign_vector, (0, pad_size), value=1)
                    else:
                        sign_vector = sign_vector
                    
                    # Apply Hadamard transform to weights: 
                    weight_prime = fwht(weight * sign_vector)
                    
                    if args.norm_before_quantize:
                        print ("Normalizing RHT transformed weights")
                        # Normalization Step:
                        # Compute mean and std
                        mean = weight_prime.mean()
                        std = weight_prime.std(unbiased=False)
                        if std < 1e-8:
                            # If std is tiny, fallback: just quantize directly
                            # or treat all weights as a constant
                            std = 1.0

                        # Normalize: (W - mean) / std
                        weight_prime = (weight_prime - mean) / std

                    if args.kmean_quantize:
                        print("Using kmeans clusting to quantize weights")
                        deq_w = kmeans_quantize(weight_prime, args.weight_quantize_bitwidth)
                    elif args.gaussian_quantize:
                        print("Using gaussian codebook to quantize weights")
                        deq_w = gaussian_quantize(weight_prime, args.weight_quantize_bitwidth)
                    else:
                        print ("Using standard minmax quantzation for weights")
                        deq_w = quantize_tensor_min_max(weight_prime, args.weight_quantize_bitwidth, signed=True)
                    
                    # Store transformed and quantized model weights
                    module.weight.data = deq_w

                    # Quantize and dequantize bias if present
                    if module.bias is not None:
                        deq_b = quantize_tensor_min_max(module.bias.data, args.weight_quantize_bitwidth, signed=True)
                        module.bias.data.copy_(deq_b)
                    
                    # Store sign_vector and padding info in the module
                    # We'll need this during activation quantization
                    module.register_buffer('rht_sign_vector', sign_vector)
                    module.rht_N_padded = N_padded
    else:
        # Use default minmax Uniform quantization or static quantization, depending on args
        print(f"Applying {'dynamic' if args.dynamic_quantize else 'static'} quantization (W{args.weight_quantize_bitwidth} bits) to the model's Linear and Conv2d layers...")
        for name, module in model.named_modules():
            if isinstance(module, (torch.nn.Linear, torch.nn.Conv2d)):
                with torch.no_grad():
                    # Quantize and dequantize weights to simulate quantization
                    deq_w = quantize_tensor_min_max(module.weight.data, args.weight_quantize_bitwidth, signed=True)
                    module.weight.data.copy_(deq_w)

                    # Quantize and dequantize bias if present
                    if module.bias is not None:
                        deq_b = quantize_tensor_min_max(module.bias.data, args.weight_quantize_bitwidth, signed=True)
                        module.bias.data.copy_(deq_b)

def quantize_activations(model, args, activation_ranges=None):
    """
    Main activation quantization function that handles different quantization modes.
    """
    if args.rht_quant:
        print(f"Applying RHT incoherence processing and dynamic activation quantization (A{args.activation_quantize_bitwidth} bits)...")
        register_activation_quantization_hooks(model, args, activation_ranges=None)
    else:
        if args.dynamic_quantize:
            print(f"Applying dynamic activation quantization (A{args.activation_quantize_bitwidth} bits)...")
            register_activation_quantization_hooks(model, args, activation_ranges=None)
        elif args.static_quantize:
            print(f"Applying static activation quantization (A{args.activation_quantize_bitwidth} bits)...")
            if activation_ranges is None:
                raise ValueError("Activation ranges must be provided for static quantization.")
            register_activation_quantization_hooks(model, args, activation_ranges)
        else:
            pass  # No activation quantization

########################
# Helper Functions for Quantization:
########################
def quantize_tensor_min_max(tensor, num_bits=8, signed=True):
    """
   Simple uniform quantization function (either signed or unsigned)
    """
    if signed:
        qmin = - (2 ** (num_bits - 1))
        qmax = (2 ** (num_bits - 1)) - 1
        zero_point = 0
        max_abs = tensor.abs().max()
        if max_abs == 0:
            scale = 1.0
        else:
            scale = max_abs / qmax
    else:
        qmin = 0
        qmax = (2 ** num_bits) - 1
        min_val = tensor.min()
        max_val = tensor.max()
        if min_val == max_val:
            scale = 1.0
            zero_point = 0
        else:
            scale = (max_val - min_val) / (qmax - qmin)
            zero_point = qmin - min_val / scale

    # Quantize
    q_x = ((tensor / scale) + zero_point).round().clamp(qmin, qmax)

    # Dequantize
    deq_x = (q_x - zero_point) * scale

    return deq_x


def quantize_and_dequantize_tensor(tensor, scale, zero_point, qmin, qmax):
    """
    Quantizes and immediately dequantizes a tensor using provided scale and zero_point.
    """
    q_x = ((tensor / scale) + zero_point).round().clamp(qmin, qmax)
    deq_x = (q_x - zero_point) * scale
    return deq_x

def register_activation_quantization_hooks(model, args, activation_ranges=None):
    """
    Register activation quantization hooks for random hadamard transformation, dynamic, and static quantization.
    """
    module_to_name = {}
    handles = []

    # Build module_to_name mapping
    for name, module in model.named_modules():
        module_to_name[module] = name

    def rht_activation_pre_hook(module, input):
        # This pre-hook transforms the input activations using the RHT logic
        # only if rht_sign_vector exists in the module.
        if not hasattr(module, 'rht_sign_vector'):
            return input
        
        x = input[0]
        N = module.in_features
    
        sign_vector = module.rht_sign_vector
        N_padded = module.rht_N_padded
        pad_size = N_padded - N

        # Pad input if needed
        if pad_size > 0:
            x = F.pad(x, (0, pad_size))

        # Apply Hadamard transform
        x = fwht(x * sign_vector)

        if args.norm_before_quantize:
            print ("Normalizing RHT transformed activations")
            # Normalization Step:
            # Compute mean and std
            mean = x.mean()
            std = x.std(unbiased=False)
            if std < 1e-8:
                # If std is tiny, fallback: just quantize directly
                # or treat all activations as a constant
                std = 1.0

            # Normalize: (X - mean) / std
            x = (x - mean) / std

        # Quantize + dequantize activations
        x = quantize_tensor_min_max(x, num_bits=args.activation_quantize_bitwidth, signed=True)

        return (x,)
    
    # Define hook for activation quantization
    def activation_quantization_hook(module, input, output):
        module_name = module_to_name[module]

        if args.dynamic_quantize:
            # Use standard minmax Uniform quantization
            deq_x = quantize_tensor_min_max(output, num_bits=args.activation_quantize_bitwidth, signed=False)
            return deq_x
        elif args.static_quantize:
            # Retrieve quantization parameters
            params = activation_ranges.get(module_name, None)
            if params is None:
                return output

            scale = params['scale']
            zero_point = params['zero_point']
            qmin = params['qmin']
            qmax = params['qmax']

            # Quantize with calibrated parameters
            deq_x = quantize_and_dequantize_tensor(output, scale, zero_point, qmin, qmax)
            return deq_x
        else:
            return output

    # Register hooks for specific layer types
    if args.rht_quant:
        print("Registering RHT hooks for linear layers...")
        # For RHT, we do a forward_pre_hook to handle input transformations before the linear layer
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Linear):
                # Add forward pre-hook for RHT transform
                handle = module.register_forward_pre_hook(rht_activation_pre_hook)
                handles.append(handle)
    else:
        print ("Registering hooks for normal dynamic/static quantization...")
        # Add forward hook to all linear and Conv2d layers
        for name, module in model.named_modules():
            if isinstance(module, (torch.nn.Linear, torch.nn.Conv2d)):
                handle = module.register_forward_hook(activation_quantization_hook)
                handles.append(handle)

    return handles

def collect_activation_ranges(model, dataloader, num_batches, args):
    """
    Helper function to collect activation data for static quantization parameter calibration.
    """
    activation_ranges = {}
    module_to_name = {}
    handles = []

    # Build module_to_name mapping
    for name, module in model.named_modules():
        module_to_name[module] = name

    # Define hook to collect activation ranges
    def collect_activation_ranges_hook(module, input, output):
        module_name = module_to_name[module]

        # Handle tensor output
        if isinstance(output, torch.Tensor):
            min_val = output.min().item()
            max_val = output.max().item()
            if module_name not in activation_ranges:
                activation_ranges[module_name] = {'min': min_val, 'max': max_val}
            else:
                activation_ranges[module_name]['min'] = min(activation_ranges[module_name]['min'], min_val)
                activation_ranges[module_name]['max'] = max(activation_ranges[module_name]['max'], max_val)

    # Register hooks
    for name, module in model.named_modules():
        if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear)):
            handle = module.register_forward_hook(collect_activation_ranges_hook)
            handles.append(handle)

    # Run calibration
    model.eval()
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if i >= num_batches:
                break

            try:
                # Get x_start and ensure it's in NCHW format
                x_start = batch[model.first_stage_key].to(args.device)
                if x_start.ndim == 4 and x_start.shape[1] != 3 and x_start.shape[3] == 3:
                    # Convert from NHWC to NCHW
                    x_start = x_start.permute(0, 3, 1, 2)
                elif x_start.ndim == 3 and x_start.shape[2] == 3:
                    # Handle case where batch dimension is missing
                    x_start = x_start.permute(2, 0, 1).unsqueeze(0)
                elif x_start.ndim != 4:
                    print(f"Unexpected x_start shape: {x_start.shape}")
                    continue

                # Encode x_start
                x_start = model.get_first_stage_encoding(model.encode_first_stage(x_start))

                # Prepare conditioning
                prompts = batch['caption']
                conditioning = model.get_learned_conditioning(prompts)

                # Generate noise
                noise = torch.randn_like(x_start)

                # Get timesteps
                t = torch.randint(0, model.num_timesteps, (x_start.shape[0],), device=args.device).long()

                # Add noise to x_start to get x_noisy
                x_noisy = model.q_sample(x_start=x_start, t=t, noise=noise)

                # Apply model to get noise_pred
                noise_pred = model.apply_model(x_noisy, t, conditioning)

            except Exception as e:
                print(f"Error processing batch {i}: {e}")
                continue

    # Remove hooks
    for handle in handles:
        handle.remove()

    return activation_ranges

def compute_quantization_parameters(activation_ranges, num_bits):
    """
    Compute quantization parameters using standard uniform quantization for each module
    """
    quantization_params = {}
    qmin = 0
    qmax = 2 ** num_bits - 1

    for module_name, ranges in activation_ranges.items():
        min_val = ranges['min']
        max_val = ranges['max']

        # Compute scale
        act_range = max_val - min_val
        if act_range == 0:
            scale = 1.0
            zero_point = 0
        else:
            scale = act_range / (qmax - qmin)
            zero_point = qmin - min_val / scale
            zero_point = np.clip(zero_point, qmin, qmax)

        quantization_params[module_name] = {
            'scale': float(scale),
            'zero_point': int(zero_point),
            'qmin': qmin,
            'qmax': qmax
        }

    return quantization_params

########################
# Random Hadamard Transformation Helper Functions
########################
def generate_random_sign_vector(length, device, dtype):
    """Generate a random sign vector (+1 or -1) of given length."""
    return torch.randint(0, 2, (length,), device=device, dtype=dtype) * 2 - 1  # Random +/-1

def fwht(x):
    """
    Fast Walsh-Hadamard Transform along the last dimension without normalization.
    """
    original_shape = x.shape
    N = x.shape[-1]
    x = x.reshape(-1, N)
    batch_dim, d = x.shape
    h = 2
    while h <= d:
        hf = h // 2
        x = x.view(batch_dim, d // h, h)

        half_1, half_2 = x[:, :, :hf], x[:, :, hf:]

        x = torch.cat((half_1 + half_2, half_1 - half_2), dim=-1)

        h *= 2

    return (x / np.sqrt(d)).view(*original_shape)


########################
# Main Execution
########################
if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("No GPUs found!")
        exit()

    args, unknown = parser.parse_known_args()

    os.environ["USE_XFORMERS"] = '0' if args.disable_xformers else '1' # TODO: implement option to disable xformers
    os.environ["ATTN_PRECISION"] = "fp32"

    # TODO: try enforcing deterministic behavior by setting seeds and using torch.use_deterministic_algorithms

    # TODO: we should support multi-GPU runs with HF accelerate or torchrun once we get there
    # if args.accelerate:
    #     from accelerate import Accelerator
    #     args.accelerator = Accelerator()
    #     args.device = args.accelerator.device
    #     args.num_gpus = args.accelerator.state.num_processes
    #     os.environ["TOKENIZERS_PARALLELISM"] = "false"
    # else:
    #     args.num_gpus = 1

    args.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    args.accelerator = None

    # init configs
    configs = [OmegaConf.load(cfg) for cfg in args.config]
    cli = OmegaConf.from_dotlist(unknown)
    config = OmegaConf.merge(*configs, cli)

    # update config from parsed arguments
    if args.train_samples is not None:
        config.data.params.train.params.dataset_size = args.train_samples
    if args.val_samples is not None:
        config.data.params.validation.params.dataset_size = args.val_samples
    if args.eval_samples is not None:
        config.data.params.test.params.dataset_size = args.eval_samples

    config.data.params.batch_size = args.batch_size if args.batch_size is not None else config.data.params.batch_size
    # args.lr = args.lr if args.lr is not None else config.model.base_learning_rate
    # config.model.base_learning_rate = args.lr

    args.config = config
    batch_size, base_lr = config.data.params.batch_size, config.model.base_learning_rate

    # model
    print(f'\nCreating Stable Diffusion v2.1 model\n')

    model = instantiate_from_config(config.model)
    model.learning_rate = base_lr

    # data
    data = instantiate_from_config(config.data)
    data.prepare_data()
    data.setup()
    print("\nLoading datasets:")
    try:
        for k in data.datasets:
            print(f"  - {k}, {data.datasets[k].__class__.__name__}, {len(data.datasets[k])}")
    except:
        print("datasets not yet initialized.")

    train_dataloader, val_dataloader, test_dataloader = data._train_dataloader(), data._val_dataloader(), data._test_dataloader()

    # load from checkpoint
    if args.checkpoint is None or not os.path.isfile(args.checkpoint):
        print("\nNo checkpoint found! Proceeding without checkpoint.")
    else:
        print(f'\nLoading model from {args.checkpoint}\n')
        checkpoint = torch.load(args.checkpoint, map_location="cpu")
        model.load_state_dict(checkpoint["state_dict"], strict=False)

    if args.print_model:
        with open("layers.txt", 'w') as f:
            layer_num = 0
            for name, module in model.model.named_modules():
                f.write(f"{layer_num}: {name}\n")
                layer_num += 1
        # print(model)

    args.experiment_str = get_experiment_str(args)

    args.calibrating_range = False # this tells DDIM sampler when to print clip_vals
    model.args = args

    # Apply quantization before moving the model to GPU
    if args.static_quantize:
        model = model.to(args.device)
        num_calibration_batches = 10
        print(f"Collecting activation ranges for static quantization using {num_calibration_batches} batches...")
        activation_ranges = collect_activation_ranges(model, train_dataloader, num_calibration_batches, args)
        quantization_params = compute_quantization_parameters(activation_ranges, args.activation_quantize_bitwidth)
        quantize_weights(model.model, args)
        quantize_activations(model.model, args, activation_ranges=quantization_params)
    elif args.dynamic_quantize:
        quantize_weights(model.model, args)
        quantize_activations(model.model, args)
        model = model.to(args.device)
    else:
        model = model.to(args.device)

    if args.evaluate:
        iter_str = f' for {args.eval_samples} samples' if args.eval_samples is not None else ''
        print(f'\n\n\nEvaluating model on COCO validation dataset{iter_str}\n\n')
        test_acc = evaluate_accuracy(model, dataloader=test_dataloader, args=args)
        print(f'\nEvaluation accuracy: {test_acc:.2f}')
