import torch
import numpy as np
from scipy import stats
from torch.nn.functional import adaptive_avg_pool2d
from pytorch_fid.inception import InceptionV3
from pytorch_fid.fid_score import calculate_frechet_distance
from tqdm.auto import tqdm
from einops import repeat, rearrange
from torch.utils.data import Dataset, DataLoader
import argparse
import os
import json 
from datetime import datetime
from pathlib import Path
from PIL import Image
from torchvision import transforms as T
from omegaconf import OmegaConf
from sgm.util import instantiate_from_config

# Add safe globals for PyTorch 2.6+ compatibility
try:
    torch.serialization.add_safe_globals([
        numpy.core.multiarray.scalar,
        numpy._core.multiarray.scalar,
        numpy.dtype,
        numpy.ndarray
    ])
except (AttributeError, NameError):
    # Fallback for older PyTorch versions or when globals don't exist
    pass

class EvalDataset(Dataset):
    def __init__(self, folder, image_size, exts=['jpg', 'jpeg', 'png', 'tiff']):
        super().__init__()
        self.folder = folder
        self.image_size = image_size
        self.paths = [p for ext in exts for p in Path(f'{folder}').glob(f'**/*.{ext}')]
        self.transform = T.Compose([
            T.Resize(image_size),
            T.CenterCrop(image_size),
            T.ToTensor()
        ])
    
    def __len__(self):
        return len(self.paths)
    
    def __getitem__(self, index):
        path = self.paths[index]
        img = Image.open(path)
        img = img.convert('RGB')
        return self.transform(img)

def load_model_from_config(config, ckpt):
    print(f"Loading model from {ckpt}")
    
    # Load checkpoint
    if ckpt.endswith("ckpt"):
        try:
            # First try with weights_only=True (safer)
            pl_sd = torch.load(ckpt, map_location="cpu", weights_only=True)
            state_dict = pl_sd["state_dict"] if "state_dict" in pl_sd else pl_sd
            global_step = pl_sd.get("global_step", 0) if isinstance(pl_sd, dict) and "global_step" in pl_sd else 0
        except Exception as e:
            print(f"Failed to load with weights_only=True: {e}")
            print("Trying with weights_only=False (make sure checkpoint is from trusted source)")
            # Fallback to weights_only=False for compatibility
            pl_sd = torch.load(ckpt, map_location="cpu", weights_only=False)
            state_dict = pl_sd["state_dict"] if "state_dict" in pl_sd else pl_sd
            global_step = pl_sd.get("global_step", 0) if isinstance(pl_sd, dict) and "global_step" in pl_sd else 0
    elif ckpt.endswith("safetensors"):
        from safetensors.torch import load_file as load_safetensors
        state_dict = load_safetensors(ckpt)
        global_step = 0  # safetensors doesn't store global_step
    else:
        raise NotImplementedError(f"Checkpoint format not supported: {ckpt}")
    
    # Initialize model using the new DiffusionEngine structure
    model = instantiate_from_config(config.model)
    
    # Load state dict
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print(f"Loaded model with {len(missing)} missing and {len(unexpected)} unexpected keys")
    if len(missing) > 0:
        print(f"Missing Keys: {missing}")
    if len(unexpected) > 0:
        print(f"Unexpected Keys: {unexpected}")
    
    model.cuda()
    model.eval()
    return model, global_step

class DiffusionEvaluator:
    def __init__(self, config_path, model_path, dataset_path, batch_size=10, num_samples=50000, 
                 image_size=512, num_inference_steps=50, guidance_scale=7.5, 
                 device='cuda' if torch.cuda.is_available() else 'cpu', 
                 save_generated_images=False, samples_dir='./generated_samples'):
        self.config_path = config_path
        self.model_path = model_path
        self.dataset_path = dataset_path
        self.batch_size = batch_size
        self.num_samples = num_samples
        self.device = device
        self.image_size = image_size
        self.num_inference_steps = num_inference_steps
        self.guidance_scale = guidance_scale
        self.save_generated_images = save_generated_images
        self.samples_dir = samples_dir
        
        if self.save_generated_images:
            os.makedirs(self.samples_dir, exist_ok=True)
        
        # Load config and model
        self.config = OmegaConf.load(self.config_path)
        self.model, self.global_step = load_model_from_config(self.config, self.model_path)
        
        # Initialize inception model for evaluation
        self.inception_model = InceptionV3([InceptionV3.BLOCK_INDEX_BY_DIM[2048]]).to(device)
        self.real_features = None
        self.load_real_features()
    
    def get_inception_features(self, images):
        """Extract inception features from images."""
        # Ensure images are in [0, 1] range
        if images.min() < 0:
            images = (images + 1) / 2  # Normalize from [-1, 1] to [0, 1]
        
        self.inception_model.eval()
        with torch.no_grad():
            features = self.inception_model(images)[0]
        
        if features.size(2) != 1 or features.size(3) != 1:
            features = adaptive_avg_pool2d(features, output_size=(1, 1))
        features = rearrange(features, "... 1 1 -> ...")
        return features
    
    def load_real_features(self):
        """Load and compute features for real images."""
        dataset = EvalDataset(self.dataset_path, self.image_size)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False, 
                               num_workers=4, drop_last=False)
        
        n_samples_needed = min(self.num_samples, len(dataset))
        n_batches_needed = int(np.ceil(n_samples_needed / self.batch_size))
        
        features_list = []
        
        with torch.no_grad():
            for i, batch in enumerate(tqdm(dataloader, desc="Processing real images")):
                if i >= n_batches_needed:
                    break
                
                batch = batch.to(self.device)
                features = self.get_inception_features(batch)
                features_list.append(features.cpu())
        
        self.real_features = torch.cat(features_list, dim=0)[:n_samples_needed]
        print(f"Processed {self.real_features.shape[0]} real images")
    
    def calculate_inception_score(self, features, splits=10):
        """Calculate Inception Score from features."""
        scores = []
        subset_size = features.shape[0] // splits
        
        for k in range(splits):
            subset = features[k * subset_size: (k + 1) * subset_size]
            prob = torch.nn.functional.softmax(subset, dim=1)
            prob = prob.cpu().numpy()
            
            p_y = np.mean(prob, axis=0)
            kl_d = prob * (np.log(prob + 1e-10) - np.log(p_y + 1e-10))
            kl_d = np.mean(np.sum(kl_d, axis=1))
            scores.append(np.exp(kl_d))
        
        return np.mean(scores)
    
    def calculate_fid(self, real_features, fake_features):
        """Calculate FID score between real and fake features."""
        mu1 = np.mean(fake_features.cpu().numpy(), axis=0)
        sigma1 = np.cov(fake_features.cpu().numpy(), rowvar=False)
        mu2 = np.mean(real_features.cpu().numpy(), axis=0)
        sigma2 = np.cov(real_features.cpu().numpy(), rowvar=False)
        return calculate_frechet_distance(mu1, sigma1, mu2, sigma2)
    
    def calculate_precision_recall(self, real_features, fake_features, k=3):
        """Calculate precision and recall metrics."""
        real_features = real_features.cpu()
        fake_features = fake_features.cpu()
        
        # Calculate distances for real images
        real_dists = torch.cdist(real_features, real_features, p=2)
        real_dists.fill_diagonal_(float('inf'))
        kth_dists, _ = real_dists.kthvalue(k, dim=1)
        tau = kth_dists.median().item()
        
        # Calculate precision
        dists_fake2real = torch.cdist(fake_features, real_features, p=2)
        min_dists_fake, _ = dists_fake2real.min(dim=1)
        precision = (min_dists_fake < tau).float().mean().item()
        
        # Calculate recall
        dists_real2fake = torch.cdist(real_features, fake_features, p=2)
        min_dists_real, _ = dists_real2fake.min(dim=1)
        recall = (min_dists_real < tau).float().mean().item()
        
        return precision, recall
    
    def prepare_conditioning(self, batch_size):
        """Prepare conditioning for generation. Override this method for specific conditioning."""
        # Default unconditional generation
        c = {}
        uc = {}
        
        # If the model has a conditioner, get conditioning
        if hasattr(self.model, 'conditioner'):
            # Create empty batch for unconditional generation
            empty_batch = {}
            
            # You may need to adjust this based on your specific conditioning requirements
            # For example, if you need text conditioning:
            # empty_batch['txt'] = [''] * batch_size
            
            c, uc = self.model.conditioner.get_unconditional_conditioning(
                empty_batch,
                force_uc_zero_embeddings=[]
            )
            
            # Ensure conditioning tensors are on the right device and have correct batch size
            for key in c:
                if isinstance(c[key], torch.Tensor):
                    if c[key].shape[0] != batch_size:
                        c[key] = c[key][:1].repeat(batch_size, *([1] * (c[key].ndim - 1)))
                    c[key] = c[key].to(self.device)
            
            for key in uc:
                if isinstance(uc[key], torch.Tensor):
                    if uc[key].shape[0] != batch_size:
                        uc[key] = uc[key][:1].repeat(batch_size, *([1] * (uc[key].ndim - 1)))
                    uc[key] = uc[key].to(self.device)
        
        return c, uc
    
    def compute_generated_statistics(self):
        """Generate samples and compute statistics."""
        fake_features_list = []
        n_rounds = (self.num_samples + self.batch_size - 1) // self.batch_size
        
        with torch.no_grad():
            for i in tqdm(range(n_rounds), desc="Generating samples and computing features"):
                current_batch_size = min(self.batch_size, self.num_samples - i * self.batch_size)
                if current_batch_size <= 0:
                    break
                
                # Prepare conditioning
                c, uc = self.prepare_conditioning(current_batch_size)
                
                # Determine the shape for generation
                if hasattr(self.model.model, 'diffusion_model'):
                    # Get shape from the diffusion model
                    in_channels = self.model.model.diffusion_model.in_channels
                    if hasattr(self.model.model.diffusion_model, 'image_size'):
                        latent_size = self.model.model.diffusion_model.image_size
                    else:
                        # Assume latent size is image_size // 8 (common for VAE)
                        latent_size = self.image_size // 8
                else:
                    # Default values
                    in_channels = 4
                    latent_size = self.image_size // 8
                
                shape = [in_channels, latent_size, latent_size]
                
                # Generate samples using the model's sampler
                if hasattr(self.model, 'sample') and self.model.sampler is not None:
                    samples = self.model.sample(
                        cond=c,
                        uc=uc,
                        batch_size=current_batch_size,
                        shape=shape
                    )
                else:
                    # Fallback: generate random noise (this should be replaced with proper sampling)
                    print("Warning: No sampler found, generating random samples")
                    samples = torch.randn(current_batch_size, *shape).to(self.device)
                
                # Decode samples to images
                x_samples = self.model.decode_first_stage(samples)
                
                # Save generated images if requested
                if self.save_generated_images:
                    for j in range(current_batch_size):
                        if i * self.batch_size + j < 100:  # Save first 100 images
                            sample = x_samples[j]
                            sample = torch.clamp((sample + 1) / 2, 0, 1)  # Normalize to [0, 1]
                            sample = sample.permute(1, 2, 0).cpu().numpy()
                            sample = (sample * 255).astype(np.uint8)
                            img = Image.fromarray(sample)
                            img.save(os.path.join(self.samples_dir, f'sample_{i*self.batch_size+j:05d}.png'))
                
                # Get inception features
                features = self.get_inception_features(x_samples)
                fake_features_list.append(features.cpu())
                
                # Clean up memory
                del samples, x_samples, features
                torch.cuda.empty_cache()
        
        fake_features = torch.cat(fake_features_list, dim=0)[:self.num_samples]
        return fake_features
    
    def save_results(self, results):
        """Save evaluation results to file."""
        results_dir = './eval_results'
        os.makedirs(results_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        output_dict = {
            'model_path': self.model_path,
            'config_path': self.config_path,
            'global_step': self.global_step,
            'num_samples': self.num_samples,
            'batch_size': self.batch_size,
            'num_inference_steps': self.num_inference_steps,
            'guidance_scale': self.guidance_scale,
            'inception_score': float(results['inception_score']),
            'fid_score': float(results['fid_score']),
            'precision': float(results['precision']),
            'recall': float(results['recall']),
            'timestamp': timestamp
        }
        
        filename = os.path.join(results_dir, f'eval_results_{self.global_step}_{timestamp}.json')
        with open(filename, 'w') as f:
            json.dump(output_dict, f, indent=4)
        
        print(f"\nResults saved to {filename}")
        if self.save_generated_images:
            print(f"Sample images saved to {self.samples_dir}")
    
    def evaluate(self):
        """Evaluate the model and compute metrics."""
        print(f"Starting evaluation for model at global step {self.global_step}")
        print(f"Computing metrics for {self.num_samples} samples...")
        
        # Generate samples and compute features
        fake_features = self.compute_generated_statistics()
        
        # Calculate Inception Score
        is_score = self.calculate_inception_score(fake_features)
        print(f"Inception Score: {is_score:.3f}")
        
        # Calculate FID Score
        fid = self.calculate_fid(self.real_features, fake_features)
        print(f"FID Score: {fid:.3f}")
        
        # Calculate Precision and Recall
        precision, recall = self.calculate_precision_recall(self.real_features, fake_features, k=3)
        print(f"Precision: {precision:.3f}")
        print(f"Recall: {recall:.3f}")
        
        results = {
            'inception_score': is_score,
            'fid_score': fid,
            'precision': precision,
            'recall': recall
        }
        
        self.save_results(results)
        return results

def main():
    parser = argparse.ArgumentParser(description='Evaluate Diffusion model')
    parser.add_argument('--config', type=str, required=True, help='Path to the model config file')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the model checkpoint')
    parser.add_argument('--dataset_path', type=str, required=True, help='Path to the real dataset for FID calculation')
    parser.add_argument('--batch_size', type=int, default=10, help='Batch size for evaluation')
    parser.add_argument('--num_samples', type=int, default=10000, help='Number of samples to generate')
    parser.add_argument('--num_inference_steps', type=int, default=50, help='Number of inference steps')
    parser.add_argument('--guidance_scale', type=float, default=7.5, help='Guidance scale for sampling')
    parser.add_argument('--save_images', action='store_true', help='Save generated images')
    parser.add_argument('--samples_dir', type=str, default='./generated_samples', help='Directory to save generated samples')
    args = parser.parse_args()
    
    evaluator = DiffusionEvaluator(
        config_path=args.config,
        model_path=args.model_path,
        dataset_path=args.dataset_path,
        batch_size=args.batch_size,
        num_samples=args.num_samples,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        save_generated_images=args.save_images,
        samples_dir=args.samples_dir
    )
    results = evaluator.evaluate()
    
if __name__ == '__main__':
    main()