import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

class CompositionalDataset(Dataset):
    def __init__(self, num_samples, num_basis, seq_length, unseen=False,
                 basis_seed=42, coeff_seed=None, noise=0.01, time_range=(0, 2 * np.pi)):
        """
        Generates a dataset for the compositional generalization task.
        Input: I(t) = sum_k a_k * mu_k(t)
        Output: y*(t) = sum_k a_k * eta_k(t)

        Args:
            num_samples (int): Number of data samples.
            num_basis (int): Number of basis functions (N_k, number of coefficients a_k).
            seq_length (int): Length of each sequence (T).
            unseen (bool): If True, generates coefficients from a different distribution
                           (e.g., for zero-shot testing).
            basis_seed (int): Seed for basis generation (ensures consistency).
            coeff_seed (int): Seed for coefficient sampling (optional, for reproducibility).
            noise (float): Standard deviation of Gaussian noise added to inputs and outputs.
            time_range (tuple): (min_time, max_time) for np.linspace.
        """
        self.num_samples = num_samples
        self.num_basis = num_basis # This is N_k, the number of coefficients
        self.seq_length = seq_length
        self.noise_std = noise
        self.time_range = time_range

        # Fix seeds for reproducibility of basis functions
        # Basis functions are shared across train/test/val if seed is the same
        rng_basis = np.random.RandomState(basis_seed)
        self.input_basis = self._generate_basis(rng_basis)  # Shape: (num_basis, seq_length)
        self.output_basis = self._generate_basis(rng_basis) # Shape: (num_basis, seq_length)

        # Generate coefficients
        # Coefficients seed can be different for train vs test to ensure different samples
        rng_coeffs_state = np.random.RandomState(coeff_seed) if coeff_seed is not None else np.random.RandomState(np.random.randint(0,10000))
        self.coefficients = self._generate_coefficients(rng_coeffs_state, unseen) # Shape: (num_samples, num_basis)

        # Generate dataset
        # inputs = A @ B where A is (S, N_k) and B is (N_k, T) -> (S, T)
        self.inputs_clean = self.coefficients @ self.input_basis
        self.outputs_clean = self.coefficients @ self.output_basis

        # Add noise using the same rng_coeffs_state for consistency if needed, or a new one
        self.inputs = self.inputs_clean + self.noise_std * rng_coeffs_state.randn(*self.inputs_clean.shape)
        self.outputs = self.outputs_clean + self.noise_std * rng_coeffs_state.randn(*self.outputs_clean.shape)
        
        # Ensure data is float32 for PyTorch
        self.inputs = self.inputs.astype(np.float32)
        self.outputs = self.outputs.astype(np.float32)
        self.coefficients = self.coefficients.astype(np.float32)


    def _generate_basis(self, rng):
        """
        Generates a set of non-orthogonal basis functions (e.g., sine waves) and normalizes them.
        Returns:
            basis (np.array): Shape (num_basis, seq_length)
        """
        basis = np.zeros((self.num_basis, self.seq_length))
        t = np.linspace(self.time_range[0], self.time_range[1], self.seq_length)
        
        for i in range(self.num_basis):
            # Frequencies and phases chosen to make them somewhat distinct
            freq = rng.uniform(0.5 + i*0.2, 3.0 + i*0.2) # Try to make them more distinct
            phase = rng.uniform(0, np.pi)
            amplitude_variation = rng.uniform(0.8, 1.2)
            basis[i, :] = amplitude_variation * np.sin(freq * t + phase)
        
        # Normalize basis vectors (optional, but can be helpful)
        # norm = np.linalg.norm(basis, axis=1, keepdims=True)
        # basis = basis / (norm + 1e-6) # Add epsilon to avoid division by zero
        return basis

    def _generate_coefficients(self, rng, unseen):
        """
        Samples random coefficients for the linear combination of basis functions.
        Args:
            rng (np.random.RandomState): Random number generator.
            unseen (bool): If True, use a different distribution for coefficients.
        Returns:
            coeffs (np.array): Shape (num_samples, num_basis)
        """
        if unseen: # For test/validation set to check generalization
            return rng.uniform(-2.5, 2.5, (self.num_samples, self.num_basis)) # Slightly wider range for unseen
        else: # For training set
            return rng.randn(self.num_samples, self.num_basis) * 1.5 # Slightly larger variance for training

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        """
        Retrieves a single input-output pair and its corresponding coefficients.
        Input shape: (seq_length,)
        Output shape: (seq_length,)
        Coeffs shape: (num_basis,)
        """
        return (
            torch.tensor(self.inputs[idx], dtype=torch.float32),
            torch.tensor(self.outputs[idx], dtype=torch.float32),
            torch.tensor(self.coefficients[idx], dtype=torch.float32)
        )

def create_dataloaders(num_train_samples, num_val_samples, num_test_samples,
                       num_basis, seq_length, batch_size, noise=0.01, basis_seed=42,
                       num_workers=0):
    """
    Creates DataLoaders for train, validation, and test sets.
    """
    # Ensure distinct coefficient seeds for each dataset split
    train_coeff_seed = basis_seed + 1
    val_coeff_seed = basis_seed + 2
    test_coeff_seed = basis_seed + 3

    train_dataset = CompositionalDataset(
        num_samples=num_train_samples,
        num_basis=num_basis,
        seq_length=seq_length,
        unseen=False, 
        basis_seed=basis_seed, 
        coeff_seed=train_coeff_seed, 
        noise=noise
    )
    val_dataset = CompositionalDataset(
        num_samples=num_val_samples,
        num_basis=num_basis,
        seq_length=seq_length,
        unseen=True,  
        basis_seed=basis_seed, 
        coeff_seed=val_coeff_seed, 
        noise=noise
    )
    test_dataset = CompositionalDataset(
        num_samples=num_test_samples,
        num_basis=num_basis,
        seq_length=seq_length,
        unseen=True,  
        basis_seed=basis_seed, 
        coeff_seed=test_coeff_seed, 
        noise=noise
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, drop_last=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, drop_last=False)
    
    return train_loader, val_loader, test_loader, (train_dataset.input_basis, train_dataset.output_basis)

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    N_coeffs = 3
    T_seq = 100
    bs = 4

    train_loader, val_loader, test_loader, (input_basis, output_basis) = create_dataloaders(
        num_train_samples=128, num_val_samples=64, num_test_samples=64,
        num_basis=N_coeffs, seq_length=T_seq, batch_size=bs, noise=0.05
    )

    print(f"Number of task coefficients (a_k): {N_coeffs}")
    print(f"Sequence length: {T_seq}")

    print("\nInput basis shape:", input_basis.shape) 
    print("Output basis shape:", output_basis.shape)

    for x_batch, y_batch, coeffs_batch in train_loader:
        print("\nTrain batch shapes:")
        print("x_batch:", x_batch.shape)   
        print("y_batch:", y_batch.shape)   
        print("coeffs_batch:", coeffs_batch.shape) 
        break
    
    x_sample, y_sample, coeff_sample = test_loader.dataset[0]
    
    plt.figure(figsize=(12, 6))
    plt.subplot(2,1,1)
    plt.plot(x_sample.numpy(), label=f'Sample Input I(t) (coeffs: {np.round(coeff_sample.numpy(), 2)})')
    plt.title("Sample Input from Test Set")
    plt.legend()
    
    plt.subplot(2,1,2)
    plt.plot(y_sample.numpy(), label=f'Sample Output y*(t) (coeffs: {np.round(coeff_sample.numpy(), 2)})', color='orange')
    plt.title("Sample Output from Test Set")
    plt.legend()
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(12, min(15, 3 * N_coeffs))) # Adjusted figure size
    for i in range(N_coeffs):
        plt.subplot(N_coeffs, 2, 2*i + 1)
        plt.plot(input_basis[i, :])
        plt.title(f"Input Basis mu_{i+1}(t)")
        plt.subplot(N_coeffs, 2, 2*i + 2)
        plt.plot(output_basis[i, :])
        plt.title(f"Output Basis eta_{i+1}(t)")
    plt.tight_layout()
    plt.suptitle("Input and Output Basis Functions", y=1.02 if N_coeffs > 1 else 1.05)
    plt.show()