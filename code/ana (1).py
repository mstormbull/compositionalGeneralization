import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split # For splitting decoder data

def plot_learning_curves(all_val_losses, title="Model Performance", save_path="learning_curves.png"):
    """
    Plots validation learning curves for multiple models.

    Args:
        all_val_losses (dict): A dictionary where keys are model names and values are lists of validation losses.
        title (str): The title of the plot.
        save_path (str): The full path (including filename) to save the plot.
    """
    plt.figure(figsize=(10, 6)) # Adjusted figure size for better legend visibility
    for model_name, val_losses in all_val_losses.items():
        if val_losses and not all(np.isnan(l) for l in val_losses if l is not None): # Check if list is not empty and not all NaNs
            plt.plot(val_losses, label=f"{model_name} Val Loss", lw=2.5) # Adjusted line width
        else:
            print(f"Skipping plotting for {model_name} in '{title}' due to empty or all-NaN loss data.")
    
    plt.xlabel("Epoch")
    plt.ylabel("Validation Loss (MSE)")
    plt.title(title)
    # Only show legend if there are valid lines plotted
    if any(val_losses and not all(np.isnan(l) for l in val_losses if l is not None) for val_losses in all_val_losses.values()):
        plt.legend(loc='best')
    plt.grid(True, linestyle='--', alpha=0.7) # Softer grid
    plt.tight_layout()
    
    # Ensure directory exists before saving
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close() # Close plot to free memory
    print(f"Learning curves saved to {save_path}")


class LinearDecoder(nn.Module):
    """
    A simple linear decoder.
    """
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.fc = nn.Linear(input_dim, output_dim)
        # Weight initialization
        nn.init.xavier_uniform_(self.fc.weight) 
        if self.fc.bias is not None:
            nn.init.zeros_(self.fc.bias)

    def forward(self, x):
        return self.fc(x)

def train_simple_decoder(hidden_states, coefficients, device,
                         epochs=100, lr=1e-3, batch_size=32, test_size=0.2, random_state=42):
    """
    Trains a simple linear decoder and evaluates it.

    Args:
        hidden_states (torch.Tensor): Processed hidden states (Samples x Features).
        coefficients (torch.Tensor): Target coefficients (Samples x NumCoefficients).
        device (torch.device): Device to use for training.
        epochs (int): Number of training epochs.
        lr (float): Learning rate.
        batch_size (int): Batch size for training.
        test_size (float): Proportion of data to use for testing.
        random_state (int): Random seed for train/test split.

    Returns:
        tuple: (final_mse, final_r2) on the test set. Returns (nan, nan) if training fails.
    """
    if hidden_states.shape[0] < 2 : # Need at least 2 samples for train/test split
        print("Warning (train_simple_decoder): Not enough samples to train/test decoder.")
        return float('nan'), float('nan')

    # Ensure coefficients are 2D (Samples x NumCoefficients) even if NumCoefficients is 1
    y_coeffs = coefficients
    if y_coeffs.ndim == 1:
        y_coeffs = y_coeffs.unsqueeze(1)

    X_train, X_test, y_train, y_test = train_test_split(
        hidden_states, y_coeffs, test_size=test_size, random_state=random_state
    )
    
    if X_train.shape[0] == 0 or X_test.shape[0] == 0:
        print("Warning (train_simple_decoder): Not enough samples in train or test split for decoder.")
        return float('nan'), float('nan')

    input_dim = X_train.shape[1]
    output_dim = y_train.shape[1] # Should be NumCoefficients
    
    decoder = LinearDecoder(input_dim, output_dim).to(device)
    optimizer = optim.Adam(decoder.parameters(), lr=lr)
    criterion = nn.MSELoss()

    train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
    # Drop last batch if it's smaller than batch_size and contains < 1 sample, to avoid issues with some layers if needed
    # For simple linear layer, it's usually fine.
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=(X_train.shape[0] % batch_size == 1 and X_train.shape[0]>1) )


    for epoch in range(epochs):
        decoder.train()
        for hs_batch, coeffs_batch in train_loader:
            hs_batch, coeffs_batch = hs_batch.to(device), coeffs_batch.to(device)
            optimizer.zero_grad()
            preds = decoder(hs_batch)
            loss = criterion(preds, coeffs_batch)
            loss.backward()
            optimizer.step()

    decoder.eval()
    final_mse = float('nan')
    final_r2 = float('nan')
    with torch.no_grad():
        X_test_dev, y_test_dev = X_test.to(device), y_test.to(device)
        if X_test_dev.shape[0] > 0: # Ensure test set is not empty
            test_preds = decoder(X_test_dev)
            final_mse = criterion(test_preds, y_test_dev).item()
            # Ensure y_test_dev and test_preds are suitable for r2_score (e.g., not all constant)
            try:
                final_r2 = r2_score(y_test_dev.cpu().numpy(), test_preds.cpu().numpy())
            except ValueError as e:
                print(f"Warning (train_simple_decoder): R2 score calculation failed. {e}")
                final_r2 = float('nan') # R2 can be ill-defined (e.g. constant y_true)
        else:
            print("Warning (train_simple_decoder): Test set empty after split, cannot evaluate decoder.")
            
    return final_mse, final_r2


def perform_decodability_analysis(model_name, hidden_states, coefficients,
                                  decoder_type='linear', decoding_metric='r2', 
                                  results_dir=None, device=None, random_state=42, **kwargs):
    """
    Performs decodability analysis of task coefficients from hidden states.

    Args:
        model_name (str): Name of the model being analyzed.
        hidden_states (torch.Tensor): Hidden states (Samples x Time x Features).
        coefficients (torch.Tensor): Task coefficients (Samples x NumCoefficients).
        decoder_type (str): Type of decoder ('linear' or 'ridge').
        decoding_metric (str): Metric to return ('r2' or 'mse').
        results_dir (str, optional): Directory to save plots.
        device (torch.device, optional): PyTorch device.
        random_state (int): Random seed for reproducibility of train/test splits.

    Returns:
        float: The specified decoding_metric score (R2 or MSE).
    """
    if hidden_states is None or coefficients is None:
        print(f"Skipping decodability for {model_name}: Missing hidden states or coefficients.")
        return float('nan')
    if hidden_states.shape[0] != coefficients.shape[0]:
        print(f"Error for {model_name}: Mismatch samples for hidden_states ({hidden_states.shape[0]}) and coeffs ({coefficients.shape[0]})")
        return float('nan')
    
    # Ensure coefficients are 2D
    if coefficients.ndim == 1:
        coefficients = coefficients.unsqueeze(1)

    if hidden_states.numel() == 0 or coefficients.numel() == 0 or hidden_states.shape[0] < 5: # Min samples for analysis
        print(f"Warning: Empty or too few hidden states/coefficients for {model_name} (samples: {hidden_states.shape[0]}).")
        return float('nan')

    print(f"  Analyzing decodability for {model_name}...")
    print(f"  Hidden states shape: {hidden_states.shape}")  
    print(f"  Coefficients shape: {coefficients.shape}")    

    # Process Hidden States: Use mean over time as default
    # Ensure hidden_states is at least 2D (Samples x Features) after this. If it was (S, T, F), it becomes (S, F).
    if hidden_states.ndim > 2: # If time dimension exists
        processed_hidden_states = torch.mean(hidden_states, dim=1)  
    else: # If already (Samples, Features)
        processed_hidden_states = hidden_states

    if processed_hidden_states.is_complex():
        processed_hidden_states = torch.cat(
            (processed_hidden_states.real, processed_hidden_states.imag), dim=-1
        ).float()
    else:
        processed_hidden_states = processed_hidden_states.float()

    print(f"  Processed hidden states shape for decoder: {processed_hidden_states.shape}")

    if processed_hidden_states.shape[1] == 0: # No features after processing
        print(f"  Warning: Processed hidden states have 0 features for {model_name}.")
        return float('nan')

    score = float('nan')
    mse_val = float('nan')
    r2_val = float('nan')
    
    # Ensure coefficients are on the CPU for scikit-learn and some PyTorch operations if needed
    coefficients_cpu = coefficients.cpu()
    processed_hidden_states_cpu = processed_hidden_states.cpu()


    if decoder_type == 'linear':
        if device is None: device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        mse_val, r2_val = train_simple_decoder(processed_hidden_states, coefficients, device, random_state=random_state)
        score = r2_val if decoding_metric == 'r2' else mse_val
        print(f"  PyTorch Linear Decoder for {model_name} - Test MSE: {mse_val:.4f}, Test R2: {r2_val:.4f}")

    elif decoder_type == 'ridge':
        X = processed_hidden_states_cpu.numpy()
        y = coefficients_cpu.numpy() # y is (Samples, NumCoefficients)
        
        if X.shape[0] < 5 : # Min samples for RidgeCV with some internal splitting
            print(f"  Skipping RidgeCV for {model_name}: Not enough samples ({X.shape[0]})")
            return float('nan')

        # Split data for RidgeCV to get a held-out test score
        # y_train and y_test will have shape (n_split_samples, NumCoefficients)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state)
        
        if X_train.shape[0] < 2 or X_test.shape[0] < 1:
            print(f"  Skipping RidgeCV for {model_name}: Not enough samples after split.")
            return float('nan')

        # If y_train is (N,1), RidgeCV handles it as single-target.
        # If y_train is (N,M) with M>1, RidgeCV can do multi-target.
        ridge_pipeline = Pipeline([
            ('scaler', StandardScaler()),  
            ('ridge', RidgeCV(alphas=np.logspace(-4, 2, 20), store_cv_results=True)) # Use store_cv_results for newer sklearn
        ])
        
        preds = None # Initialize preds
        try:
            ridge_pipeline.fit(X_train, y_train)
            preds = ridge_pipeline.predict(X_test) # preds shape: (n_test_samples,) if y_train was (N,1), else (n_test_samples, n_targets)
            
            mse_val = mean_squared_error(y_test, preds)
            r2_val = r2_score(y_test, preds)
            score = r2_val if decoding_metric == 'r2' else mse_val
            alpha_val = ridge_pipeline.named_steps['ridge'].alpha_
            print(f"  RidgeCV Decoder for {model_name} - Test MSE: {mse_val:.4f}, Test R2: {r2_val:.4f} (best alpha: {alpha_val:.4f})")
        except Exception as e:
            print(f"  Error during RidgeCV for {model_name}: {e}")
            score = float('nan')
            # Fallback values if prediction failed
            mse_val = float('nan')
            r2_val = float('nan')

    else:
        raise ValueError(f"Unsupported decoder_type: {decoder_type}")

    # Plotting section
    # Check if 'preds' was defined (i.e., prediction happened) and if y_test has samples
    if results_dir and coefficients.shape[1] > 0 and preds is not None and y_test.shape[0] > 0:
        os.makedirs(results_dir, exist_ok=True) # Ensure dir exists
        
        num_coeffs_to_plot = coefficients.shape[1]
        
        # Determine how many subplots - let's aim for a max of e.g. 3x3 grid if many coeffs
        # For simplicity, just plot the first one if many, or all if few.
        # Here, we'll just plot the first coefficient as per original logic (coeff_idx_to_plot = 0)
        coeff_idx_to_plot = 0 # Plotting the first coefficient (index 0)

        # y_test has shape (num_samples, num_coeffs)
        true_vals_to_plot = y_test[:, coeff_idx_to_plot] # This is 1D: (num_samples,)

        # preds has shape (num_samples,) if num_coeffs_in_y_train == 1
        # OR (num_samples, num_coeffs_in_y_train) if num_coeffs_in_y_train > 1
        if preds.ndim == 1:
            # This happens if num_coeffs == 1 (single target regression for RidgeCV)
            # or if LinearDecoder output_dim was 1 and it squeezed.
            pred_vals_to_plot = preds
        elif preds.ndim == 2 and preds.shape[1] > coeff_idx_to_plot :
            pred_vals_to_plot = preds[:, coeff_idx_to_plot]
        else:
            print(f"  Warning: Predictions array shape {preds.shape} not suitable for plotting coefficient {coeff_idx_to_plot}.")
            return score # Skip plotting if preds shape is unexpected


        plt.figure(figsize=(7, 7)) # Slightly larger for better readability
        plt.scatter(true_vals_to_plot, pred_vals_to_plot, alpha=0.6, s=50, edgecolor='k', linewidth=0.5, label=f"Predictions (R²={r2_val:.3f})")
        
        # Ensure true_vals_to_plot and pred_vals_to_plot are not empty before calling .min()/.max()
        if true_vals_to_plot.size > 0 and pred_vals_to_plot.size > 0:
            # Calculate min/max for the perfect fit line, handling potential NaNs if any
            valid_true = true_vals_to_plot[~np.isnan(true_vals_to_plot)]
            valid_preds = pred_vals_to_plot[~np.isnan(pred_vals_to_plot)]
            if valid_true.size > 0 and valid_preds.size > 0:
                min_val = min(valid_true.min(), valid_preds.min())
                max_val = max(valid_true.max(), valid_preds.max())
                plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label="Perfect Fit")
            else:
                print("Warning: Not enough valid (non-NaN) data points to determine plot range for decodability.")
        else:
            print("Warning: Not enough data points to determine plot range for decodability.")

        plt.xlabel(f"True Coefficient $a_{{{coeff_idx_to_plot+1}}}$", fontsize=12)
        plt.ylabel(f"Predicted Coefficient $\\hat{{a}}_{{{coeff_idx_to_plot+1}}}$", fontsize=12)
        plt.title(f"Decodability: {model_name} (Coeff {coeff_idx_to_plot+1})", fontsize=14)
        plt.legend(fontsize=10)
        plt.grid(True, linestyle=':', alpha=0.7)
        
        # Attempt to make axis equal, but catch errors if data range is too small or problematic
        try:
            plt.axis('equal')
        except ValueError as e:
            print(f"  Warning: Could not set axis to 'equal' for decodability plot. {e}")

        plt.tight_layout()
        plot_filename = f"decodability_{model_name}_coeff{coeff_idx_to_plot+1}.png" # Use 1-based indexing for filename
        save_path = os.path.join(results_dir, plot_filename)
        plt.savefig(save_path)
        plt.close()
        print(f"  Decodability plot saved to {save_path}")

    return score
