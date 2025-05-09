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
    plt.figure(figsize=(6, 4))
    for model_name, val_losses in all_val_losses.items():
        if val_losses and not all(np.isnan(val_losses)): # Check if list is not empty and not all NaNs
            plt.plot(val_losses, label=f"{model_name} Val Loss", lw = 3)
        else:
            print(f"Skipping plotting for {model_name} due to empty or all-NaN loss data.")
    
    plt.xlabel("Epoch")
    plt.ylabel("Validation Loss (MSE)")
    plt.title(title)
    if any(all_val_losses.values()): plt.legend() # Only show legend if there's something to plot
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Learning curves saved to {save_path}")


class LinearDecoder(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.fc = nn.Linear(input_dim, output_dim)
        nn.init.xavier_uniform_(self.fc.weight) # Initialize weights
        if self.fc.bias is not None: nn.init.zeros_(self.fc.bias)


    def forward(self, x):
        return self.fc(x)

def train_simple_decoder(hidden_states, coefficients, device,
                         epochs=100, lr=1e-3, batch_size=32, test_size=0.2):
    if hidden_states.shape[0] < 2 : # Need at least 2 samples for train/test split
        print("Warning: Not enough samples to train/test decoder.")
        return float('nan'), float('nan')

    X_train, X_test, y_train, y_test = train_test_split(
        hidden_states, coefficients, test_size=test_size, random_state=42
    )
    
    if X_train.shape[0] == 0 or X_test.shape[0] == 0:
        print("Warning: Not enough samples in train or test split for decoder.")
        return float('nan'), float('nan')

    input_dim = X_train.shape[1]
    output_dim = y_train.shape[1]
    
    decoder = LinearDecoder(input_dim, output_dim).to(device)
    optimizer = optim.Adam(decoder.parameters(), lr=lr)
    criterion = nn.MSELoss()

    train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

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
    with torch.no_grad():
        X_test_dev, y_test_dev = X_test.to(device), y_test.to(device)
        test_preds = decoder(X_test_dev)
        final_mse = criterion(test_preds, y_test_dev).item()
        final_r2 = r2_score(y_test_dev.cpu().numpy(), test_preds.cpu().numpy())
    
    return final_mse, final_r2


def perform_decodability_analysis(model_name, hidden_states, coefficients,
                                  decoder_type='linear', decoding_metric='r2', # Changed default to r2
                                  results_dir=None, device=None, **kwargs):
    if hidden_states is None or coefficients is None:
        print(f"Skipping decodability for {model_name}: Missing hidden states or coefficients.")
        return float('nan')
    if hidden_states.shape[0] != coefficients.shape[0]:
        print(f"Error for {model_name}: Mismatch samples for hidden_states ({hidden_states.shape[0]}) and coeffs ({coefficients.shape[0]})")
        return float('nan')
    if hidden_states.numel() == 0 or coefficients.numel() == 0 or hidden_states.shape[0] < 5: # Min samples for analysis
        print(f"Warning: Empty or too few hidden states/coefficients for {model_name} (samples: {hidden_states.shape[0]}).")
        return float('nan')

    print(f"  Analyzing decodability for {model_name}...")
    print(f"  Hidden states shape: {hidden_states.shape}") 
    print(f"  Coefficients shape: {coefficients.shape}")   

    # Process Hidden States: Use mean over time as default
    processed_hidden_states = torch.mean(hidden_states, dim=1) 
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

    if decoder_type == 'linear':
        mse_val, r2_val = train_simple_decoder(processed_hidden_states, coefficients, device)
        score = r2_val if decoding_metric == 'r2' else mse_val
        print(f"  PyTorch Linear Decoder for {model_name} - Test MSE: {mse_val:.4f}, Test R2: {r2_val:.4f}")

    elif decoder_type == 'ridge':
        X = processed_hidden_states.cpu().numpy()
        y = coefficients.cpu().numpy()
        
        if X.shape[0] < 5 : # Min samples for RidgeCV with some internal splitting
             print(f"  Skipping RidgeCV for {model_name}: Not enough samples ({X.shape[0]})")
             return float('nan')

        # Split data for RidgeCV to get a held-out test score
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        if X_train.shape[0] < 2 or X_test.shape[0] < 1:
            print(f"  Skipping RidgeCV for {model_name}: Not enough samples after split.")
            return float('nan')

        ridge_pipeline = Pipeline([
            ('scaler', StandardScaler()), 
            ('ridge', RidgeCV(alphas=np.logspace(-4, 2, 20), store_cv_values=False)) # store_cv_values=False if not needed
        ])
        
        try:
            ridge_pipeline.fit(X_train, y_train)
            preds = ridge_pipeline.predict(X_test)
            
            mse_val = mean_squared_error(y_test, preds)
            r2_val = r2_score(y_test, preds)
            score = r2_val if decoding_metric == 'r2' else mse_val
            print(f"  RidgeCV Decoder for {model_name} - Test MSE: {mse_val:.4f}, Test R2: {r2_val:.4f} (best alpha: {ridge_pipeline.named_steps['ridge'].alpha_:.4f})")
        except Exception as e:
            print(f"  Error during RidgeCV for {model_name}: {e}")
            score = float('nan')
    else:
        raise ValueError(f"Unsupported decoder_type: {decoder_type}")

    if results_dir and coefficients.shape[1] > 0 and 'preds' in locals() and y_test.shape[0] > 0:
        os.makedirs(results_dir, exist_ok=True) # Ensure dir exists
        coeff_idx_to_plot = 0
        plt.figure(figsize=(6,6))
        plt.scatter(y_test[:, coeff_idx_to_plot], preds[:, coeff_idx_to_plot], alpha=0.5, label=f"Predictions (R2={r2_val:.2f})")
        min_val = min(y_test[:, coeff_idx_to_plot].min(), preds[:, coeff_idx_to_plot].min())
        max_val = max(y_test[:, coeff_idx_to_plot].max(), preds[:, coeff_idx_to_plot].max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', label="Perfect Fit")
        plt.xlabel(f"True Coefficient a_{coeff_idx_to_plot+1}")
        plt.ylabel(f"Predicted Coefficient a_{coeff_idx_to_plot+1}")
        plt.title(f"Decodability: {model_name} (Coeff {coeff_idx_to_plot+1})")
        plt.legend()
        plt.grid(True)
        plt.axis('equal')
        plt.tight_layout()
        save_path = os.path.join(results_dir, f"decodability_{model_name}_coeff{coeff_idx_to_plot}.png")
        plt.savefig(save_path)
        plt.close()
        # print(f"  Decodability plot saved to {save_path}")

    return score