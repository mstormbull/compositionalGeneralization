import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import os
# from tqdm import tqdm # Optional progress bar, uncomment if you want to use it

def train_model_comparative(model, model_name, train_loader, val_loader, test_loader,
                            epochs, lr, device, num_task_coefficients, # num_task_coefficients is not directly used here but passed by sweep
                            results_dir, plot_intermediate_results=False):
    """
    Trains a given model and collects specified metrics.

    Args:
        model: The PyTorch model to train.
        model_name (str): Name of the model for saving and logging.
        train_loader (DataLoader): DataLoader for training data.
        val_loader (DataLoader): DataLoader for validation data.
        test_loader (DataLoader): DataLoader for test data (used for final hidden state extraction).
        epochs (int): Number of epochs to train.
        lr (float): Learning rate.
        device (torch.device): Device to train on (e.g., 'cuda' or 'cpu').
        num_task_coefficients (int): Number of task coefficients (passed from sweep, mainly for context if needed).
        results_dir (str): Directory to save intermediate plots.
        plot_intermediate_results (bool): If True, plots sample predictions during training.

    Returns:
        tuple:
            - train_losses_over_epochs (list): List of average training loss per epoch.
            - val_losses_over_epochs (list): List of average validation loss per epoch.
            - best_model_state_dict (dict): State dictionary of the model with the best validation loss.
            - all_hidden_states_test (torch.Tensor or None): Concatenated hidden states from the test set.
            - all_coeffs_test (torch.Tensor or None): Concatenated coefficients from the test set.
    """
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    train_losses_over_epochs = []
    val_losses_over_epochs = []
    
    best_val_loss = float('inf')
    best_model_state_dict = None # Store the state_dict of the best model
    
    # For collecting hidden states and coefficients from the test set using the best model
    all_hidden_states_test_list = []
    all_coeffs_test_list = []

    # Determine model's expected input dimension attribute (common variations)
    # This helps in reshaping the input batch if necessary.
    # It's a heuristic; ensure your models have one of these attributes if input reshaping is needed.
    model_input_dim_attr = None
    if hasattr(model, 'input_dim'):
        model_input_dim_attr = model.input_dim
    elif hasattr(model, 'input_size'):
        model_input_dim_attr = model.input_size
    elif hasattr(model, 'inputdim'):
         model_input_dim_attr = model.inputdim
    # Add other common names if your models use different ones

    for epoch in range(epochs):
        model.train()
        epoch_train_loss = 0.0
        # if not tqdm_disable: progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1} Train", leave=False)
        # else: progress_bar = train_loader
        
        for x_batch, y_batch, _ in train_loader: # Assuming loader yields (data, target, coefficients)
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)

            # --- Input Reshaping ---
            # This logic attempts to make the input shape compatible with common model expectations.
            # (B, T) -> (B, 1, T) if input_dim is 1 (e.g. for Conv1D or RNNs expecting channel dim)
            # (B, T, C) -> (B, C, T) if C matches model's input_dim (common for PyTorch layers)
            x_batch_model_input = x_batch
            if model_input_dim_attr is not None:
                if x_batch.ndim == 2: # Typically (Batch, SequenceLength)
                    # If model expects a channel dimension (e.g., input_dim=1), unsqueeze it.
                    # This is a common case for univariate time series.
                    if model_input_dim_attr == 1: # Heuristic: if model input_dim is 1, assume it's channel
                         x_batch_model_input = x_batch.unsqueeze(1) # (B, 1, T)
                    # else: # if input_dim > 1 and x_batch.ndim == 2, this might be an issue or handled by model
                         # x_batch_model_input = x_batch # or x_batch.unsqueeze(-1) if model expects (B, T, C)
                elif x_batch.ndim == 3: # Typically (Batch, SequenceLength, Features) or (Batch, Features, SequenceLength)
                    # If shape is (B, T, C) and model expects (B, C, T)
                    if x_batch.shape[2] == model_input_dim_attr and x_batch.shape[1] != model_input_dim_attr:
                        x_batch_model_input = x_batch.permute(0, 2, 1)
            # Else, assume x_batch is already in the correct format (e.g. (B, C, T) or (B, T, C) as model expects)


            optimizer.zero_grad()
            # Assuming model returns (output_sequence, hidden_states_sequence)
            y_pred_seq, _ = model(x_batch_model_input) 
            
            # --- Output Reshaping for Loss ---
            # Ensure y_pred_seq and y_batch have compatible shapes for the loss function.
            # Common case: y_pred_seq is (B, T, 1) and y_batch is (B, T).
            y_pred_seq_for_loss = y_pred_seq
            if y_pred_seq.ndim == 3 and y_pred_seq.shape[-1] == 1 and y_batch.ndim == 2:
                y_pred_seq_for_loss = y_pred_seq.squeeze(-1)
            elif y_pred_seq.ndim == 2 and y_batch.ndim == 2: # Both (B,T)
                y_pred_seq_for_loss = y_pred_seq
            elif y_pred_seq.shape == y_batch.shape: # If shapes already match (e.g. (B,T,O) and (B,T,O))
                y_pred_seq_for_loss = y_pred_seq
            else: # Attempt to match if y_batch is (B,T) and y_pred is (B,T,O)
                if y_batch.ndim == 2 and y_pred_seq.ndim == 3 and y_pred_seq.shape[0:2] == y_batch.shape[0:2]:
                    if y_pred_seq.shape[2] == 1: # (B,T,1) vs (B,T)
                        y_pred_seq_for_loss = y_pred_seq.squeeze(-1)
                    else: # (B,T,O) vs (B,T) - this is a problem if O > 1 and not handled by criterion
                        # print(f"Warning: y_pred_seq shape {y_pred_seq.shape} and y_batch shape {y_batch.shape} mismatch for loss in training.")
                        y_pred_seq_for_loss = y_pred_seq # Will likely error if criterion expects exact match
                # else:
                    # print(f"Warning: y_pred_seq shape {y_pred_seq.shape} and y_batch shape {y_batch.shape} mismatch for loss in training.")
                    # y_pred_seq_for_loss = y_pred_seq


            try:
                loss = criterion(y_pred_seq_for_loss, y_batch)
            except RuntimeError as e:
                print(f"RuntimeError during loss calculation for {model_name} (Training): {e}")
                print(f"  y_pred_for_loss shape: {y_pred_seq_for_loss.shape}, y_batch shape: {y_batch.shape}")
                raise e # Re-raise after printing info

            loss.backward()
            # Gradient clipping for models that define it (e.g., nmRNN variants)
            if hasattr(model, 'grad_clip') and model.grad_clip is not None: # Check for nmRNN's grad_clip
                torch.nn.utils.clip_grad_norm_(model.parameters(), model.grad_clip)
            elif hasattr(model, 'grad_clip_value') and model.grad_clip_value is not None: # Alternative name
                 torch.nn.utils.clip_grad_norm_(model.parameters(), model.grad_clip_value)
            optimizer.step()
            epoch_train_loss += loss.item()
        
        avg_train_loss = epoch_train_loss / len(train_loader) if len(train_loader) > 0 else float('nan')
        train_losses_over_epochs.append(avg_train_loss)

        # Validation phase
        model.eval()
        epoch_val_loss = 0.0
        with torch.no_grad():
            for x_batch_val, y_batch_val, _ in val_loader:
                x_batch_val, y_batch_val = x_batch_val.to(device), y_batch_val.to(device)

                # --- Input Reshaping for Validation (consistent with training) ---
                x_batch_val_model_input = x_batch_val
                if model_input_dim_attr is not None:
                    if x_batch_val.ndim == 2:
                        if model_input_dim_attr == 1:
                            x_batch_val_model_input = x_batch_val.unsqueeze(1)
                    elif x_batch_val.ndim == 3:
                        if x_batch_val.shape[2] == model_input_dim_attr and x_batch_val.shape[1] != model_input_dim_attr:
                            x_batch_val_model_input = x_batch_val.permute(0, 2, 1)
                
                y_pred_seq_val, _ = model(x_batch_val_model_input)
                
                # --- Output Reshaping for Loss (consistent with training) ---
                y_pred_seq_val_for_loss = y_pred_seq_val
                if y_pred_seq_val.ndim == 3 and y_pred_seq_val.shape[-1] == 1 and y_batch_val.ndim == 2 :
                    y_pred_seq_val_for_loss = y_pred_seq_val.squeeze(-1)
                elif y_pred_seq_val.ndim == 2 and y_batch_val.ndim == 2:
                    y_pred_seq_val_for_loss = y_pred_seq_val
                elif y_pred_seq_val.shape == y_batch_val.shape:
                     y_pred_seq_val_for_loss = y_pred_seq_val
                else:
                    if y_batch_val.ndim == 2 and y_pred_seq_val.ndim == 3 and y_pred_seq_val.shape[0:2] == y_batch_val.shape[0:2]:
                        if y_pred_seq_val.shape[2] == 1: y_pred_seq_val_for_loss = y_pred_seq_val.squeeze(-1)
                        # else: y_pred_seq_val_for_loss = y_pred_seq_val # Potential mismatch
                    # else: y_pred_seq_val_for_loss = y_pred_seq_val # Potential mismatch

                try:
                    val_loss_item = criterion(y_pred_seq_val_for_loss, y_batch_val)
                except RuntimeError as e:
                    print(f"RuntimeError during loss calculation for {model_name} (Validation): {e}")
                    print(f"  y_pred_val_for_loss shape: {y_pred_seq_val_for_loss.shape}, y_batch_val shape: {y_batch_val.shape}")
                    # In validation, we might not want to crash the whole sweep.
                    # Consider logging and continuing with a NaN loss for this batch/epoch.
                    # For now, we'll let it propagate if it's critical, but for sweeps, robust handling is better.
                    # To make it robust for sweeps, you could do:
                    # val_loss_item = torch.tensor(float('nan'), device=device)
                    raise e 
                epoch_val_loss += val_loss_item.item()
        
        avg_val_loss = epoch_val_loss / len(val_loader) if len(val_loader) > 0 else float('nan')
        val_losses_over_epochs.append(avg_val_loss)

        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

        if not np.isnan(avg_val_loss) and avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_state_dict = model.state_dict().copy() # Save a copy of the state_dict
            print(f"  New best validation loss: {best_val_loss:.4f}")

        # Intermediate plotting (optional)
        if plot_intermediate_results and (epoch % max(1, epochs // 10) == 0 or epoch == epochs - 1) :
            model.eval()
            with torch.no_grad():
                if len(val_loader.dataset) == 0:
                    print("Warning: Validation dataset is empty, cannot plot intermediate results.")
                    continue # Skip plotting if no data

                # Get a sample from validation set
                x_sample, y_sample, coeff_sample = val_loader.dataset[0] 
                x_sample_dev = x_sample.to(device).unsqueeze(0) # Add batch dim
                
                x_sample_model_input = x_sample_dev
                if model_input_dim_attr is not None:
                    if x_sample_dev.ndim == 2:
                        if model_input_dim_attr == 1:
                            x_sample_model_input = x_sample_dev.unsqueeze(1)
                    elif x_sample_dev.ndim == 3:
                         if x_sample_dev.shape[2] == model_input_dim_attr and x_sample_dev.shape[1] != model_input_dim_attr:
                            x_sample_model_input = x_sample_dev.permute(0,2,1)

                y_pred_sample_seq, hidden_sample_seq = model(x_sample_model_input)
                y_pred_sample = y_pred_sample_seq.squeeze().cpu().numpy() # Squeeze batch and channel (if 1)
                y_sample_np = y_sample.cpu().numpy()
                
                plt.figure(figsize=(15, 6))
                plt.subplot(1,2,1)
                plt.plot(y_sample_np, label='Target y(t)')
                plt.plot(y_pred_sample, label=f'Predicted y_hat(t) ({model_name})', linestyle='--')
                coeff_str = np.round(coeff_sample.numpy(), 2) if coeff_sample is not None else "N/A"
                plt.title(f'Epoch {epoch+1} - Val Sample (Coeffs: {coeff_str})')
                plt.xlabel("Time step")
                plt.ylabel("Value")
                plt.legend()
                plt.grid(True, linestyle=':', alpha=0.7)

                plt.subplot(1,2,2)
                if hidden_sample_seq is not None and hidden_sample_seq.numel() > 0:
                    hidden_to_plot = hidden_sample_seq.squeeze().cpu() # (T, H_dim) or (H_dim) if T=1
                    if hidden_to_plot.ndim == 1: hidden_to_plot = hidden_to_plot.unsqueeze(0) # Make it (1, H_dim) for consistent plotting
                    
                    num_dims_to_plot = min(5, hidden_to_plot.shape[-1]) # Plot up to 5 hidden dimensions
                    for i in range(num_dims_to_plot):
                        label_text = f'h_{i+1}'
                        if hasattr(model, 'N_oscillators') and hidden_to_plot.is_complex(): # Specific for ComplexOscillatorNet
                            plt.plot(torch.abs(hidden_to_plot[:, i]).numpy(), label=f'|z_{i+1}|')
                        else:
                            plt.plot(hidden_to_plot[:, i].numpy(), label=label_text)
                    plt.title(f'Hidden/Mode Dynamics ({model_name})')
                    plt.xlabel("Time step")
                    plt.ylabel("Activation")
                    plt.legend()
                    plt.grid(True, linestyle=':', alpha=0.7)
                else:
                    plt.text(0.5, 0.5, "No hidden states to plot or returned as None", ha='center', va='center')
                
                plt.tight_layout()
                plot_save_path = os.path.join(results_dir, f"{model_name}_epoch_{epoch+1}_val_sample.png")
                plt.savefig(plot_save_path)
                plt.close()
    
    # After all epochs, load the best model state for hidden state extraction
    if best_model_state_dict:
        model.load_state_dict(best_model_state_dict)
        print(f"\nLoaded best model (Val Loss: {best_val_loss:.4f}) for final hidden state extraction.")
    else:
        print("\nWarning: No best model state found (e.g., all val losses were NaN or training failed early). Using model from last epoch for hidden states.")

    # Extract hidden states from the test set using the (best) trained model
    model.eval()
    with torch.no_grad():
        if len(test_loader.dataset) == 0:
            print("Warning: Test dataset is empty. Cannot collect hidden states for decodability.")
        else:
            for x_batch_test, _, coeffs_batch_test in test_loader: # y_batch_test is not needed here
                x_batch_test = x_batch_test.to(device)
                coeffs_batch_test = coeffs_batch_test.to(device) # Ensure coeffs are on the right device too

                x_batch_test_model_input = x_batch_test
                if model_input_dim_attr is not None:
                    if x_batch_test.ndim == 2:
                        if model_input_dim_attr == 1:
                            x_batch_test_model_input = x_batch_test.unsqueeze(1)
                    elif x_batch_test.ndim == 3:
                        if x_batch_test.shape[2] == model_input_dim_attr and x_batch_test.shape[1] != model_input_dim_attr:
                             x_batch_test_model_input = x_batch_test.permute(0,2,1)
                
                _, hidden_states_batch_test = model(x_batch_test_model_input) 
                
                if hidden_states_batch_test is not None:
                    all_hidden_states_test_list.append(hidden_states_batch_test.cpu())
                    all_coeffs_test_list.append(coeffs_batch_test.cpu()) # Store corresponding coefficients

    all_hidden_states_test = None
    all_coeffs_test = None
    if all_hidden_states_test_list: # Check if list is not empty
        try:
            all_hidden_states_test = torch.cat(all_hidden_states_test_list, dim=0)
        except Exception as e:
            print(f"Error concatenating hidden states: {e}. Hidden states might have inconsistent shapes.")
            # Optionally, you could return the list of tensors or handle this more gracefully.
    if all_coeffs_test_list: # Check if list is not empty
        try:
            all_coeffs_test = torch.cat(all_coeffs_test_list, dim=0)
        except Exception as e:
            print(f"Error concatenating coefficients: {e}. Coefficients might have inconsistent shapes.")


    # Ensure the return order matches the sweep script's expectation:
    # (train_losses_epoch, val_losses_epoch, best_model_state, hidden_states_test, coeffs_test)
    return train_losses_over_epochs, val_losses_over_epochs, best_model_state_dict, all_hidden_states_test, all_coeffs_test
