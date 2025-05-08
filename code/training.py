import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import os
# from tqdm import tqdm # Optional progress bar

def train_model_comparative(model, model_name, train_loader, val_loader, test_loader,
                            epochs, lr, device, num_task_coefficients,
                            results_dir, plot_intermediate_results=False):
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    train_losses_over_epochs = []
    val_losses_over_epochs = []
    
    best_val_loss = float('inf')
    best_model_state_dict = None
    
    all_hidden_states_test_list = []
    all_coeffs_test_list = []

    for epoch in range(epochs):
        model.train()
        epoch_train_loss = 0.0
        for x_batch, y_batch, _ in train_loader: 
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)

            if x_batch.ndim == 2: 
                x_batch_model_input = x_batch.unsqueeze(1) 
            elif x_batch.ndim == 3 and x_batch.shape[1] != model.input_dim and x_batch.shape[2] == model.input_dim : # (B, T, C) -> (B, C, T)
                x_batch_model_input = x_batch.permute(0,2,1)
            else: 
                x_batch_model_input = x_batch

            optimizer.zero_grad()
            y_pred_seq, _ = model(x_batch_model_input) # Model returns (output_seq, hidden_states_seq)
            
            if y_pred_seq.ndim == 3 and y_pred_seq.shape[-1] == 1 and y_batch.ndim == 2 :
                y_pred_seq_for_loss = y_pred_seq.squeeze(-1)
            elif y_pred_seq.ndim == 2 and y_batch.ndim == 2: # Both (B,T)
                 y_pred_seq_for_loss = y_pred_seq
            elif y_pred_seq.shape == y_batch.shape: # If shapes already match (e.g. (B,T,O) and (B,T,O))
                 y_pred_seq_for_loss = y_pred_seq
            else: # Attempt to match if y_batch is (B,T) and y_pred is (B,T,O)
                if y_batch.ndim == 2 and y_pred_seq.ndim == 3 and y_pred_seq.shape[0:2] == y_batch.shape[0:2]:
                    if y_pred_seq.shape[2] == 1: # (B,T,1) vs (B,T)
                        y_pred_seq_for_loss = y_pred_seq.squeeze(-1)
                    else: # (B,T,O) vs (B,T) - this is a problem if O > 1
                        print(f"Warning: y_pred_seq shape {y_pred_seq.shape} and y_batch shape {y_batch.shape} mismatch for loss.")
                        y_pred_seq_for_loss = y_pred_seq # Will likely error
                else:
                    print(f"Warning: y_pred_seq shape {y_pred_seq.shape} and y_batch shape {y_batch.shape} mismatch for loss.")
                    y_pred_seq_for_loss = y_pred_seq


            try:
                loss = criterion(y_pred_seq_for_loss, y_batch)
            except RuntimeError as e:
                print(f"RuntimeError during loss calculation for {model_name}: {e}")
                print(f"  y_pred_for_loss shape: {y_pred_seq_for_loss.shape}, y_batch shape: {y_batch.shape}")
                raise e # Re-raise after printing info

            loss.backward()
            if hasattr(model, 'grad_clip_value') and model.grad_clip_value is not None: # For nmRNN
                 torch.nn.utils.clip_grad_norm_(model.parameters(), model.grad_clip_value)
            optimizer.step()
            epoch_train_loss += loss.item()
        
        avg_train_loss = epoch_train_loss / len(train_loader) if len(train_loader) > 0 else float('nan')
        train_losses_over_epochs.append(avg_train_loss)

        model.eval()
        epoch_val_loss = 0.0
        with torch.no_grad():
            for x_batch_val, y_batch_val, _ in val_loader:
                x_batch_val, y_batch_val = x_batch_val.to(device), y_batch_val.to(device)
                if x_batch_val.ndim == 2:
                    x_batch_val_model_input = x_batch_val.unsqueeze(1)
                elif x_batch_val.ndim == 3 and x_batch_val.shape[1] != model.input_dim and x_batch_val.shape[2] == model.input_dim :
                    x_batch_val_model_input = x_batch_val.permute(0,2,1)
                else:
                    x_batch_val_model_input = x_batch_val

                y_pred_seq_val, _ = model(x_batch_val_model_input)
                
                if y_pred_seq_val.ndim == 3 and y_pred_seq_val.shape[-1] == 1 and y_batch_val.ndim == 2 :
                    y_pred_seq_val_for_loss = y_pred_seq_val.squeeze(-1)
                elif y_pred_seq_val.ndim == 2 and y_batch_val.ndim == 2:
                     y_pred_seq_val_for_loss = y_pred_seq_val
                elif y_pred_seq_val.shape == y_batch_val.shape:
                     y_pred_seq_val_for_loss = y_pred_seq_val
                else:
                    if y_batch_val.ndim == 2 and y_pred_seq_val.ndim == 3 and y_pred_seq_val.shape[0:2] == y_batch_val.shape[0:2]:
                        if y_pred_seq_val.shape[2] == 1: y_pred_seq_val_for_loss = y_pred_seq_val.squeeze(-1)
                        else: y_pred_seq_val_for_loss = y_pred_seq_val 
                    else: y_pred_seq_val_for_loss = y_pred_seq_val


                val_loss_item = criterion(y_pred_seq_val_for_loss, y_batch_val)
                epoch_val_loss += val_loss_item.item()
        
        avg_val_loss = epoch_val_loss / len(val_loader) if len(val_loader) > 0 else float('nan')
        val_losses_over_epochs.append(avg_val_loss)

        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_state_dict = model.state_dict()
            print(f"  New best validation loss: {best_val_loss:.4f}")

        if plot_intermediate_results and (epoch % max(1, epochs // 5) == 0 or epoch == epochs -1) : # Plot more often
            model.eval()
            with torch.no_grad():
                # Ensure val_loader.dataset is not empty
                if len(val_loader.dataset) == 0:
                    print("Warning: Validation dataset is empty, cannot plot intermediate results.")
                    continue

                x_sample, y_sample, coeff_sample = val_loader.dataset[0]
                x_sample_dev = x_sample.to(device).unsqueeze(0) # Add batch dim
                
                if x_sample_dev.ndim == 2: 
                    x_sample_model_input = x_sample_dev.unsqueeze(1) 
                elif x_sample_dev.ndim == 3 and x_sample_dev.shape[1] != model.input_dim and x_sample_dev.shape[2] == model.input_dim :
                    x_sample_model_input = x_sample_dev.permute(0,2,1)
                else: 
                    x_sample_model_input = x_sample_dev

                y_pred_sample_seq, hidden_sample_seq = model(x_sample_model_input)
                y_pred_sample = y_pred_sample_seq.squeeze() # Remove batch and potentially channel dim if 1
                
                plt.figure(figsize=(15, 5))
                plt.subplot(1,2,1)
                plt.plot(y_sample.cpu().numpy(), label='Target y(t)')
                plt.plot(y_pred_sample.cpu().numpy(), label=f'Predicted y_hat(t) ({model_name})')
                plt.title(f'Epoch {epoch+1} - Val Sample (Coeffs: {np.round(coeff_sample.numpy(),1)})')
                plt.legend()

                plt.subplot(1,2,2)
                if hidden_sample_seq is not None and hidden_sample_seq.numel() > 0:
                    hidden_to_plot = hidden_sample_seq.squeeze() # (T, H_dim) or (H_dim) if T=1
                    if hidden_to_plot.ndim == 1: hidden_to_plot = hidden_to_plot.unsqueeze(0) # Make it (1, H_dim)
                    
                    num_dims_to_plot = min(5, hidden_to_plot.shape[-1])
                    for i in range(num_dims_to_plot):
                        label_text = f'h_{i+1}'
                        if hasattr(model, 'N_oscillators') and hidden_to_plot.is_complex():
                            plt.plot(torch.abs(hidden_to_plot[:, i]).cpu().numpy(), label=f'|z_{i+1}|')
                        else:
                            plt.plot(hidden_to_plot[:, i].cpu().numpy(), label=label_text)
                    plt.title(f'Hidden/Mode Dynamics ({model_name})')
                    plt.legend()
                else:
                    plt.text(0.5, 0.5, "No hidden states to plot", ha='center', va='center')
                
                plot_save_path = os.path.join(results_dir, f"{model_name}_epoch_{epoch+1}_val_sample.png")
                plt.savefig(plot_save_path)
                plt.close()
    
    if best_model_state_dict:
        model.load_state_dict(best_model_state_dict)
        print(f"\nLoaded best model (Val Loss: {best_val_loss:.4f}) for final hidden state extraction.")
    else:
        print("\nWarning: No best model state found. Using model from last epoch for hidden states.")

    model.eval()
    with torch.no_grad():
        if len(test_loader.dataset) == 0:
            print("Warning: Test dataset is empty. Cannot collect hidden states for decodability.")
        else:
            for x_batch_test, _, coeffs_batch_test in test_loader: 
                x_batch_test, coeffs_batch_test = x_batch_test.to(device), coeffs_batch_test.to(device)
                if x_batch_test.ndim == 2:
                    x_batch_test_model_input = x_batch_test.unsqueeze(1)
                elif x_batch_test.ndim == 3 and x_batch_test.shape[1] != model.input_dim and x_batch_test.shape[2] == model.input_dim :
                     x_batch_test_model_input = x_batch_test.permute(0,2,1)
                else:
                    x_batch_test_model_input = x_batch_test
                
                _, hidden_states_batch_test = model(x_batch_test_model_input) 
                
                if hidden_states_batch_test is not None:
                    all_hidden_states_test_list.append(hidden_states_batch_test.cpu())
                    all_coeffs_test_list.append(coeffs_batch_test.cpu())

    all_hidden_states_test = None
    all_coeffs_test = None
    if all_hidden_states_test_list:
        all_hidden_states_test = torch.cat(all_hidden_states_test_list, dim=0)
    if all_coeffs_test_list:
        all_coeffs_test = torch.cat(all_coeffs_test_list, dim=0)

    return val_losses_over_epochs, best_model_state_dict, all_hidden_states_test, all_coeffs_test