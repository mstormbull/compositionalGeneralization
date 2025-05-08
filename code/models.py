# models/oscillator_net.py
import torch
import torch.nn as nn
import numpy as np
import math
import scipy.spatial as ss # For SpatialWeight

class NonlinearOscillatorNet(nn.Module):
    def __init__(self, N_oscillators, device, outputdim=1, inputdim=1, 
                 seq_length=200, dt=0.1, seed=0):
        super().__init__()
        torch.manual_seed(seed)
        
        self.N_oscillators = N_oscillators 
        self.outputdim = outputdim
        self.inputdim = inputdim
        self.device = device
        self.dt = dt
        self.seq_length = seq_length 

        self.mu = nn.Parameter(torch.randn(1, N_oscillators) * 0.1) 
        self.omega = nn.Parameter(torch.randn(1, N_oscillators) * 0.5 + 3.0)  
        self.K = nn.Parameter(torch.randn(N_oscillators, N_oscillators, dtype=torch.cfloat) * 0.01)
        self.Q = nn.Parameter(torch.randn(N_oscillators, inputdim, dtype=torch.cfloat) * 0.1)
        self.r_param = nn.Parameter(torch.randn(1, N_oscillators, dtype=torch.cfloat) * 0.01)
        self.z_init = nn.Parameter(torch.randn(1, N_oscillators, dtype=torch.cfloat) * 0.1)
        self.D = nn.Parameter(torch.randn(outputdim, N_oscillators, dtype=torch.cfloat) * 0.1)

    def forward(self, I_t_seq):
        """
        I_t_seq: (batch_size, input_dim, seq_length).
        Returns:
            o_t: (batch_size, seq_length, output_dim) or (batch_size, seq_length) if outputdim=1.
            z_trajectory: (batch, seq_length, N_oscillators).
        """
        batch_size = I_t_seq.shape[0]
        current_seq_length = I_t_seq.shape[2]

        z = self.z_init.repeat(batch_size, 1).to(self.device)
        r = self.r_param.repeat(batch_size, 1).to(self.device)
        z_trajectory_list = []

        for t_step in range(current_seq_length):
            I_t_current_step = I_t_seq[:, :, t_step] 
            
            parametric_coupling = torch.einsum('jk,bj->bk', self.K, z) 
            input_injection = torch.einsum('ni,bi->bn', self.Q, I_t_current_step.cfloat())
            
            # Following PDF code structure for dynamics:
            # (mu + iw + Kz - |z-r|^2)(z-r) + Q*I
            term_in_paren = (self.mu + 1j * self.omega + parametric_coupling - torch.abs(z - r)**2)
            dz_dt = term_in_paren * (z - r) + input_injection
            
            z = z + self.dt * dz_dt
            z_trajectory_list.append(z.clone())

        z_trajectory = torch.stack(z_trajectory_list, dim=1)
        o_t_complex = torch.einsum('btn,no->bto', z_trajectory, self.D.T)
        o_t = torch.abs(o_t_complex)**2
        
        if self.outputdim == 1:
            o_t = o_t.squeeze(-1) 

        return o_t, z_trajectory

class RNNModel(nn.Module):
    def __init__(self, hidden_size, device, outputdim=1, inputdim=1, 
                 num_layers=1, rnn_type='GRU', seed=0):
        super().__init__()
        torch.manual_seed(seed)
        
        self.hidden_size = hidden_size
        self.outputdim = outputdim
        self.inputdim = inputdim
        self.num_layers = num_layers
        self.rnn_type = rnn_type.upper()
        self.device = device

        if self.rnn_type == 'GRU':
            self.rnn = nn.GRU(input_size=inputdim, hidden_size=hidden_size,
                              num_layers=num_layers, batch_first=True, dtype=torch.float)
        elif self.rnn_type == 'LSTM':
            self.rnn = nn.LSTM(input_size=inputdim, hidden_size=hidden_size,
                               num_layers=num_layers, batch_first=True, dtype=torch.float)
        else:
            raise ValueError("Unsupported RNN type. Choose 'GRU' or 'LSTM'.")

        self.h0 = nn.Parameter(torch.randn(num_layers, 1, hidden_size, dtype=torch.float) * 0.1)
        if self.rnn_type == 'LSTM':
            self.c0 = nn.Parameter(torch.randn(num_layers, 1, hidden_size, dtype=torch.float) * 0.1)
            
        self.decoder = nn.Linear(hidden_size, outputdim, dtype=torch.float)
        self.decoder.weight.data = torch.randn(outputdim, hidden_size, dtype=torch.float) * 0.1
        if self.decoder.bias is not None:
            self.decoder.bias.data.fill_(0.0)

    def forward(self, x_seq):
        """
        x_seq: (batch_size, input_dim, seq_length).
        Returns:
            output_seq: (batch_size, seq_length, output_dim) or (batch_size, seq_length).
            hidden_states_seq: (batch, seq_length, hidden_size).
        """
        batch_size = x_seq.shape[0]
        
        if x_seq.shape[1] == self.inputdim: 
             x_rnn_input = x_seq.permute(0, 2, 1) 
        elif x_seq.shape[2] == self.inputdim: 
             x_rnn_input = x_seq
        else:
            raise ValueError(f"Unexpected input shape {x_seq.shape} for RNN.")

        h_init = self.h0.repeat(1, batch_size, 1).to(self.device)
        if self.rnn_type == 'LSTM':
            c_init = self.c0.repeat(1, batch_size, 1).to(self.device)
            initial_state = (h_init, c_init)
        else: 
            initial_state = h_init
            
        rnn_output, _ = self.rnn(x_rnn_input, initial_state)
        hidden_states_seq = rnn_output 
        output_seq = self.decoder(rnn_output) 

        if self.outputdim == 1:
            output_seq = output_seq.squeeze(-1) 
            
        return output_seq, hidden_states_seq

class PositionalEncoding(nn.Module):
    # Standard Positional Encoding
    def __init__(self, d_model, max_len=5000, dropout=0.1): # Increased max_len
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1) 
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)) 
        
        # Changed to handle d_model not being even
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 != 0:
            # For odd d_model, the last column of div_term for cos might be missing
            # We can use the same div_term or a slightly adjusted one.
            # Here, we ensure cosine part is only up to d_model//2 * 2
            pe[:, 1::2] = torch.cos(position * div_term[:,:d_model//2]) # Ensure correct shape for cos
        else:
            pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer('pe', pe.unsqueeze(0)) # Shape (1, max_len, d_model) for batch_first

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, embedding_dim]
        """
        # x is (B, S, E), self.pe is (1, max_len, E)
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

class TransformerModel(nn.Module):
    def __init__(self, d_model, device, outputdim=1, inputdim=1, 
                 num_heads=4, num_layers=2, seq_length=200, dropout=0.1, seed=0):
        super().__init__()
        torch.manual_seed(seed)

        self.d_model = d_model
        self.outputdim = outputdim
        self.inputdim = inputdim
        self.device = device
        self.seq_length = seq_length # Used for PositionalEncoding max_len

        self.input_projection = nn.Linear(inputdim, d_model, dtype=torch.float)
        self.pos_encoder = PositionalEncoding(d_model, max_len=seq_length, dropout=dropout) # seq_length as max_len
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=num_heads, 
            dropout=dropout, 
            batch_first=True, 
            dtype=torch.float
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.decoder = nn.Linear(d_model, outputdim, dtype=torch.float)
        self._init_weights()

    def _init_weights(self):
        for lin_layer in [self.input_projection, self.decoder]:
            nn.init.xavier_uniform_(lin_layer.weight)
            if lin_layer.bias is not None:
                nn.init.zeros_(lin_layer.bias)

    def forward(self, x_seq):
        """
        x_seq: (batch_size, input_dim, seq_length).
        Returns:
            output_seq: (batch_size, seq_length, output_dim) or (batch_size, seq_length).
            encoded_seq: (batch, seq_length, d_model).
        """
        if x_seq.shape[1] == self.inputdim: 
             x_permuted = x_seq.permute(0, 2, 1) 
        elif x_seq.shape[2] == self.inputdim: 
             x_permuted = x_seq
        else:
            raise ValueError(f"Unexpected input shape {x_seq.shape} for Transformer.")

        embedded_input = self.input_projection(x_permuted) * math.sqrt(self.d_model)
        pos_encoded_input = self.pos_encoder(embedded_input) # PE expects (B,S,E)
        
        encoded_seq = self.transformer_encoder(pos_encoded_input) 
        output_seq = self.decoder(encoded_seq) 
        
        if self.outputdim == 1:
            output_seq = output_seq.squeeze(-1) 
            
        return output_seq, encoded_seq

# --- HIPPO Matrix Generation ---
def get_legs_matrices(N, device='cpu'): # Added device parameter
    """
    Generates the A and B matrices for the HIPPO-LegS method.
    Args:
        N (int): The approximation order (dimension of the state).
        device (str or torch.device): Device to create tensors on.
    Returns:
        A (torch.Tensor): The N x N A matrix for LegS.
        B (torch.Tensor): The N x 1 B matrix for LegS.
    """
    n = torch.arange(N, dtype=torch.float32, device=device)
    k = torch.arange(N, dtype=torch.float32, device=device)
    N_grid, K_grid = torch.meshgrid(n, k, indexing='ij')

    A = torch.zeros((N, N), dtype=torch.float32, device=device)
    mask_n_gt_k = N_grid > K_grid
    A[mask_n_gt_k] = (torch.sqrt(2 * N_grid + 1) * torch.sqrt(2 * K_grid + 1))[mask_n_gt_k]
    mask_n_eq_k = N_grid == K_grid
    A[mask_n_eq_k] = (N_grid + 1)[mask_n_eq_k] # Paper states (n+1) but usually it's - (n+1) for stability
                                            # Or A is defined for dx/dt = Ax, not -Ax.
                                            # The discretization uses -A, so if A is positive definite, -A is stable.
                                            # Let's assume the provided formula is correct for their system.
    B = torch.sqrt(2 * n + 1)
    return A, B.unsqueeze(1)

def get_legt_matrices(N, theta, device='cpu'): # Added device parameter
    """
    Generates the A and B matrices for the HIPPO-LegT method (LMU variant).
    Args:
        N (int): The approximation order (dimension of the state).
        theta (float): The fixed window length.
        device (str or torch.device): Device to create tensors on.
    Returns:
        A (torch.Tensor): The N x N A matrix for LegT.
        B (torch.Tensor): The N x 1 B matrix for LegT.
    """
    if theta <= 0:
        raise ValueError("theta must be positive")

    n = torch.arange(N, dtype=torch.float32, device=device)
    k = torch.arange(N, dtype=torch.float32, device=device)
    N_grid, K_grid = torch.meshgrid(n, k, indexing='ij')

    A = torch.zeros((N, N), dtype=torch.float32, device=device)
    factor_2n_plus_1 = (2 * N_grid + 1)

    mask_n_ge_k = N_grid >= K_grid
    A[mask_n_ge_k] = (factor_2n_plus_1 * ((-1.0)**(N_grid - K_grid)))[mask_n_ge_k]
    mask_n_lt_k = N_grid < K_grid
    A[mask_n_lt_k] = factor_2n_plus_1[mask_n_lt_k] # Paper says (-1)^(k-n-1) * (2n+1) for n < k.
                                                 # The provided code is (2n+1). Sticking to provided code.
    A = A / theta

    B = (2 * n + 1) * ((-1.0)**n)
    B = B / theta
    return A, B.unsqueeze(1)

# --- Discretization ---
def discretize_bilinear(A, B, dt):
    N = A.shape[0]
    A_64 = A.double()
    B_64 = B.double()
    I_64 = torch.eye(N, dtype=torch.float64, device=A.device)
    M = I_64 + dt / 2.0 * A_64 # Original HiPPO paper defines dx/dt = -Ax + Bu.
                              # If A here is the negative of that A, then M = I - dt/2 * A_from_paper
                              # Let's assume A here is the A from dx/dt = Ax.
                              # Bilinear for dx/dt = Ax + Bu is (I - dt/2 A)^-1 (I + dt/2 A) and (I - dt/2 A)^-1 dt B
                              # If the system is dx/dt = -Ax + Bu (common in HiPPO papers)
                              # Then Bar_A = (I + dt/2 A)^-1 (I - dt/2 A)
                              # And  Bar_B = (I + dt/2 A)^-1 dt B
                              # The provided code matches this for dx/dt = -Ax + Bu.
    try:
        M_inv = torch.linalg.inv(M)
    except torch.linalg.LinAlgError as e:
        print(f"Warning: Matrix inversion failed during discretization: {e}. Using pseudo-inverse.")
        M_inv = torch.linalg.pinv(M)
    
    Bar_A = M_inv @ (I_64 - dt / 2.0 * A_64)
    Bar_B = (dt * M_inv) @ B_64
    return Bar_A.float(), Bar_B.float()

# --- HIPPO RNN Layer ---
class HippoLayer(nn.Module):
    def __init__(self, hidden_size, method='legs', theta=None, dt=1.0, inv_eps=1e-10, clip_val=100.0, device='cpu', input_dim_gru=1):
        super().__init__()
        self.N = hidden_size
        self.method = method
        self.dt = dt 
        self.inv_eps = inv_eps 
        self.clip_val = clip_val
        self.device = device # Store device

        if method == 'legs':
            A_cont, B_cont = get_legs_matrices(self.N, device=self.device)
            self.register_buffer('A_cont_legs', A_cont)
            self.register_buffer('B_cont_legs', B_cont)
        elif method == 'legt':
            if theta is None:
                raise ValueError("theta must be provided for method 'legt'")
            self.theta = theta
            A_cont_legt, B_cont_legt = get_legt_matrices(self.N, self.theta, device=self.device)
            Bar_A_legt, Bar_B_legt = discretize_bilinear(A_cont_legt, B_cont_legt, self.dt)
            self.register_buffer('Bar_A_legt', Bar_A_legt)
            self.register_buffer('Bar_B_legt', Bar_B_legt)
        else:
            raise ValueError("Invalid method. Choose 'legs' or 'legt'.")

        self.rnn_cell = nn.GRUCell(input_size=input_dim_gru + self.N, hidden_size=self.N) # input_dim_gru for x_t
        self.fc_f = nn.Linear(self.N, 1) # Computes 1D feature f_t from h_t

    def forward(self, x_t, h_prev, c_prev, k):
        """
        x_t (torch.Tensor): External input (batch, input_dim_gru).
        h_prev (torch.Tensor): Previous RNN hidden state (batch, N).
        c_prev (torch.Tensor): Previous HIPPO state (batch, N).
        k (int): Current time step (1-based).
        """
        gru_input = torch.cat([x_t, c_prev], dim=-1)
        h_t = self.rnn_cell(gru_input, h_prev)
        f_t = self.fc_f(h_t) # (batch, 1)

        if self.method == 'legs':
            if k <= 0: raise ValueError("k must be a positive integer (1-based index)")
            A_cont = self.A_cont_legs.double()
            B_cont = self.B_cont_legs.double()
            c_prev_64 = c_prev.double()
            f_t_64 = f_t.double()
            I = torch.eye(self.N, dtype=torch.float64, device=self.device)

            M_k = I - A_cont / (2.0 * k) + self.inv_eps * I 
            try:
                M_k_inv = torch.linalg.inv(M_k)
            except torch.linalg.LinAlgError:
                M_k_inv = torch.linalg.pinv(M_k)

            if k == 1: Bar_A_k = M_k_inv
            else: Bar_A_k = M_k_inv @ (I + A_cont / (2.0 * (k - 1)))
            
            if k == 1: Bar_B_k = M_k_inv @ B_cont
            else: Bar_B_k = (1.0 / (k - 1)) * M_k_inv @ B_cont
            
            c_t = Bar_A_k @ c_prev_64.unsqueeze(-1) + Bar_B_k @ f_t_64.unsqueeze(-1)
            c_t = c_t.squeeze(-1).float()
        elif self.method == 'legt':
            Bar_A = self.Bar_A_legt.to(dtype=c_prev.dtype)
            Bar_B = self.Bar_B_legt.to(dtype=f_t.dtype)
            c_t = Bar_A @ c_prev.unsqueeze(-1) + Bar_B @ f_t.unsqueeze(-1)
            c_t = c_t.squeeze(-1)

        if self.clip_val is not None:
            c_t = torch.clamp(c_t, -self.clip_val, self.clip_val)
        
        # NaN/Inf checks (simplified)
        if torch.isnan(c_t).any() or torch.isinf(c_t).any():
            # print(f"NaN/Inf in c_t at k={k}. c_prev norm: {torch.linalg.norm(c_prev):.2e}, f_t norm: {torch.linalg.norm(f_t):.2e}")
            c_t = torch.nan_to_num(c_t, nan=0.0, posinf=self.clip_val, neginf=-self.clip_val)
        if torch.isnan(h_t).any() or torch.isinf(h_t).any():
            # print(f"NaN/Inf in h_t at k={k}")
            h_t = torch.nan_to_num(h_t, nan=0.0, posinf=self.clip_val, neginf=-self.clip_val) # Clip h_t too
            
        return h_t, c_t

# --- Full HIPPO RNN Model ---
class HippoRNNModel(nn.Module):
    def __init__(self, hidden_size, outputdim=1, inputdim=1, method='legs', 
                 theta=None, dt=1.0, inv_eps=1e-10, clip_val=100.0, device='cpu', seed=0):
        super().__init__()
        torch.manual_seed(seed)
        self.N = hidden_size # This is the main hidden size for HIPPO state and GRU
        self.inputdim = inputdim
        self.outputdim = outputdim
        self.device = device

        # Input projection if inputdim != hidden_size for GRU's x_t part, or if we want a learned projection
        # The HippoLayer GRUCell input_size is input_dim_gru + self.N.
        # x_t to HippoLayer is (batch, input_dim_gru).
        # If inputdim (from data) is 1, and we want to feed this 1 directly as input_dim_gru:
        self.input_projection = nn.Linear(inputdim, 1) # Project data input_dim to 1 (for HippoLayer's x_t)
                                                       # Or, if inputdim is already 1, this can be Identity or removed.
                                                       # For now, assume we always project to 1 for the GRU input part.

        self.hippo_layer = HippoLayer(hidden_size=self.N, method=method, theta=theta, 
                                      dt=dt, inv_eps=inv_eps, clip_val=clip_val, device=device,
                                      input_dim_gru=1) # x_t to GRU cell will be 1D

        self.output_decoder = nn.Linear(self.N, outputdim) # From GRU hidden state h_t to output y_t

    def forward(self, x_seq_orig):
        """
        x_seq_orig (torch.Tensor): Input (batch, input_dim, seq_length).
        Returns:
            o_seq_permuted (torch.Tensor): Output (batch, seq_length, outputdim) or (batch, seq_length).
            h_seq (torch.Tensor): GRU hidden states (batch, seq_length, N).
        """
        # Permute input to (batch, seq_length, input_dim)
        x_seq = x_seq_orig.permute(0, 2, 1) 
        batch_size, seq_length, _ = x_seq.shape

        h_prev = torch.zeros(batch_size, self.N, device=self.device)
        c_prev = torch.zeros(batch_size, self.N, device=self.device)
        
        h_states_list = []
        # c_states_list = [] # Optionally collect c_states

        for t in range(seq_length):
            x_t_input_dim = x_seq[:, t, :] # (batch, inputdim)
            x_t_projected = self.input_projection(x_t_input_dim) # (batch, 1) for GRU input part

            h_t, c_t = self.hippo_layer(x_t_projected, h_prev, c_prev, k=t + 1)
            
            h_states_list.append(h_t)
            # c_states_list.append(c_t)
            
            h_prev, c_prev = h_t, c_t

        h_seq = torch.stack(h_states_list, dim=1) # (batch, seq_length, N)
        # c_seq = torch.stack(c_states_list, dim=1) # (batch, seq_length, N)

        o_seq = self.output_decoder(h_seq) # (batch, seq_length, outputdim)
        
        if self.outputdim == 1:
            o_seq = o_seq.squeeze(-1) # (batch, seq_length)
            
        return o_seq, h_seq # Return GRU hidden states as the primary "hidden_states"

# --- Spatial Weight Module (from user) ---
class SpatialWeight(nn.Module):
    def __init__(self, hidden_size, N_nm=4, ell=0.1, scale=1.0, pinhib=0.5, seed=1): # Renamed observable_size to hidden_size
        super(SpatialWeight, self).__init__()
        np.random.seed(seed) 
        self.pos = nn.Parameter(torch.tensor(np.random.random([hidden_size,2]), dtype=torch.float32), requires_grad=False)
        
        # Pairwise distances: (hidden_size, hidden_size)
        delpoints_2d = ss.distance.cdist(self.pos.cpu().numpy(), self.pos.cpu().numpy()) # cdist needs numpy cpu
        
        # Expand for N_nm: (hidden_size, hidden_size, N_nm)
        self.delpoints = nn.Parameter(torch.tensor(delpoints_2d[:,:, None] * np.ones([1, 1, N_nm]), dtype=torch.float32), requires_grad=False)

        self.ell = ell 
        self.scale = scale 
        
        inhib_np = np.random.choice([0,1], hidden_size, p=[1-pinhib, pinhib])[:,None,None] * np.ones(self.delpoints.shape)
        self.inhib = nn.Parameter(torch.tensor(inhib_np, dtype=torch.float32), requires_grad=False)
        
        self.Delta = nn.Parameter(self.delpoints / self.ell, requires_grad=False)
        
        mask_np = np.logical_and(
            delpoints_2d[:,:,None] < 5 * self.ell, 
            np.eye(hidden_size)[:,:,None] * np.ones(self.delpoints.shape) == 0
        )
        self.mask = nn.Parameter(torch.tensor(mask_np, dtype=torch.float32), requires_grad=False)

    def forward(self, W_base):
        """ W_base is (hidden_size, hidden_size, N_nm) """
        # Ensure W_base has the correct shape for broadcasting with spatial properties
        if W_base.shape != self.Delta.shape:
            # This might happen if W_base is just initialized and needs to be expanded
            # Or if it's a single (H,H) matrix to be modulated by N_nm versions of spatial weights
            # Assuming W_base is the learnable part that gets modulated by fixed spatial structure.
            # The original code initializes self.weight_hh = nn.Parameter(self.spatialNet(torch.Tensor(H,H,N_NM)))
            # This implies W_base in forward is the torch.Tensor(H,H,N_NM) part.
            pass # Let einsum handle broadcasting if W_base is (H,H) and self.Delta is (H,H,N_nm)

        # Ensure all parts are on the same device as W_base
        inhib = self.inhib.to(W_base.device)
        Delta = self.Delta.to(W_base.device)
        mask = self.mask.to(W_base.device)
        
        return self.scale * ((-1)**inhib) * torch.exp(W_base - Delta) * mask

# --- Base nmRNN Cell (modified from user) ---
class nmRNNCellBase(nn.Module):
    def __init__(self, input_size, hidden_size, N_nm, activation_fn, bias, keepW0, use_spatial_weights,
                 spatial_ell, spatial_scale, seed=0):
        super().__init__()
        torch.manual_seed(seed)
        np.random.seed(seed)

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.N_nm = N_nm
        self.activation_fn = activation_fn
        self.keepW0 = keepW0
        self.use_spatial_weights = use_spatial_weights
        self.g = 10.0 # Gain factor from user code

        if self.use_spatial_weights:
            self.spatial_net = SpatialWeight(hidden_size=hidden_size, N_nm=N_nm, ell=spatial_ell, scale=spatial_scale, seed=seed)
            # Base tensor for modulated weights, spatial structure applied during initialization of self.weight_hh
            # Or, self.weight_hh_base is learned, and spatialNet is applied in forward.
            # User code: self.weight_hh = nn.Parameter(self.spatialNet(torch.Tensor(H, H, N_NM)))
            # This makes weight_hh fixed after init if spatialNet.forward takes the *base* W.
            # Let's make W_base learnable and apply spatialNet in forward, or make spatialNet's output learnable.
            # For simplicity with original structure: initialize a base and apply spatial modulation once.
            # The parameter will be the result of spatialNet(initial_W_base)
            initial_W_base_hh = torch.Tensor(hidden_size, hidden_size, N_nm) # This is what spatialNet.forward expects
            nn.init.kaiming_uniform_(initial_W_base_hh, a=self.g / math.sqrt(self.hidden_size) if self.hidden_size > 0 else 0.01)
            self.weight_hh = nn.Parameter(self.spatial_net(initial_W_base_hh))
        else:
            self.weight_hh = nn.Parameter(torch.Tensor(hidden_size, hidden_size, N_nm))
            nn.init.kaiming_uniform_(self.weight_hh, a=self.g / math.sqrt(self.hidden_size) if self.hidden_size > 0 else 0.01)


        self.weight_ih = nn.Parameter(torch.Tensor(hidden_size, input_size))
        self.weight_h2nm = nn.Parameter(torch.Tensor(N_nm, hidden_size))
        self.weight_nm2nm = nn.Parameter(torch.Tensor(N_nm, N_nm))

        if keepW0:
            self.weight0_hh = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        else:
            self.register_parameter('weight0_hh', None)
        
        if bias:
            self.bias = nn.Parameter(torch.Tensor(hidden_size))
        else:
            self.register_parameter('bias', None)
        
        self.reset_parameters_custom()

    def reset_parameters_custom(self):
        nn.init.kaiming_uniform_(self.weight_ih, a=math.sqrt(5))
        # self.weight_hh is already initialized (either spatially or kaiming)
        
        if self.N_nm > 0 : # Only init NM weights if N_nm > 0
             nn.init.sparse_(self.weight_h2nm, sparsity=0.9) # Higher sparsity (0.1 density)
             nn.init.zeros_(self.weight_nm2nm)

        if self.weight0_hh is not None: # if keepW0 is True
            nn.init.kaiming_uniform_(self.weight0_hh, a=math.sqrt(5))
        
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight_ih)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

# --- nmRNN Cell (modified from user's s_nmRNNCell) ---
class NMRNNCell(nmRNNCellBase):
    def __init__(self, input_size, hidden_size, N_nm, activation_fn, decay, bias, keepW0,
                 use_spatial_weights, spatial_ell, spatial_scale, seed=0):
        super().__init__(input_size, hidden_size, N_nm, activation_fn, bias, keepW0,
                         use_spatial_weights, spatial_ell, spatial_scale, seed=seed)
        self.decay = decay # Expected to be like exp(-dt/tau)

    def forward(self, current_input, prev_states):
        """
        current_input (Tensor): (batch, input_size)
        prev_states (Tensor): (batch, hidden_size + N_nm)
        Returns:
            next_states (Tensor): (batch, hidden_size + N_nm)
        """
        batch_size = current_input.shape[0]
        
        prev_hidden = prev_states[:, :self.hidden_size]
        prev_nm = prev_states[:, self.hidden_size:] if self.N_nm > 0 else None

        pre_act_hidden = current_input @ self.weight_ih.t()
        if self.weight0_hh is not None:
            pre_act_hidden += prev_hidden @ self.weight0_hh.t()
        
        if prev_nm is not None and self.N_nm > 0:
            # Modulated recurrent contribution
            # self.weight_hh is (H, H, N_NM)
            # prev_hidden is (B, H), prev_nm is (B, N_NM)
            # We need sum_j sum_k (prev_hidden_bj * weight_hh_ijk * prev_nm_bk) -> output_bi
            modulated_rec = torch.einsum('bj,hjk,bk->bh', prev_hidden, self.weight_hh, prev_nm)
            pre_act_hidden += modulated_rec
            
        if self.bias is not None:
            pre_act_hidden += self.bias
        
        current_activity_hidden = self.activation_fn(pre_act_hidden)
        next_hidden = (1.0 - self.decay) * prev_hidden + self.decay * current_activity_hidden # Euler: prev + dt * (-prev/tau + activity)
                                                                                             # If decay = dt/tau, then (1-dt/tau)prev + dt/tau * act
                                                                                             # User code: decay * prev + (1-decay) * activity
                                                                                             # Let's follow user: decay is like (1 - dt/tau)
                                                                                             # So, next = decay*prev + (1-decay)*activity is correct if decay = (1-dt/tau)
                                                                                             # If decay = exp(-dt/tau), then next = decay*prev + (something)*activity
                                                                                             # Sticking to user's: next_h = self.decay * prev_h + (1-self.decay) * activity_h
        
        next_nm_states = None
        if prev_nm is not None and self.N_nm > 0:
            pre_act_nm = prev_hidden @ self.weight_h2nm.t()
            pre_act_nm += prev_nm @ self.weight_nm2nm.t()
            current_activity_nm = self.activation_fn(pre_act_nm) # Assuming same activation for NMs
            next_nm_states = (1.0 - self.decay) * prev_nm + self.decay * current_activity_nm
            # next_nm_states = self.decay * prev_nm + (1 - self.decay) * current_activity_nm


        if next_nm_states is not None:
            next_states_combined = torch.cat([next_hidden, next_nm_states], dim=1)
        else:
            next_states_combined = next_hidden
            
        return next_states_combined

# --- nmRNN Layer Wrapper (adapting s_nmRNNLayer) ---
class NMRNNLayer(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, N_nm, activation_fn, decay, bias, keepW0,
                 use_spatial_weights, spatial_ell, spatial_scale, use_modulated_readout, seed=0, device='cpu'):
        super().__init__()
        self.hidden_size = hidden_size
        self.N_nm = N_nm
        self.output_size = output_size
        self.device = device # Store device
        self.use_modulated_readout = use_modulated_readout

        self.cell = NMRNNCell(input_size, hidden_size, N_nm, activation_fn, decay, bias, keepW0,
                              use_spatial_weights, spatial_ell, spatial_scale, seed=seed)
        
        if self.use_modulated_readout:
            if self.N_nm == 0:
                print("Warning: Modulated readout requested but N_nm is 0. Using fixed readout.")
                self.use_modulated_readout = False # Fallback
                self.readout_fc = nn.Linear(hidden_size, output_size)
                nn.init.xavier_uniform_(self.readout_fc.weight)
                if self.readout_fc.bias is not None: nn.init.zeros_(self.readout_fc.bias)
            else:
                self.weight_readout_modulated = nn.Parameter(torch.Tensor(output_size, hidden_size, N_nm))
                nn.init.kaiming_uniform_(self.weight_readout_modulated, a=1/math.sqrt(hidden_size) if hidden_size > 0 else 0.01)
        else: # Fixed readout
            self.readout_fc = nn.Linear(hidden_size, output_size)
            nn.init.xavier_uniform_(self.readout_fc.weight)
            if self.readout_fc.bias is not None: nn.init.zeros_(self.readout_fc.bias)

    def forward(self, x_seq_permuted, initial_states=None):
        """
        x_seq_permuted (Tensor): Input (seq_length, batch_size, input_size).
        initial_states (Tensor, optional): (batch_size, hidden_size + N_nm).
        Returns:
            outputs_seq (Tensor): (seq_length, batch_size, output_size).
            hidden_states_seq (Tensor): (seq_length, batch_size, hidden_size).
            nm_states_seq (Tensor): (seq_length, batch_size, N_nm) or None.
        """
        seq_length, batch_size, _ = x_seq_permuted.shape

        if initial_states is None:
            current_states = torch.zeros(batch_size, self.hidden_size + self.N_nm, device=self.device)
        else:
            current_states = initial_states
        
        outputs_list = []
        hidden_list = []
        nm_list = []

        for t in range(seq_length):
            current_input = x_seq_permuted[t, :, :]
            current_states = self.cell(current_input, current_states) # (B, H+N_NM)
            
            current_hidden = current_states[:, :self.hidden_size]
            hidden_list.append(current_hidden)

            if self.N_nm > 0:
                current_nm = current_states[:, self.hidden_size:]
                nm_list.append(current_nm)
                if self.use_modulated_readout:
                    # output_t = torch.einsum('bi,oij,bj->bo', current_hidden, self.weight_readout_modulated, current_nm)
                    # Corrected einsum: hidden (b,h), weight (o,h,n), nm (b,n) -> out (b,o)
                    output_t = torch.einsum('bh,ohn,bn->bo', current_hidden, self.weight_readout_modulated, current_nm)

                else: # Fixed readout (even if NMs exist, they don't modulate readout)
                    output_t = self.readout_fc(current_hidden)
            else: # No NMs, must use fixed readout
                output_t = self.readout_fc(current_hidden)
            
            outputs_list.append(output_t)

        outputs_seq = torch.stack(outputs_list, dim=0) # (T, B, O)
        hidden_states_seq = torch.stack(hidden_list, dim=0) # (T, B, H)
        nm_states_seq = torch.stack(nm_list, dim=0) if self.N_nm > 0 else None # (T, B, N_NM)
            
        return outputs_seq, hidden_states_seq, nm_states_seq


# --- Main nmRNN Model Wrappers for different variants ---
class BaseNMRNNWrapper(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, N_nm, activation_fn_name,
                 decay, bias, keepW0, use_spatial_weights, spatial_ell, spatial_scale,
                 use_modulated_readout, grad_clip, device='cpu', seed=0):
        super().__init__()
        torch.manual_seed(seed)
        self.input_size = input_size
        self.hidden_size = hidden_size # n_rnn
        self.output_size = output_size # n_output
        self.N_nm = N_nm
        self.device = device
        
        if activation_fn_name.lower() == 'relu': nonlinearity = nn.ReLU()
        elif activation_fn_name.lower() == 'tanh': nonlinearity = nn.Tanh()
        else: raise ValueError(f"Unsupported activation: {activation_fn_name}")

        self.nm_rnn_layer = NMRNNLayer(
            input_size=input_size, hidden_size=hidden_size, output_size=output_size, N_nm=N_nm,
            activation_fn=nonlinearity, decay=decay, bias=bias, keepW0=keepW0,
            use_spatial_weights=use_spatial_weights, spatial_ell=spatial_ell, spatial_scale=spatial_scale,
            use_modulated_readout=use_modulated_readout, seed=seed, device=device
        )
        
        # Gradient clipping (from user code, applied differently in PyTorch)
        # It's generally better to apply clipping in the optimizer step or via hooks if needed per-parameter.
        # For now, this is a placeholder if specific per-parameter hook logic is complex to replicate.
        self.grad_clip_value = grad_clip


    def forward(self, x_seq_orig): # Expected input (B, C, T)
        # Permute to (T, B, C) for internal RNN layer
        x_seq_permuted = x_seq_orig.permute(2, 0, 1) 
        
        # initial_states are handled inside NMRNNLayer if None is passed
        output_seq_TBC, hidden_seq_TBH, _ = self.nm_rnn_layer(x_seq_permuted, initial_states=None)
        
        # Permute output to (B, T, O)
        output_seq_BTO = output_seq_TBC.permute(1, 0, 2)
        # Permute hidden states to (B, T, H)
        hidden_seq_BTH = hidden_seq_TBH.permute(1, 0, 2)

        if self.output_size == 1:
            output_seq_BTO = output_seq_BTO.squeeze(-1) # (B,T)

        return output_seq_BTO, hidden_seq_BTH


class NMRNN_Spatial_ModulatedReadout(BaseNMRNNWrapper):
    def __init__(self, input_size, hidden_size, output_size, N_nm, activation_fn_name, decay, bias, keepW0, 
                 spatial_ell, spatial_scale, grad_clip, device='cpu', seed=0):
        super().__init__(input_size, hidden_size, output_size, N_nm, activation_fn_name, decay, bias, keepW0,
                         use_spatial_weights=True, spatial_ell=spatial_ell, spatial_scale=spatial_scale,
                         use_modulated_readout=True, grad_clip=grad_clip, device=device, seed=seed)

class NMRNN_NoSpatial_ModulatedReadout(BaseNMRNNWrapper):
    def __init__(self, input_size, hidden_size, output_size, N_nm, activation_fn_name, decay, bias, keepW0,
                 grad_clip, device='cpu', seed=0):
        super().__init__(input_size, hidden_size, output_size, N_nm, activation_fn_name, decay, bias, keepW0,
                         use_spatial_weights=False, spatial_ell=0.1, spatial_scale=1.0, # Dummy spatial params
                         use_modulated_readout=True, grad_clip=grad_clip, device=device, seed=seed)

class NMRNN_Spatial_FixedReadout(BaseNMRNNWrapper):
    def __init__(self, input_size, hidden_size, output_size, N_nm, activation_fn_name, decay, bias, keepW0,
                 spatial_ell, spatial_scale, grad_clip, device='cpu', seed=0):
        super().__init__(input_size, hidden_size, output_size, N_nm, activation_fn_name, decay, bias, keepW0,
                         use_spatial_weights=True, spatial_ell=spatial_ell, spatial_scale=spatial_scale,
                         use_modulated_readout=False, grad_clip=grad_clip, device=device, seed=seed)

