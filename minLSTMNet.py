import torch
import torch.nn as nn
import torch.nn.functional as F

def g(x):
    return torch.where(x >= 0, x+0.5, torch.sigmoid(x))

def log_g(x):
    return torch.where(x >= 0, (F.relu(x)+0.5).log(),-F.softplus(-x))

def parallel_scan_log(log_coeffs, log_values):
    a_star = F.pad(torch.cumsum(log_coeffs, dim=1), (0, 0, 1, 0)).squeeze(1)
    log_h0_plus_b_star = torch.logcumsumexp(log_values - a_star, dim=1).squeeze(1)
    log_h = a_star + log_h0_plus_b_star
    return torch.exp(log_h)

class MinLSTMCell(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, mode='seq', device=None, dtype=None):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        self.linear_f = nn.Linear(input_size, hidden_size, device=device, dtype=dtype)
        self.linear_i = nn.Linear(input_size, hidden_size, device=device, dtype=dtype)
        self.linear_h = nn.Linear(input_size, hidden_size, device=device, dtype=dtype)

        self.mode = mode


    def forward(self, x_t, h_prev=None):
        
        if h_prev is None:
            h_prev = torch.zeros(x_t.size(0), self.hidden_size, device=x_t.device, dtype=x_t.dtype)
        
        if self.mode == 'seq':
            is_batched = x_t.dim() == 2
            if not is_batched:
                x_t = x_t.unsqueeze(0)
            f_t = torch.sigmoid(self.linear_f(x_t))
            i_t = torch.sigmoid(self.linear_i(x_t))
            
            h_tilde = g(self.linear_h(x_t))

            f_prime_t = f_t / (f_t + i_t)
            i_prime_t = i_t / (f_t + i_t)
            h_t = f_prime_t * h_prev + i_prime_t * h_tilde

            if not is_batched:
                h_t = h_t.squeeze(0)
        elif self.mode == 'par':
            diff = F.softplus(-self.linear_f(x_t)) - F.softplus(-self.linear_i(x_t))
            log_f = -F.softplus(diff)
            log_i = -F.softplus(-diff)
            log_h_0 = log_g(h_prev)
            log_tilde_h = log_g(self.linear_h(x_t))

            log_coeff = log_f.unsqueeze(1)
            log_val = torch.cat([log_h_0.unsqueeze(1), (log_i + log_tilde_h)], dim=1)
            print(f"coeff: {log_coeff.shape}")
            print(f"Logval : {log_val.shape}")
            h_t = parallel_scan_log(log_coeff,log_val)
            print(f"h_t.shape: {h_t.shape}")
        else:
            raise ValueError(f"Expected mode 'seq' or 'par', but got {self.mode}") 
        
        
        return h_t

class MinLSTM(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, mode="seq", 
                 batch_first: bool = False, bidirectional: bool = False, device=None, dtype=None):
        
        super().__init__()
        self.mode = mode
        self.mini_lstm_fwd = MinLSTMCell(input_size, hidden_size, mode, device=device, dtype=dtype)
        self.bidirectional = bidirectional
        if bidirectional:
            self.mini_lstm_bwd = MinLSTMCell(input_size, hidden_size, mode, device=device, dtype=dtype)
        self.batch_first = batch_first

    def forward(self, x, h_0=None):
        if self.batch_first:
            x = x.transpose(0, 1)
        
        seq_len, batch_size, _ = x.size()
        hidden_size = self.mini_lstm_fwd.hidden_size
        

        if self.mode == 'seq':
            if h_0 is None:
                h_0 = torch.zeros(batch_size, hidden_size, device=x.device, dtype=x.dtype)
            
            output_fwd = []
            h_t_fwd = h_0
            
            for t in range(seq_len):
                h_t_fwd = self.mini_lstm_fwd(x[t], h_t_fwd)
                output_fwd.append(h_t_fwd)
            
            output_fwd = torch.stack(output_fwd, dim=0)
            
            if self.bidirectional:
                output_bwd = []
                h_t_bwd = h_0
                for t in reversed(range(seq_len)):
                    h_t_bwd = self.mini_lstm_bwd(x[t], h_t_bwd)
                    output_bwd.append(self.dropout(h_t_bwd))
                output_bwd = torch.stack(output_bwd[::-1], dim=0)
                output = torch.cat([output_fwd, output_bwd], dim=2)
            else:
                output = output_fwd

            if self.batch_first:
                output = output.transpose(0, 1)
            
            return output, h_t_fwd
        elif self.mode == 'par':
            output = self.mini_lstm_fwd(x)
            return output,output
        else:
            raise ValueError(f"Expected mode 'seq' or 'par', but got {self.mode}") 
