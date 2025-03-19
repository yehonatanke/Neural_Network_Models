import numpy as np
import time
import random
import math
from typing import List, Tuple, Dict, Any, Optional


class Tensor:
    def __init__(self, data, requires_grad=False):
        self.data = np.array(data, dtype=np.float64)
        self.requires_grad = requires_grad
        self.grad = None
        self.grad_fn = None
        self._backward_hooks = []
        self.shape = self.data.shape
        
    def backward(self, grad=None):
        if grad is None:
            grad = np.ones_like(self.data)
        
        self.grad = grad
        
        for hook in self._backward_hooks:
            hook(self)
            
        if self.grad_fn:
            self.grad_fn(grad)
            
    def __add__(self, other):
        if isinstance(other, (int, float)):
            other = Tensor(np.full_like(self.data, other))
        
        result = Tensor(self.data + other.data, requires_grad=self.requires_grad or other.requires_grad)
        
        if self.requires_grad or other.requires_grad:
            def _backward(grad):
                if self.requires_grad:
                    if self.grad is None:
                        self.grad = grad
                    else:
                        self.grad += grad
                        
                if other.requires_grad:
                    if other.grad is None:
                        other.grad = grad
                    else:
                        other.grad += grad
            
            result.grad_fn = _backward
            
        return result
        
    def __mul__(self, other):
        if isinstance(other, (int, float)):
            other = Tensor(np.full_like(self.data, other))
            
        result = Tensor(self.data * other.data, requires_grad=self.requires_grad or other.requires_grad)
        
        if self.requires_grad or other.requires_grad:
            def _backward(grad):
                if self.requires_grad:
                    self_grad = grad * other.data
                    if self.grad is None:
                        self.grad = self_grad
                    else:
                        self.grad += self_grad
                        
                if other.requires_grad:
                    other_grad = grad * self.data
                    if other.grad is None:
                        other.grad = other_grad
                    else:
                        other.grad += other_grad
            
            result.grad_fn = _backward
            
        return result
        
    def matmul(self, other):
        result = Tensor(np.matmul(self.data, other.data), requires_grad=self.requires_grad or other.requires_grad)
        
        if self.requires_grad or other.requires_grad:
            def _backward(grad):
                if self.requires_grad:
                    self_grad = np.matmul(grad, other.data.T)
                    if self.grad is None:
                        self.grad = self_grad
                    else:
                        self.grad += self_grad
                        
                if other.requires_grad:
                    other_grad = np.matmul(self.data.T, grad)
                    if other.grad is None:
                        other.grad = other_grad
                    else:
                        other.grad += other_grad
            
            result.grad_fn = _backward
            
        return result
        
    def mean(self):
        result = Tensor(np.mean(self.data), requires_grad=self.requires_grad)
        
        if self.requires_grad:
            def _backward(grad):
                if self.grad is None:
                    self.grad = np.full_like(self.data, grad / np.size(self.data))
                else:
                    self.grad += np.full_like(self.data, grad / np.size(self.data))
            
            result.grad_fn = _backward
            
        return result
        
    def sum(self, axis=None, keepdims=False):
        result = Tensor(np.sum(self.data, axis=axis, keepdims=keepdims), requires_grad=self.requires_grad)
        
        if self.requires_grad:
            def _backward(grad):
                if not keepdims and axis is not None:
                    grad_expanded = np.expand_dims(grad, axis)
                else:
                    grad_expanded = grad
                
                if axis is not None:
                    grad_expanded = np.broadcast_to(grad_expanded, self.data.shape)
                
                if self.grad is None:
                    self.grad = grad_expanded
                else:
                    self.grad += grad_expanded
            
            result.grad_fn = _backward
            
        return result
        
    def exp(self):
        result = Tensor(np.exp(self.data), requires_grad=self.requires_grad)
        
        if self.requires_grad:
            def _backward(grad):
                self_grad = grad * np.exp(self.data)
                if self.grad is None:
                    self.grad = self_grad
                else:
                    self.grad += self_grad
            
            result.grad_fn = _backward
            
        return result
        
    def log(self):
        result = Tensor(np.log(self.data + 1e-12), requires_grad=self.requires_grad)
        
        if self.requires_grad:
            def _backward(grad):
                self_grad = grad / (self.data + 1e-12)
                if self.grad is None:
                    self.grad = self_grad
                else:
                    self.grad += self_grad
            
            result.grad_fn = _backward
            
        return result
        
    def reshape(self, *shape):
        result = Tensor(self.data.reshape(*shape), requires_grad=self.requires_grad)
        
        if self.requires_grad:
            def _backward(grad):
                if self.grad is None:
                    self.grad = grad.reshape(self.data.shape)
                else:
                    self.grad += grad.reshape(self.data.shape)
            
            result.grad_fn = _backward
            
        return result
        
    def transpose(self):
        result = Tensor(self.data.T, requires_grad=self.requires_grad)
        
        if self.requires_grad:
            def _backward(grad):
                if self.grad is None:
                    self.grad = grad.T
                else:
                    self.grad += grad.T
            
            result.grad_fn = _backward
            
        return result
        
    def __neg__(self):
        return self * -1
        
    def __sub__(self, other):
        return self + (-other)
        
    def __rsub__(self, other):
        return other + (-self)
        
    def __truediv__(self, other):
        if isinstance(other, (int, float)):
            return self * (1 / other)
        return self * (other ** -1)
        
    def __pow__(self, power):
        result = Tensor(self.data ** power, requires_grad=self.requires_grad)
        
        if self.requires_grad:
            def _backward(grad):
                self_grad = grad * power * self.data ** (power - 1)
                if self.grad is None:
                    self.grad = self_grad
                else:
                    self.grad += self_grad
            
            result.grad_fn = _backward
            
        return result
        
    def tanh(self):
        result = Tensor(np.tanh(self.data), requires_grad=self.requires_grad)
        
        if self.requires_grad:
            def _backward(grad):
                self_grad = grad * (1 - np.tanh(self.data) ** 2)
                if self.grad is None:
                    self.grad = self_grad
                else:
                    self.grad += self_grad
            
            result.grad_fn = _backward
            
        return result
        
    def relu(self):
        result = Tensor(np.maximum(0, self.data), requires_grad=self.requires_grad)
        
        if self.requires_grad:
            def _backward(grad):
                self_grad = grad * (self.data > 0)
                if self.grad is None:
                    self.grad = self_grad
                else:
                    self.grad += self_grad
            
            result.grad_fn = _backward
            
        return result
        
    def softmax(self, axis=-1):
        exp_data = np.exp(self.data - np.max(self.data, axis=axis, keepdims=True))
        result_data = exp_data / np.sum(exp_data, axis=axis, keepdims=True)
        result = Tensor(result_data, requires_grad=self.requires_grad)
        
        if self.requires_grad:
            def _backward(grad):
                s = result_data
                grad_data = s * (grad - np.sum(grad * s, axis=axis, keepdims=True))
                if self.grad is None:
                    self.grad = grad_data
                else:
                    self.grad += grad_data
            
            result.grad_fn = _backward
            
        return result

class Parameter(Tensor):
    def __init__(self, data, requires_grad=True):
        super().__init__(data, requires_grad=requires_grad)
        
    def zero_grad(self):
        self.grad = None

class Module:
    def __init__(self):
        self._parameters = {}
        self._modules = {}
        self._is_training = True
        
    def parameters(self):
        params = []
        for param in self._parameters.values():
            params.append(param)
            
        for module in self._modules.values():
            params.extend(module.parameters())
            
        return params
        
    def zero_grad(self):
        for param in self.parameters():
            param.zero_grad()
            
    def train(self, mode=True):
        self._is_training = mode
        for module in self._modules.values():
            module.train(mode)
        return self
        
    def eval(self):
        return self.train(False)
        
    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

class Linear(Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        
        limit = math.sqrt(6 / (in_features + out_features))
        weights = np.random.uniform(-limit, limit, (in_features, out_features))
        self._parameters['weight'] = Parameter(weights)
        
        if bias:
            self._parameters['bias'] = Parameter(np.zeros(out_features))
        else:
            self._parameters['bias'] = None
            
    def forward(self, x):
        weight = self._parameters['weight']
        bias = self._parameters['bias']
        
        out = x.matmul(weight)
        if bias is not None:
            out = out + bias
            
        return out

class Sequential(Module):
    def __init__(self, *modules):
        super().__init__()
        
        for idx, module in enumerate(modules):
            self._modules[str(idx)] = module
            
    def forward(self, x):
        for module in self._modules.values():
            x = module(x)
            
        return x

class Dropout(Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p
        
    def forward(self, x):
        if not self._is_training or self.p == 0:
            return x
            
        mask = np.random.binomial(1, 1 - self.p, x.shape) / (1 - self.p)
        return x * Tensor(mask)

class BatchNorm1d(Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super().__init__()
        
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        
        self._parameters['weight'] = Parameter(np.ones(num_features))
        self._parameters['bias'] = Parameter(np.zeros(num_features))
        
        self.register_buffer('running_mean', np.zeros(num_features))
        self.register_buffer('running_var', np.ones(num_features))
        
    def register_buffer(self, name, data):
        setattr(self, name, data)
        
    def forward(self, x):
        if self._is_training:
            mean = np.mean(x.data, axis=0)
            var = np.var(x.data, axis=0)
            
            self.running_mean = self.momentum * mean + (1 - self.momentum) * self.running_mean
            self.running_var = self.momentum * var + (1 - self.momentum) * self.running_var
        else:
            mean = self.running_mean
            var = self.running_var
            
        weight = self._parameters['weight']
        bias = self._parameters['bias']
        
        x_normalized = (x.data - mean) / np.sqrt(var + self.eps)
        y = weight.data * x_normalized + bias.data
        
        result = Tensor(y, requires_grad=x.requires_grad)
        
        if x.requires_grad:
            def _backward(grad):
                N = x.data.shape[0]
                dx_normalized = grad * weight.data
                
                dvar = np.sum(dx_normalized * (x.data - mean) * -0.5 * (var + self.eps) ** -1.5, axis=0)
                dmean = np.sum(dx_normalized * -1 / np.sqrt(var + self.eps), axis=0) + dvar * np.mean(-2 * (x.data - mean), axis=0)
                
                dx = dx_normalized / np.sqrt(var + self.eps) + dvar * 2 * (x.data - mean) / N + dmean / N
                
                if x.grad is None:
                    x.grad = dx
                else:
                    x.grad += dx
                    
                if weight.requires_grad:
                    dweight = np.sum(grad * x_normalized, axis=0)
                    if weight.grad is None:
                        weight.grad = dweight
                    else:
                        weight.grad += dweight
                        
                if bias.requires_grad:
                    dbias = np.sum(grad, axis=0)
                    if bias.grad is None:
                        bias.grad = dbias
                    else:
                        bias.grad += dbias
            
            result.grad_fn = _backward
            
        return result

class RNN(Module):
    def __init__(self, input_size, hidden_size, num_layers=1, nonlinearity='tanh', bias=True, dropout=0):
        super().__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.nonlinearity = nonlinearity
        
        self.rnn_cells = []
        for i in range(num_layers):
            layer_input_size = input_size if i == 0 else hidden_size
            cell = RNNCell(layer_input_size, hidden_size, nonlinearity, bias)
            self._modules[f'rnn_{i}'] = cell
            self.rnn_cells.append(cell)
            
        if dropout > 0 and num_layers > 1:
            self.dropout = Dropout(dropout)
        else:
            self.dropout = None
            
    def forward(self, x, hidden=None):
        seq_len, batch_size, input_size = x.shape
        
        if hidden is None:
            hidden = [Tensor(np.zeros((batch_size, self.hidden_size))) for _ in range(self.num_layers)]
            
        outputs = []
        new_hidden = []
        
        for i in range(seq_len):
            layer_input = Tensor(x.data[i])
            
            for j, cell in enumerate(self.rnn_cells):
                h = cell(layer_input, hidden[j])
                layer_input = h
                
                if self.dropout is not None and j < self.num_layers - 1:
                    layer_input = self.dropout(layer_input)
                    
                if i == seq_len - 1:
                    new_hidden.append(h)
                    
            outputs.append(layer_input.data)
            
        output = Tensor(np.stack(outputs))
        return output, new_hidden

class RNNCell(Module):
    def __init__(self, input_size, hidden_size, nonlinearity='tanh', bias=True):
        super().__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.nonlinearity = nonlinearity
        
        self._parameters['weight_ih'] = Parameter(np.random.randn(input_size, hidden_size) * 0.1)
        self._parameters['weight_hh'] = Parameter(np.random.randn(hidden_size, hidden_size) * 0.1)
        
        if bias:
            self._parameters['bias_ih'] = Parameter(np.zeros(hidden_size))
            self._parameters['bias_hh'] = Parameter(np.zeros(hidden_size))
        else:
            self._parameters['bias_ih'] = None
            self._parameters['bias_hh'] = None
            
    def forward(self, x, hidden):
        weight_ih = self._parameters['weight_ih']
        weight_hh = self._parameters['weight_hh']
        bias_ih = self._parameters['bias_ih']
        bias_hh = self._parameters['bias_hh']
        
        gates = x.matmul(weight_ih)
        if bias_ih is not None:
            gates = gates + bias_ih
            
        gates = gates + hidden.matmul(weight_hh)
        if bias_hh is not None:
            gates = gates + bias_hh
            
        if self.nonlinearity == 'tanh':
            hidden = gates.tanh()
        else:
            hidden = gates.relu()
            
        return hidden

class LSTM(Module):
    def __init__(self, input_size, hidden_size, num_layers=1, bias=True, dropout=0):
        super().__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm_cells = []
        for i in range(num_layers):
            layer_input_size = input_size if i == 0 else hidden_size
            cell = LSTMCell(layer_input_size, hidden_size, bias)
            self._modules[f'lstm_{i}'] = cell
            self.lstm_cells.append(cell)
            
        if dropout > 0 and num_layers > 1:
            self.dropout = Dropout(dropout)
        else:
            self.dropout = None
            
    def forward(self, x, hidden=None):
        seq_len, batch_size, input_size = x.shape
        
        if hidden is None:
            h = [Tensor(np.zeros((batch_size, self.hidden_size))) for _ in range(self.num_layers)]
            c = [Tensor(np.zeros((batch_size, self.hidden_size))) for _ in range(self.num_layers)]
            hidden = (h, c)
            
        h, c = hidden
        outputs = []
        new_h = []
        new_c = []
        
        for i in range(seq_len):
            layer_input = Tensor(x.data[i])
            
            for j, cell in enumerate(self.lstm_cells):
                h_j, c_j = cell(layer_input, (h[j], c[j]))
                layer_input = h_j
                
                if self.dropout is not None and j < self.num_layers - 1:
                    layer_input = self.dropout(layer_input)
                    
                if i == seq_len - 1:
                    new_h.append(h_j)
                    new_c.append(c_j)
                    
            outputs.append(layer_input.data)
            
        output = Tensor(np.stack(outputs))
        return output, (new_h, new_c)

class LSTMCell(Module):
    def __init__(self, input_size, hidden_size, bias=True):
        super().__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        self._parameters['weight_ih'] = Parameter(np.random.randn(input_size, 4 * hidden_size) * 0.1)
        self._parameters['weight_hh'] = Parameter(np.random.randn(hidden_size, 4 * hidden_size) * 0.1)
        
        if bias:
            self._parameters['bias_ih'] = Parameter(np.zeros(4 * hidden_size))
            self._parameters['bias_hh'] = Parameter(np.zeros(4 * hidden_size))
        else:
            self._parameters['bias_ih'] = None
            self._parameters['bias_hh'] = None
            
    def forward(self, x, hidden):
        h, c = hidden
        weight_ih = self._parameters['weight_ih']
        weight_hh = self._parameters['weight_hh']
        bias_ih = self._parameters['bias_ih']
        bias_hh = self._parameters['bias_hh']
        
        gates = x.matmul(weight_ih)
        if bias_ih is not None:
            gates = gates + bias_ih
            
        gates = gates + h.matmul(weight_hh)
        if bias_hh is not None:
            gates = gates + bias_hh
            
        chunked_gates = np.split(gates.data, 4, axis=1)
        i = Tensor(chunked_gates[0]).sigmoid()
        f = Tensor(chunked_gates[1]).sigmoid()
        g = Tensor(chunked_gates[2]).tanh()
        o = Tensor(chunked_gates[3]).sigmoid()
        
        new_c = f * c + i * g
        new_h = o * new_c.tanh()
        
        return new_h, new_c
        
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

Tensor.sigmoid = lambda self: Tensor(sigmoid(self.data), requires_grad=self.requires_grad)

class Optimizer:
    def __init__(self, params, lr=0.01):
        self.params = list(params)
        self.lr = lr
        
    def zero_grad(self):
        for param in self.params:
            param.zero_grad()
            
    def step(self):
        raise NotImplementedError

class SGD(Optimizer):
    def __init__(self, params, lr=0.01, momentum=0, weight_decay=0, nesterov=False):
        super().__init__(params, lr)
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.nesterov = nesterov
        self.state = {}
        
    def step(self):
        for param in self.params:
            if param.grad is None:
                continue
                
            grad = param.grad
            
            if self.weight_decay != 0:
                grad = grad + self.weight_decay * param.data
                
            if param not in self.state:
                self.state[param] = np.zeros_like(param.data)
                
            momentum_buffer = self.state[param]
            momentum_buffer = self.momentum * momentum_buffer + grad
            
            if self.nesterov:
                grad = grad + self.momentum * momentum_buffer
            else:
                grad = momentum_buffer
                
            self.state[param] = momentum_buffer
            param.data -= self.lr * grad

class Adam(Optimizer):
    def __init__(self, params, lr=0.001, betas=(0.9, 0.999), eps=1e-8, weight_decay=0):
        super().__init__(params, lr)
        self.betas = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.state = {}
        self.t = 0
        
    def step(self):
        self.t += 1
        
        for param in self.params:
            if param.grad is None:
                continue
                
            grad = param.grad
            
            if self.weight_decay != 0:
                grad = grad + self.weight_decay * param.data
                
            if param not in self.state:
                self.state[param] = {
                    'm': np.zeros_like(param.data),
                    'v': np.zeros_like(param.data)
                }
                
            state = self.state[param]
            
            m, v = state['m'], state['v']
            beta1, beta2 = self.betas
            
            m = beta1 * m + (1 - beta1) * grad
            v = beta2 * v + (1 - beta2) * (grad ** 2)
            
            m_hat = m / (1 - beta1 ** self.t)
            v_hat = v / (1 - beta2 ** self.t)
            
            state['m'], state['v'] = m, v
            
            param.data -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)


class CrossEntropyLoss:
    def __init__(self, reduction='mean'):
        self.reduction = reduction
        
    def __call__(self, logits, targets):
        batch_size = logits.shape[0]
        num_classes = logits.shape[1]
        
        if targets.shape[-1] == num_classes:
            targets_oh = targets.data
        else:
            targets_oh = np.zeros((batch_size, num_classes))
            targets_oh[np.arange(batch_size), targets.data.astype(int).flatten()] = 1
            
        log_softmax = logits.log_softmax(axis=1)
        loss = -np.sum(targets_oh * log_softmax.data, axis=1)
        
        if self.reduction == 'mean':
            loss = np.mean(loss)
        elif self.reduction == 'sum':
            loss = np.sum(loss)
            
        result = Tensor(loss, requires_grad=logits.requires_grad)
        
        if logits.requires_grad:
            def _backward(grad):
                if isinstance(grad, np.ndarray):
                    g = grad
                else:
                    g = np.array(grad)
                    
                if self.reduction == 'mean':
                    g = g / batch_size
                    
                dx = np.zeros_like(logits.data)
                dx += np.exp(logits.data) / np.sum(np.exp(logits.data), axis=1, keepdims=True)
                dx -= targets_oh
                
                if self.reduction == 'mean':
                    dx /= batch_size
                elif self.reduction == 'sum':
                    dx *= g
                    
                if logits.grad is None:
                    logits.grad = dx
                else:
                    logits.grad += dx
                    
            result.grad_fn = _backward
            
        return result

class DeepLearningModel:
    def __init__(self, input_size, hidden_sizes, output_size, 
                 dropout_rate=0.5, use_batch_norm=True, use_residual=True):
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size
        self.dropout_rate = dropout_rate
        self.use_batch_norm = use_batch_norm
        self.use_residual = use_residual
        
        self.layers = []
        
        prev_size = input_size
        for i, size in enumerate(hidden_sizes):
            layer = []
            
            layer.append(Linear(prev_size, size))
            
            if use_batch_norm:
                layer.append(BatchNorm1d(size))
                
            layer.append(Dropout(dropout_rate))
            
            if use_residual and prev_size == size:
                self.layers.append(ResidualBlock(Sequential(*layer)))
            else:
                self.layers.append(Sequential(*layer))
                
            prev_size = size
            
        self.output_layer = Linear(prev_size, output_size)
        self.model = Sequential(*self.layers, self.output_layer)
        
    def forward(self, x):
        return self.model(x)
        
    def train(self, x_train, y_train, x_val=None, y_val=None, 
              batch_size=32, epochs=100, lr=0.001, optimizer_type='adam',
              early_stopping_patience=10, lr_scheduler=None):
        
        if optimizer_type.lower() == 'adam':
            optimizer = Adam(self.model.parameters(), lr=lr)
        else:
            optimizer = SGD(self.model.parameters(), lr=lr, momentum=0.9)
            
        criterion = CrossEntropyLoss()
        
        n_samples = x_train.shape[0]
        n_batches = math.ceil(n_samples / batch_size)
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(epochs):
            epoch_start_time = time.time()
            self.model.train()
            
            indices = np.random.permutation(n_samples)
            epoch_loss = 0
            
            for batch in range(n_batches):
                start_idx = batch * batch_size
                end_idx = min((batch + 1) * batch_size, n_samples)
                
                batch_indices = indices[start_idx:end_idx]
                
                x_batch = Tensor(x_train[batch_indices])
                y_batch = Tensor(y_train[batch_indices])
                
                optimizer.zero_grad()
                
                y_pred = self.forward(x_batch)
                loss = criterion(y_pred, y_batch)
                
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.data * (end_idx - start_idx)
            
            epoch_loss /= n_samples
            
            if x_val is not None and y_val is not None:
                self.model.eval()
                
                val_pred = self.forward(Tensor(x_val))
                val_loss = criterion(val_pred, Tensor(y_val))
                
                if val_loss.data < best_val_loss:
                    best_val_loss = val_loss.data
                    patience_counter = 0
                else:
                    patience_counter += 1
                    
                if patience_counter >= early_stopping_patience:
                    print(f"Early stopping at epoch {epoch}")
                    break
                    
            epoch_end_time = time.time()
            
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}, Time: {epoch_end_time - epoch_start_time:.2f}s")
                
            if lr_scheduler:
                if isinstance(lr_scheduler, StepLR):
                    if (epoch + 1) % lr_scheduler.step_size == 0:
                        optimizer.lr *= lr_scheduler.gamma
                elif isinstance(lr_scheduler, ReduceLROnPlateau):
                    if x_val is not None
