# metrics.py
import torch

# --- Simple Proxy Metrics ---

def calculate_mse_norm(delta_g_layer: torch.Tensor, num_params: int):
    """Calculates the normalized Mean Squared Error (MSE) of the quantization error."""
    return torch.sum(delta_g_layer.pow(2)) / num_params

def calculate_fisher_norm(g_layer: torch.Tensor, delta_g_layer: torch.Tensor, num_params: int):
    """
    Calculates the normalized Fisher-weighted error.
    Uses the diagonal of the Fisher Information Matrix, approximated by the square of the gradient.
    Metric: (delta_g)^T * diag(g^2) * (delta_g) = sum((g * delta_g)^2)
    """
    # Element-wise multiplication approximates the diagonal Fisher weighting
    return torch.sum(g_layer.pow(2) * delta_g_layer.pow(2)) / num_params

def calculate_distribution_metric(g: torch.Tensor, g_quant: torch.Tensor) -> float:
    """
    计算基于数据分布的度量 (Distribution Metric)。
    公式: | sum(|g|) - sum(|g_quant|) | / sum(|g|)
    即: | ||g||_1 - ||g_quant||_1 | / ||g||_1
    """
    # 1. 计算 L1 范数 (即 sum(|x|))
    l1_g = torch.norm(g, p=1)
    l1_g_quant = torch.norm(g_quant, p=1)
    
    # 2. 防止分母为 0
    # epsilon = 1e-12
    
    # 3. 计算公式
    # 分子: | sum(|g|) - sum(|g_quant|) |
    numerator = torch.abs(l1_g - l1_g_quant)
    
    # 分母: sum(|g|)
    denominator = l1_g # + epsilon
    
    # 返回标量 float
    return (numerator / denominator).item()

def calculate_taylor_1st_g_t_norm(g_layer: torch.Tensor, delta_g_layer: torch.Tensor, 
                              learning_rate: float, num_params: int):
    """
    Calculates the normalized first-order Taylor approximation of the loss change.
    Approximation: L_diff ≈ η * g_t^T ⋅ Δg_t
    """
    return -learning_rate * torch.sum(g_layer * delta_g_layer) / num_params

def calculate_taylor_1st_g_t_plus_1_norm(g_t_plus_1_layer: torch.Tensor, delta_g_layer: torch.Tensor,
                                        learning_rate: float, num_params: int):
    """
    Calculates the normalized first-order Taylor approximation of the loss change using g_{t+1}.
    Approximation: L_diff ≈ η * g_{t+1}^T ⋅ Δg_t
    This is a "lookahead" metric used to validate the g_t approximation.
    """
    return -learning_rate * torch.sum(g_t_plus_1_layer * delta_g_layer) / num_params

# --- Second-Order Proxy Metric (HVP) ---

def calculate_hvp_and_taylor_2nd_norm(
    model: torch.nn.Module,
    loss_fn: torch.nn.Module,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    layer_params_list: list,
    delta_g_layer: torch.Tensor,
    learning_rate: float,
    num_params: int
):
    """
    Calculates the normalized second-order Taylor approximation using a Hessian-Vector Product (HVP).
    Approximation: L_diff ≈ 0.5 * η^2 * Δg_t^T ⋅ H ⋅ Δg_t

    This is computationally expensive as it requires a second backward pass.

    Args:
        model: The neural network model.
        loss_fn: The loss function.
        inputs, targets: The current batch of data.
        layer_params_list: A list of parameter tensors for the specific layer being analyzed.
        delta_g_layer: The flattened quantization error vector for the layer.
        learning_rate: The current learning rate.
        num_params: Number of parameters in the layer.

    Returns:
        torch.Tensor: The normalized second-order term value.
    """
    # Ensure delta_g_layer does not have a grad_fn.
    # .detach() creates a new tensor that shares the same storage but is detached from the graph.
    delta_g_layer_detached = delta_g_layer.detach()

    model.zero_grad()
    
    # --- Step 1: First backward pass to get the gradients (g_t) ---
    outputs = model(inputs)
    loss = loss_fn(outputs, targets)
    
    # We need to create the graph for the gradients to compute the HVP
    grads_with_graph = torch.autograd.grad(loss, layer_params_list, create_graph=True, allow_unused=True)

    # Filter out None gradients and flatten the rest
    # If a parameter was not used, its grad will be None. We should treat it as a zero vector.
    filtered_grads = []
    for p, g in zip(layer_params_list, grads_with_graph):
        if g is not None:
            filtered_grads.append(g)
    if not filtered_grads:
        return torch.tensor(0.0, device=delta_g_layer.device)
        
    g_layer_with_graph = torch.cat([g.contiguous().view(-1) for g in filtered_grads])
    
    # --- Step 2: Compute the dot product between gradient g_t and quantization error delta_g ---
    # This is the term inside the derivative for the HVP: (g_t^T ⋅ Δg_t)
    grad_x_delta_dot_product = torch.sum(g_layer_with_graph * delta_g_layer_detached)

    # --- Step 3: Second backward pass (the HVP) ---
    # This computes (d/dw (g_t^T ⋅ Δg_t)), which is equal to H ⋅ Δg_t
    # 这是最关键的修改。我们对点积结果调用 .backward()。
    # PyTorch会自动计算这个点积对于模型中所有参数的梯度。
    # 这个梯度，根据链式法则，正好就是 HVP (H * delta_g) 的结果。
    # 这些结果会自动累加到对应参数的 .grad 属性中。
    grad_x_delta_dot_product.backward()

    # Filter out None HVPs and flatten the rest, treating None as zero.
    hvp_results = []
    for p in layer_params_list:
        # 如果参数没有梯度（例如因为 allow_unused=True），我们将其视为0
        if p.grad is not None:
            hvp_results.append(p.grad.contiguous().view(-1))
        else:
            hvp_results.append(torch.zeros_like(p).view(-1))
            
    hvp_vector = torch.cat(hvp_results)


    # --- Step 4: Calculate the final second-order term ---
    # Term = 0.5 * η^2 * Δg_t^T ⋅ (H ⋅ Δg_t)
    taylor_2nd_term = 0.5 * (learning_rate**2) * torch.sum(delta_g_layer * hvp_vector)
    
    return taylor_2nd_term / num_params

# 5. 定义模拟更新步骤的函数
def _simulate_adamw_update_step(grad: torch.Tensor, 
                                m_prev: torch.Tensor, 
                                v_prev: torch.Tensor, 
                                step: int, 
                                lr: float, 
                                beta1: float, 
                                beta2: float, 
                                eps: float) -> torch.Tensor:
    """
    [底层辅助函数] 模拟 AdamW 的单步更新逻辑。
    只进行数学运算，不依赖 optimizer 对象。

    Args:
        grad: 梯度
        m_prev: 上一步的一阶矩
        v_prev: 上一步的二阶矩
        step: 当前步数
        lr, beta1, beta2, eps: AdamW超参数
        
    Returns:
        delta_w: 参数变化量 (Δθ = -lr * step_val)
    """
    # 1. 模拟动量更新 (不改变输入的 m_prev, v_prev)
    m_t = torch.add(m_prev * beta1, grad, alpha=1 - beta1)
    v_t = torch.add(v_prev * beta2, grad.pow(2), alpha=1 - beta2)
    
    # 2. 偏差修正 (Bias Correction)
    bias_correction1 = 1 - beta1 ** step
    bias_correction2 = 1 - beta2 ** step
    
    m_hat = m_t / bias_correction1
    v_hat = v_t / bias_correction2
    
    # 3. 计算更新步长
    denom = v_hat.sqrt().add_(eps)
    step_val = m_hat / denom
    
    # 返回: -lr * step
    return -lr * step_val

def _calculate_adamw_step_diff(g_t: torch.Tensor, 
                               g_t_quant: torch.Tensor, 
                               optimizer: torch.optim.Optimizer, 
                               param: torch.nn.Parameter) -> torch.Tensor:
    """
    [中间辅助函数] 从优化器提取状态，计算量化前后参数更新步长的差值。

    Args:
        g_t: 全精度梯度
        g_t_quant: 量化梯度
        optimizer: AdamW优化器
        param: 对应的参数
        
    Returns: 
        diff_w = Step(g_t_quant) - Step(g_t)
    """

    # 输入验证
    assert g_t.shape == g_t_quant.shape, f"Gradient shape mismatch: {g_t.shape} vs {g_t_quant.shape}"

    # 1. 获取优化器超参数
    # 注意: 如果有多个param_group,需要找到param所属的group
    
    group = None
    for g in optimizer.param_groups:
        # 必须遍历并使用 'is' 判断对象身份，不能使用 'in'
        for p in g['params']:
            if p is param:
                group = g
                break
        if group is not None:
            break
    
    if group is None:
        # 如果找不到（极少见），默认使用第一个组，或者抛出异常
        print("Warning: Param not found in any group, using group[0]")
        # group = optimizer.param_groups[0]
    
    beta1, beta2 = group['betas']
    eps = group['eps']
    lr = group['lr']
    
    # 2. 获取历史状态 (m_{t-1}, v_{t-1})
    state = optimizer.state[param]
    
    # 如果是第一步，状态可能为空，初始化为0
    if len(state) == 0:
        m_prev = torch.zeros_like(g_t)
        v_prev = torch.zeros_like(g_t)
        step = 1
    else:
        m_prev = state['exp_avg'].clone()
        v_prev = state['exp_avg_sq'].clone()
        step = state['step'] + 1 # 预测当前这一步 (t)
        
    # 3. 分别计算两种情况下的更新量
    # 调用独立的数学计算函数
    delta_w_clean = _simulate_adamw_update_step(
        g_t, m_prev, v_prev, step, lr, beta1, beta2, eps
    )
    
    delta_w_quant = _simulate_adamw_update_step(
        g_t_quant, m_prev, v_prev, step, lr, beta1, beta2, eps
    )
    
    # 4. 返回差值
    return delta_w_quant - delta_w_clean

def calculate_adamw_taylor_1st_g_t_norm(g_t: torch.Tensor, 
                                   g_t_quant: torch.Tensor, 
                                   optimizer: torch.optim.Optimizer, 
                                   param: torch.nn.Parameter,
                                   num_params: int) -> float:
    """
    计算 AdamW 下的一阶泰勒展开项（使用 g_t 近似 g_{t+1}）。
    数学形式: |L(θ̃_{t+1}) - L(θ_{t+1})| ≈ |g_t^T * (ΔW(g_quant) - ΔW(g))|

    Args:
        g_t: 全精度梯度
        g_t_quant: 量化梯度
        optimizer: AdamW优化器
        param: 对应的参数
        num_params: 参数数量(用于归一化)
        
    Returns:
        归一化后的泰勒一阶项
    """
    # 1. 计算参数更新的差值
    diff_w = _calculate_adamw_step_diff(g_t, g_t_quant, optimizer, param)
    
    # 2. 投影到 g_t 上
    taylor_val = torch.sum(g_t * diff_w).item()
    
    return taylor_val / num_params


def calculate_adamw_taylor_1st_g_t_plus_1_norm(g_t: torch.Tensor,
                                          g_t_quant: torch.Tensor,
                                          g_t_plus_1: torch.Tensor, 
                                          optimizer: torch.optim.Optimizer, 
                                          param: torch.nn.Parameter,
                                          num_params: int) -> float:
    """
    计算 AdamW 下的精确一阶泰勒展开项（使用真实的 g_{t+1}）。
    数学形式: |L(θ̃_{t+1}) - L(θ_{t+1})| ≈ |g_{t+1}^T * (ΔW(g_quant) - ΔW(g))|
    
    Args:
        g_t: 全精度梯度 (用于计算diff_w)
        g_t_quant: 量化梯度 (用于计算diff_w)
        g_t_plus_1: 更新后位置的梯度 (用于泰勒展开)
        optimizer: AdamW优化器
        param: 对应的参数
        num_params: 参数数量
        
    Returns:
        归一化后的泰勒一阶项绝对值
    """
    # 1. 计算参数更新的差值 (依然基于 t 时刻的梯度产生)
    diff_w = _calculate_adamw_step_diff(g_t, g_t_quant, optimizer, param)
    
    # 2. 投影到 g_{t+1} 上
    taylor_val = torch.sum(g_t_plus_1 * diff_w).item()
    
    return taylor_val / num_params