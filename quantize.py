import torch
import copy

def n_bit_quantize(emb, n_bits=4, per_row=True):
    """
    将任意维度的 tensor 做 n-bit 量化并返回 dequantized 结果
    :param emb:     输入浮点 tensor
    :param n_bits:  量化位数
    :param per_row: True=按行量化(2D+), False=全局量化
    :return: dequantized tensor (shape 不变, float32)
    """
    q_min = 0
    q_max = 2 ** n_bits - 1

    original_shape = emb.shape

    # 统一转成 2D 处理: (rows, -1)
    # 1D tensor -> (1, N)，之后 reshape 回去
    if emb.dim() == 1 or not per_row:
        flat = emb.reshape(1, -1)
    else:
        flat = emb.reshape(emb.shape[0], -1)  # (out_channels, ...)

    # 按行计算 min/max
    t_min = flat.min(dim=1, keepdim=True)[0]   # (rows, 1)
    t_max = flat.max(dim=1, keepdim=True)[0]   # (rows, 1)

    scale = (t_max - t_min) / (q_max - q_min)
    scale = torch.clamp(scale, min=1e-8)

    quantized = torch.round((flat - t_min) / scale)
    quantized = torch.clamp(quantized, q_min, q_max).to(torch.uint8)

    dequantized = quantized.float() * scale + t_min

    return dequantized.reshape(original_shape)


def quantize_state_dict(model_param, n_bits=4, per_row=True, skip_small=True, min_elements=64):
    """
    对 model.state_dict() 中所有浮点参数做 n-bit 量化
    :param model:        PyTorch 模型
    :param n_bits:       量化位数
    :param per_row:      是否按行量化
    :param skip_small:   是否跳过小 tensor (如 bias, bn 统计量)
    :param min_elements: 元素数量低于此值时跳过量化
    :return: 量化后的 state_dict (可直接 load 回模型)
    """
    quantized_sd = copy.deepcopy(model_param)

    skipped, quantized = [], []

    for name, param in model_param.items():
        # 只处理浮点类型
        if not param.is_floating_point():
            skipped.append((name, "non-float"))
            continue

        # 跳过元素数量过少的 tensor
        if skip_small and param.numel() < min_elements:
            skipped.append((name, f"too small ({param.numel()} elements)"))
            continue

        quantized_sd[name] = n_bit_quantize(param, n_bits=n_bits, per_row=per_row)
        quantized.append(name)

    # print(f"\n[quantize_state_dict] {n_bits}-bit | per_row={per_row}")
    # print(f"  Quantized : {len(quantized)} tensors")
    # print(f"  Skipped   : {len(skipped)} tensors")
    # for n, reason in skipped:
    #     print(f"    - {n:50s} ({reason})")

    return quantized_sd