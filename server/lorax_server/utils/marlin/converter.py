import torch
from marlin import Layer as MarlinLayer

@torch.no_grad()
def unpack_4bit_to_32bit_signed(qweight, qzeros):
    # Unpack 4-bit values and interpret them as signed integers
    unpacked_weights = torch.zeros((qweight.shape[0]*8, qweight.shape[1]), dtype=torch.int8, device=qweight.device, requires_grad=False)
    unpacked_zeros = torch.zeros((qzeros.shape[0], qzeros.shape[1]*8), dtype=torch.int8, device=qzeros.device, requires_grad=False)

    for row in range(unpacked_weights.shape[0]):
        i = row % 8
        unpacked_weights[row, :] = (qweight[row // 8, :] >> (4 * i)) & 0xF

    for col in range(unpacked_zeros.shape[1]):
        i = col % 8
        unpacked_zeros[:, col] = (qzeros[:, col // 8] >> (4 * i)) & 0xF

    return unpacked_weights, unpacked_zeros + 1

@torch.no_grad()
def dequantize_weight(qweight, qzeros, scales):
    unpacked_qweight, unpacked_qzeros = unpack_4bit_to_32bit_signed(qweight, qzeros)
    group_size = unpacked_qweight.shape[0] // scales.shape[0]
    scales = scales.repeat_interleave(group_size, dim=0)
    unpacked_qzeros = unpacked_qzeros.repeat_interleave(group_size, dim=0)
    unpacked_qweight = (unpacked_qweight - unpacked_qzeros) * scales

    return unpacked_qweight.T

@torch.no_grad()
def convert_to_marlin(qweight, qzeros, scales, group_size, verbose=True):

    # Dequantize the weight.
    dequantized_weight = dequantize_weight(qweight, qzeros, scales).to(torch.float16)
    linear_module = torch.nn.Linear(
        in_features=dequantized_weight.shape[1],
        out_features=dequantized_weight.shape[0],
        bias=False,
        dtype=torch.float16,
        device="cuda")
    linear_module.weight.data.copy_(dequantized_weight)

    # Create new linear method and copy to model.
    new_module = MarlinLayer(
        infeatures=linear_module.in_features,
        outfeatures=linear_module.out_features,
        groupsize=group_size)
    new_module.pack(linear_module, scales=scales)

    return new_module