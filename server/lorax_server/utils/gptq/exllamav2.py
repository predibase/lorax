# Adapted from turboderp exllama: https://github.com/turboderp/exllamav2

from logging import getLogger

import torch
import torch.nn as nn

logger = getLogger(__name__)

try:
    from exllamav2_kernels import gemm_half_q_half, make_q_matrix
except ImportError:
    logger.error("exllamav2_kernels not installed.")
    raise

# Dummy tensor to pass instead of g_idx since there is no way to pass "None" to a C++ extension
none_tensor = torch.empty((1, 1), device="meta")


def ext_gemm_half_q_half(x, q_handle, q4_width, force_cuda):
    """Matrix multiplication, returns x @ q4"""
    output_shape = x.shape[:-1] + (q4_width,)
    x = x.view(-1, x.shape[-1])
    output = torch.empty((x.shape[0], q4_width), dtype=torch.half, device=x.device)
    gemm_half_q_half(x, q_handle, output, force_cuda)
    return output.view(output_shape)


def make_group_map(q_groups, num_qrows):
    # Convert q_groups to a list
    gr = q_groups.tolist()

    # Initialize an empty list for group_map
    group_map = []

    # Calculate the number of groups
    num_groups = len(gr) // 2

    # Loop through each group
    for i in range(num_groups):
        # Get the number of bits for the current group
        bits = gr[i * 2]

        # Calculate the number of qrows for the current group
        if i < num_groups - 1:
            qrows = gr[i * 2 + 3] - gr[i * 2 + 1]
        else:
            qrows = num_qrows - gr[i * 2 + 1]

        # Calculate the number of rows for the current group
        rows = qrows * 32 // bits

        # Loop through each row in the current group
        for j in range(rows):
            # Add the current group index to the group_map
            group_map.append(i)

            # Add the remaining rows to the group_map
            remaining_rows = rows - j
            group_map.append(remaining_rows)

    # Convert the group_map to a torch tensor and return it
    group_map_tensor = torch.tensor(group_map, dtype=torch.short, device=q_groups.device)
    return group_map_tensor


def ext_make_q_matrix(w: dict, temp_dq):
    """
    Create Q matrix
    """
    # Check if 'q_weight' is in the dictionary
    q_weight_exists = "q_weight" in w

    if q_weight_exists:
        # Adjust 'q_scale_max' value
        w["q_scale_max"] = w["q_scale_max"] / 256

        # Convert 'q_perm' and 'q_invperm' to short type
        w["q_perm"] = w["q_perm"].short()
        w["q_invperm"] = w["q_invperm"].short()

        # Check if 'q_group_map' is not in the dictionary
        q_group_map_not_exists = "q_group_map" not in w

        if q_group_map_not_exists:
            # Create 'q_group_map'
            w["q_group_map"] = make_group_map(w["q_groups"], w["q_weight"].shape[0])

        # Create Q matrix for EXL2
        q_matrix = make_q_matrix(
            w["q_weight"],
            w["q_perm"],
            w["q_invperm"],
            w["q_scale"],
            w["q_scale_max"],
            w["q_groups"],
            w["q_group_map"],
            none_tensor,
            none_tensor,
            none_tensor,
            temp_dq,
        )
        return q_matrix

    # Check if 'qweight' is in the dictionary
    qweight_exists = "qweight" in w

    if qweight_exists:
        # Check if 'scales' dtype is float
        scales_dtype_is_float = w["scales"].dtype == torch.float

        if scales_dtype_is_float:
            # Convert 'scales' to half type
            w["scales"] = w["scales"].half()

        # Check if 'g_idx' exists and is not all zeros
        g_idx_exists_and_not_all_zeros = w.get("g_idx", None) is not None and not (w["g_idx"] == 0).all().item()

        if g_idx_exists_and_not_all_zeros:
            # Create 'q_perm' and 'q_invperm'
            w["q_perm"] = torch.empty(
                (w["qweight"].shape[0] * 8,),
                dtype=torch.short,
                device=w["qweight"].device,
            )
            w["q_invperm"] = torch.empty_like(w["q_perm"])

            # Create Q matrix for GPTQ with 'g_idx'
            q_matrix = make_q_matrix(
                w["qweight"],
                w["q_perm"],
                w["q_invperm"],
                none_tensor,
                none_tensor,
                none_tensor,
                none_tensor,
                w["qzeros"],
                w["scales"],
                w["g_idx"].cpu(),
                temp_dq,
            )
            return q_matrix

        else:
            # Create Q matrix for GPTQ without 'g_idx'
            q_matrix = make_q_matrix(
                w["qweight"],
                none_tensor,
                none_tensor,
                none_tensor,
                none_tensor,
                none_tensor,
                none_tensor,
                w["qzeros"],
                w["scales"],
                none_tensor,
                temp_dq,
            )
            return q_matrix


DEVICE = None
FIXED_BYTES = 0
LAYERS = []


def set_device(device):
    global DEVICE
    DEVICE = device


def create_exllama_buffers():
    global FIXED_BYTES, LAYERS, DEVICE
    temp_dq = ExLlamaV2DeviceTensors(DEVICE, FIXED_BYTES)

    for layer in LAYERS:
        layer.post_init(temp_dq)


class QuantLinear(nn.Module):
    QUANT_TYPE = "exllamav2"

    """Linear layer implementation with per-group 4-bit quantization of the weights"""

    def __init__(self, qweight, qzeros, scales, g_idx, bias, bits, groupsize):
        super().__init__()
        if bits != 4:
            raise ValueError(
                f"Exllamav2 kernel supports only bits=4, requested bits={bits}. Something is wrong in the model initialization."
            )
        self.q_handle = None
        self.q_tensors = None
        self.bits = bits
        self.maxq = 2**self.bits - 1
        self.infeatures = qweight.shape[0] // self.bits * 32
        self.outfeatures = qweight.shape[1] + qweight.shape[1] % 32

        self.device = qweight.device
        self.qweight = qweight
        self.qzeros = qzeros
        self.scales = scales
        self.g_idx = g_idx
        self.bias = bias if bias is not None else None
        self.group_size = groupsize

        global FIXED_BYTES, LAYERS
        FIXED_BYTES = max(FIXED_BYTES, self.scratch_spacing())
        LAYERS.append(self)

    def post_init(self, temp_dq):
        assert self.qweight.device.type == "cuda"
        assert self.qweight.device.index is not None
        self.q_tensors = {
            "qweight": self.qweight,
            "qzeros": self.qzeros,
            "scales": self.scales,
            "g_idx": self.g_idx,
        }
        temp_dq = temp_dq.get_scratch_slice(self.temp_dq_size())
        self.q_handle = ext_make_q_matrix(self.q_tensors, temp_dq)

    def forward(self, x, force_cuda=False):
        output = ext_gemm_half_q_half(x, self.q_handle, self.outfeatures, force_cuda)

        if self.bias is not None:
            output.add_(self.bias)
        return output

    def temp_dq_size(self):
        return self.infeatures * self.outfeatures * 2 + 128

    def temp_fwd_size(self, max_input_len, max_batch_size):
        return self.outfeatures * max_input_len * max_batch_size * 4 + 128

    def scratch_spacing(self, max_input_len=8192, max_batch_size=32):
        return self.temp_dq_size() + self.temp_fwd_size(max_input_len, max_batch_size)

    @property
    def weight(self) -> torch.Tensor:
        return self.qweight


class ExLlamaV2DeviceTensors:
    device_idx: int
    scratch_bytes: int
    scratch_idx: int
    scratch: torch.tensor = None

    def __init__(self, device, scratch_bytes):
        self.device = device
        self.scratch_bytes = scratch_bytes

    def prepare(self):
        self.scratch = torch.empty((self.scratch_bytes // 2,), dtype=torch.half, device=self.device)

    def get_scratch_slice(self, size_bytes):
        if self.scratch is None:
            self.prepare()

        size_bytes = ((size_bytes + 127) // 128) * 128
        size_half = size_bytes // 2
        scratch_slice = self.scratch.narrow(0, 0, size_half)
        return scratch_slice
