from typing import List, Optional

import torch
import torch.distributed
from loguru import logger
from transformers import AutoModelForCausalLM, AutoTokenizer

from lorax_server.models.causal_lm import CausalLM

FIM_PREFIX = "<fim-prefix>"
FIM_MIDDLE = "<fim-middle>"
FIM_SUFFIX = "<fim-suffix>"
FIM_PAD = "<fim-pad>"
EOD = "<|endoftext|>"


class SantaCoder(CausalLM):
    def __init__(
        self,
        model_id: str,
        revision: Optional[str] = None,
        quantize: Optional[str] = None,
        compile: bool = False,
        dtype: Optional[torch.dtype] = None,
        trust_remote_code: bool = False,
    ):
        if compile:
            logger.info(f"Model {model_id} does not support CUDA graph compilation. Skipping compilation.")

        if torch.cuda.is_available():
            device = torch.device("cuda")
            dtype = torch.float16 if dtype is None else dtype
        else:
            if quantize:
                raise ValueError("quantization is not available on CPU")

            device = torch.device("cpu")
            dtype = torch.float32

        tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            revision=revision,
            padding_side="left",
            truncation_side="left",
            trust_remote_code=trust_remote_code,
        )
        tokenizer.add_special_tokens(
            {
                "additional_special_tokens": [
                    EOD,
                    FIM_PREFIX,
                    FIM_MIDDLE,
                    FIM_SUFFIX,
                    FIM_PAD,
                ],
                "pad_token": EOD,
            }
        )
        with device:
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                revision=revision,
                torch_dtype=dtype,
                load_in_8bit=quantize == "bitsandbytes",
                trust_remote_code=trust_remote_code,
            )

        super(CausalLM, self).__init__(
            model_id=model_id,
            model=model,
            tokenizer=tokenizer,
            requires_padding=True,
            dtype=dtype,
            device=device,
            trust_remote_code=trust_remote_code,
        )

    def decode(self, generated_ids: List[int]) -> str:
        # Do not skip special tokens as they are used for custom parsing rules of the generated text
        return self.tokenizer.decode(generated_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
