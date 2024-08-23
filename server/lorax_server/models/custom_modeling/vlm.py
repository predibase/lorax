def load_text_model(prefix, config, weights, name=None):
    if config.model_type == "llama":
        from lorax_server.models.custom_modeling.flash_llama_modeling import (
            FlashLlamaForCausalLM,
        )

        return FlashLlamaForCausalLM(prefix, config, weights)
    elif config.model_type == "mistral":
        from lorax_server.models.custom_modeling.flash_mistral_modeling import (
            FlashMistralForCausalLM,
        )

        return FlashMistralForCausalLM(prefix, config, weights, name=name)
    elif config.model_type == "gemma":
        from lorax_server.models.custom_modeling.flash_gemma_modeling import (
            FlashGemmaForCausalLM,
        )

        return FlashGemmaForCausalLM(prefix, config, weights, causal=False)
    elif config.model_type == "paligemma":
        from lorax_server.models.custom_modeling.flash_gemma_modeling import (
            FlashGemmaForCausalLM,
        )

        return FlashGemmaForCausalLM(prefix, config, weights)
    else:
        raise RuntimeError(f"Unsupported model type {config.model_type}")


def load_vision_model(prefix, config, weights):
    if config.model_type == "clip_vision_model":
        from lorax_server.models.custom_modeling.clip import (
            CLIPVisionTransformer,
        )

        return CLIPVisionTransformer(prefix=f"{prefix}.vision_model", config=config, weights=weights)
    if config.model_type == "siglip_vision_model":
        from lorax_server.models.custom_modeling.siglip import (
            SiglipVisionTransformer,
        )

        return SiglipVisionTransformer(prefix="vision_tower.vision_model", config=config, weights=weights)
    else:
        raise RuntimeError(f"Unsupported model type {config.model_type}")
