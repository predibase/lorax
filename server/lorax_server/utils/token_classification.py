from typing import List, Tuple

from transformers import PreTrainedTokenizerBase


def _get_tag(entity_name: str) -> Tuple[str, str]:
    if entity_name.startswith("B-"):
        bi = "B"
        tag = entity_name[2:]
    elif entity_name.startswith("I-"):
        bi = "I"
        tag = entity_name[2:]
    else:
        # It's not in B-, I- format
        # Default to I- for continuation.
        bi = "I"
        tag = entity_name
    return bi, tag


def format_ner_output(
    predicted_token_class: List[str], scores: List[float], input_ids: List[int], tokenizer: PreTrainedTokenizerBase
) -> List[dict]:
    tokens = tokenizer.convert_ids_to_tokens(input_ids.tolist())

    ner_results = []
    current_entity = None

    # skip first and last token
    tokens = tokens[1:-1]
    predicted_token_class = predicted_token_class[1:-1]
    scores = scores[1:-1]
    input_ids = input_ids[1:-1]

    for i, (token, token_class, score) in enumerate(zip(tokens, predicted_token_class, scores)):
        if token_class != "O":
            bi, tag = _get_tag(token_class)
            if bi == "B" or (current_entity and tag != current_entity["entity"]):
                if current_entity:
                    ner_results.append(current_entity)
                current_entity = {
                    "entity": tag,
                    "score": score,
                    "index": i,
                    "word": token,
                    "start": len(tokenizer.decode(input_ids[:i])),
                    "end": len(tokenizer.decode(input_ids[: i + 1])),
                }
            elif bi == "I" and current_entity:
                current_entity["word"] += token.replace("##", "")
                current_entity["end"] = len(tokenizer.decode(input_ids[: i + 1]))
        else:
            if current_entity:
                ner_results.append(current_entity)
                current_entity = None

    if current_entity:
        ner_results.append(current_entity)

    return ner_results
