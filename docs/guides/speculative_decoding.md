# Speculative Decoding

## Process

### Draft

### Verification

Once the speculative logits have been generated, a separate *verification step* is performed whereby the most likely next `S` tokens
are passed through the model again (as part of the normal decoding process) to check for correctness. If any prefix of the `S` tokens
are deemed *correct*, then they can be appended to the response directly. The remaining incorrect speculative tokens are discarded.

Note that this process adds some compute overhead to the normal decoding step. As such, it will only confer benefits when:

1. The decoding step is *memory bound* (generally true for most LLMs on modern GPUs).
2. The speculation process is able to consistently predict future tokens correctly.

## Options

### Medusa

See the [Medusa](../models/adapters/medusa.md) guide for details on how this method works and how to use it.

### Prompt Lookup Decoding