# Tokenizer & Utility Configuration

Configuration for the `tokenizer:` block, which selects the tokenizer used for
token counting, exact-length prompt construction, and any data generator that
needs to encode or decode text. The tokenizer drives input/output token metrics
and lets synthetic generators hit precise prompt lengths.

This page is the field-level reference. For the high-level config overview and
full end-to-end examples, see [docs/config.md](../../../docs/config.md).

Schema: [`config.py`](./config.py).

## When a tokenizer is required

For most runs the tokenizer is **inferred** from the model server: the `vllm`,
`sglang`, and `tgi` clients derive it from `server.model_name`, so an explicit
`tokenizer:` block is optional. Set the block to override that choice (a
different tokenizer than the served model) or to supply credentials for a gated
model.

Some data generators require a usable tokenizer (inferred or explicit) and raise
at startup if one is unavailable:

- `random`
- `synthetic`
- `shared_prefix`
- `multimodal`
- `conversation_replay`
- `cnn_dailymail`
- `shareGPT` (for both `completion` and `chat` APIs)

## `CustomTokenizerConfig` fields

<!-- FIELDS: CustomTokenizerConfig -->

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `pretrained_model_name_or_path` | str | `null` | HuggingFace model ID or local path the tokenizer is loaded from. Required to activate an explicit tokenizer; if unset, the tokenizer is inferred from the model server. |
| `trust_remote_code` | bool | `null` | Allow loading custom tokenizer code shipped with the model repo. Leave unset (treated as off) unless the model requires it. |
| `token` | str | `null` | HuggingFace access token for private or gated models. |

<!-- /FIELDS -->

## Example

```yaml
tokenizer:
  pretrained_model_name_or_path: HuggingFaceTB/SmolLM2-135M-Instruct
  trust_remote_code: true
  token: ""
```
