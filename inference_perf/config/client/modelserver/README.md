# Model Server Configuration

Configuration for the `server:` block, which defines how the benchmark connects
to the model serving backend: the server flavor, the endpoint, the model to
target, and the credentials used to reach it.

This page is the field-level reference. For the high-level config overview and
full end-to-end examples, see [docs/config.md](../../../../docs/config.md).

Schema: [`config.py`](./config.py).

## Server types

`server.type` selects the backend flavor the client speaks to.

| `type` | Backend |
| --- | --- |
| `vllm` (default) | vLLM OpenAI-compatible server |
| `sglang` | SGLang server |
| `tgi` | Text Generation Inference server |
| `mock` | In-process mock server for testing without a real backend |

### When to use

- **`vllm`**: Default. Use for benchmarking a vLLM deployment over its
  OpenAI-compatible endpoint.
- **`sglang`**: Use when driving an SGLang server.
- **`tgi`**: Use when driving a Hugging Face Text Generation Inference server.
- **`mock`**: Use for local development and CI, when you want to exercise the
  load generator and reporting paths without standing up a real backend.

## Top-level `server` fields

<!-- FIELDS: ModelServerClientConfig -->

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `type` | enum | `vllm` | Backend flavor (vllm, sglang, tgi, or mock). |
| `model_name` | str | `null` | Model identifier the server should serve (for example, a Hugging Face repo ID). |
| `base_url` | str | (required) | Server endpoint, including scheme, host, and port. |
| `ignore_eos` | bool | `true` | Whether to ignore End-of-Sequence tokens so generation runs to the requested length. |
| `api_key` | str | `null` | API key for authenticated endpoints. |
| `cert_path` | str | `null` | Path to a client TLS certificate for mutual-TLS endpoints. |
| `key_path` | str | `null` | Path to the private key paired with cert_path. |

<!-- /FIELDS -->

`base_url` is the only required field. `cert_path` and `key_path` are used
together to present a client certificate to endpoints that require mutual TLS.

## Example

```yaml
server:
  type: vllm
  model_name: "HuggingFaceTB/SmolLM2-135M-Instruct"
  base_url: "http://0.0.0.0:8000"
  ignore_eos: true
  api_key: ""
```
