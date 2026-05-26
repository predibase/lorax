# HTTP Status Codes

The LoRAX API uses standard HTTP status codes to indicate the success or failure of requests. Below is a list of all status codes returned by LoRAX and what they mean.

## Success

| Code | Description |
|------|-------------|
| 200 OK | The request was successful. The response body contains the generated text, embeddings, or classification results. |

## Client Errors

| Code | Description |
|------|-------------|
| 400 Bad Request | The request contains invalid input. For example, a non-string or non-array value was provided where a string or array was expected. |
| 404 Not Found | The requested resource could not be found. This may occur if no fast tokenizer is available for the requested model on the `/tokenize` endpoint. |
| 422 Unprocessable Entity | Input validation failed. This can happen when required parameters are missing, parameters have invalid values, the input is empty, or chat template rendering fails. |
| 429 Too Many Requests | The model is currently overloaded. The server is processing too many requests and cannot accept more at this time. Retry the request after a short delay. |

## Server Errors

| Code | Description |
|------|-------------|
| 424 Failed Dependency | An upstream error occurred during generation. The request was accepted but the inference engine failed to produce a result. |
| 500 Internal Server Error | The server encountered an unexpected condition. This may indicate incomplete generation (the stream ended prematurely), embedding failure, or classification failure. |
| 503 Service Unavailable | The service is temporarily unavailable. The health check failed, indicating the server is not ready to accept requests. |

## Endpoint-Specific Status Codes

| Endpoint | Method | Status Codes |
|----------|--------|--------------|
| `/generate` | POST | 200, 422, 424, 429, 500 |
| `/generate_stream` | POST | 200, 422, 424, 429, 500 |
| `/v1/chat/completions` | POST | 200, 422, 424, 429, 500 |
| `/v1/completions` | POST | 200, 422, 424, 429, 500 |
| `/v1/embeddings` | POST | 200, 400, 500 |
| `/embed` | POST | 200, 500 |
| `/classify` | POST | 200, 500 |
| `/tokenize` | POST | 200, 404 |
| `/health` | GET | 200, 503 |
| `/info` | GET | 200 |
| `/metrics` | GET | 200 |

## Error Response Format

All error responses follow this format:

```json
{
  "error": "A human-readable error message",
  "error_type": "machine_readable_error_type"
}