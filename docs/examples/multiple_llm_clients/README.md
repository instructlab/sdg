# Multiple LLM clients in a single Pipeline

For advanced use-cases, PipelineContext accepts a `clients` dictionary of string to OpenAI client mappings. The special string of "default" sets the OpenAI client used for LLMBlocks by default, but individual LLMBlocks can override the client used by the `client` parameter in their yaml config.

See `pipeline.yaml` in this directory for an example of a Pipeline that uses different clients per `LLMBlock`.
