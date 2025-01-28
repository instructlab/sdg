# Multiple LLMs in a single Pipeline

To create a `Pipeline` that uses multiple different LLMs for inference, each `LLMBlock` in the `Pipeline` yaml should specify a `model_family` (to indicate which set of prompts to use) and a `model_id` (corresponding to the id of this model in the deployed inference server). For more control over the prompting, a direct `model_prompt` may be used in place of `model_family` to specify the exact prompt to use as opposed to choosing a generic one for that model family.

See `pipeline.yaml` in this directory for an example of a Pipeline that calls into 3 separate LLMs.
