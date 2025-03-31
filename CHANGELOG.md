## v0.7.3

### Fixes

* Update to a newer version of Docling, fixing additional cases where a user might hit a "list index out of range" error when converting documents.

## v0.7.2

### Fixes

* When chunking knowledge documents, PDF or Markdown documents containing a table would often result in a "list index out of range". The cases for that error resulting from the chunking of table content are now fixed. We've also had users report other cases where a "list index out of range" error can show up in the version of Docling we rely on, and those specific cases won't be fixed until we upgrade the Docling version.

## v0.7.1

### Fixes

* When mixing datasets, we were not always properly plumbing through the user's expected system prompt into the samples of the mixed dataset. And, specifically for the new `mix_datasets` API added in v0.7.0, we were never setting the system prompt. This adds that as a parameter to that API and ensures we use it when creating our mixed datasets.

## v0.7.0

### Features

#### Custom Blocks and Teacher Models via BlockRegistry and PromptRegistry

Advanced users are now able to supply custom Pipeline `Block` implementations by registering new blocks with the `BlockRegistry`. It's also possible to register new chat templates for custom teacher models using the new `PromptRegistry`.

See the `tests/testdata/custom_block.py` and `tests/testdata/custom_block_pipeline.yaml` files in this repository for an example of how to create custom blocks and use them from your own pipeline config yamls.

See the `tests/testdata/custom_prompt.py` file in this repository for an example how to register custom chat templates used when formatting prompts.

### New Blocks - IterBlock and LLMMessagesBlock

We have two new Block types available for pipelines in this release - `IterBlock` and `LLMMessagesBlock`. `IterBlock` allows you to execute another `Block` multiple times, based on a configured number of iterations. `LLMMessagesBlock` is like `LLMBlock` but uses the newer chat/completions API of OpenAI-compatible servers instead of the legacy completions API.

### Consolidated PDF and Markdown ingestion and chunking implementations

Instead of sending PDF input documents through Docling and using something custom for Markdown, we now send both types of documents through Docling and have consolidated the chunking implementation across both document types. This may result in different chunks being generated for markdown content compared to previous releases.

### Added a new `instructlab.sdg.mix_datasets` Python API

We've added a new Python API for advanced users that need to re-mix our generated outputs, for example to weight one taxonomy leaf node over others in the output or to have more than our default of 30 skill samples per leaf node in the final mixed output. See the example at `docs/examples/mix_datasets/` for some example Python code and Recipe yaml files to accomplish this.

### Breaking Changes

#### Pipeline configs and Prompt templates switched to Jinja

All of our [Pipeline config yamls](src/instructlab/sdg/pipelines) and [prompt template files](src/instructlab/sdg/configs) have moved to [Jinja templates](https://pypi.org/project/Jinja2/) instead of Python string `format()` calls. This brings more expressiveness into our templating language - especially for prompt templates - but does mean any variable substitutions need to be updated from single brackets to double brackets - ie `{document}` becomes `{{document}}`. This only impacts you if you were using custom pipeline config yaml files or custom prompt templates in your config blocks.

#### ImportBlock removed from Pipeline blocks

Any users that were specifying custom pipeline configs (instead of using the default `full` or `simple` shipped by us) and also using the `ImportBlock` will now need to rewrite their pipelines to no longer use that block. We do not anticipate that anyone was actually using this block, but please reach out if you were so we can capture your needs in a future release.

### Fixes

* The PyTorch dependency is removed, because SDG doesn't directly use PyTorch. The test suite still depends on `instructlab` core, which depends on PyTorch.
* The `batch_size` parameter is now respected every time we call an inference server from an `LLMBlock`. Previously, we were only batching the initial input but not accounting for some Blocks that may emit more output samples than input samples, meaning we would exceed our configured `batch_size` when actually making batching inference calls to vLLM, causing more memory to be consumed than expected as well as leading to scenarios where we were overloading inference servers in unexpected ways due to sending in batches with hundreds of completion requests instead of the configured size, which defaults to `8` on most hardware profiles.

## v0.6.3

### Fixes

* The max version constraint of PyTorch in our requirements file was raised so that we don't prevent SDG users from using it PyTorch 2.5.

## v0.6.2

### Fixes

* Fixed a bug in our version specification of `docling` and `docling_parse` dependencies that were causing new installs of InstructLab to pull in incompatible versions of these. We also fixed a similar bug in the `mypy` dependency, but that one only impacts developers of SDG as opposed to users of InstructLab.

## v0.6.1

### Fixes

* Fixed a bug where generating data from a taxonomy with 2 or more changed knowledge leaf nodes would fail with a message about a destination path `already exists and is not an empty directory`

## v0.6.0

### Features

* Small knowledge datasets will automatically get upsampled during final data mixing based on the length of any precomputed skills datasets used during data mixing. This avoids issues where very large precomputed skills datasets were swamping the comparatively minor number of knowledge samples, resulting in lower than optimal knowledge retention during multiphase training. If a large precomputed dataset isn't in use during mixing (which is how things operate by default), this change is a no-op.
* When chunking PDF documents, we'll now look for the docling models on-disk in `$XDG_DATA_HOME/instructlab/sdg/models` (as well as `$XDG_DATA_DIRS` with the same `instructlab/sdg/models` subdirectory). If they are not found on disk, they'll automatically be downloaded from HuggingFace.
* When chunking PDF documents with Docling, we'll automatically configure Docling to use `tesserocr` if a working implementation is found instead of relying on `easyocr`. We fallback to `easyocr` if Tesseract is not properly configured for use by `tesserocr`.

### Breaking Changes

* Teacher model tokenizers are loaded from the local teacher model on-disk and not downloaded automatically from HuggingFace. The typical workflows in use so far expect the teacher model to exist on-disk, and this enforces that at least its tokenizer exists.
