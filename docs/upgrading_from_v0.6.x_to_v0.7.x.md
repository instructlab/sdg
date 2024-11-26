# Upgrading from v0.6.x to v0.7.x

## New Features

### Custom Blocks and Teacher Models via BlockRegistry and PromptRegistry

Advanced users are now able to supply custom Pipeline `Block` implementations by registering new blocks with the `BlockRegistry`. It's also possible to register new chat templates for custom teacher models using the new `PromptRegistry`.

See the `tests/testdata/custom_block.py` and `tests/testdata/custom_block_pipeline.yaml` files in this repository for an example of how to create custom blocks and use them from your own pipeline config yamls.

## Breaking Changes

### Pipeline configs and Prompt templates switched to Jinja

All of our [Pipeline config yamls](src/instructlab/sdg/pipelines) and [prompt template files](src/instructlab/sdg/configs) have moved to [Jinja templates](https://pypi.org/project/Jinja2/) instead of Python string `format()` calls. This brings more expressiveness into our templating language - especially for prompt templates - but does mean any variable substitutions need to be updated from single brackets to double brackets - ie `{document}` becomes `{{document}}`. This only impacts you if you were using custom pipeline config yaml files or custom prompt templates in your config blocks.

### ImportBlock removed from Pipeline blocks

Any users that were specifying custom pipeline configs (instead of using the default `full` or `simple` shipped by us) and also using the `ImportBlock` will now need to rewrite their pipelines to no longer use that block. We do not anticipate that anyone was actually using this block, but please reach out if you were so we can capture your needs in a future release.
