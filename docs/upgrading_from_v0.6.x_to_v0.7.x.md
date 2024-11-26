# Upgrading from v0.6.x to v0.7.x

## New Features

### Custom Pipeline configs as first-class, user-visible files

While we'll still ship with some default pipeline configs, users are now expected and encouraged to consider these as a starting point and edit the files for their own needs. This brings a lot more power and flexibility for the advanced user, and cuts down on the awkwardness of trying to wire an ever-expanding number of adjustable values through CLI flags.

### Custom Blocks and Teacher Models via BlockRegistry and PromptRegistry

Advanced users are now able to supply custom Pipeline `Block` implementations by registering new blocks with the `BlockRegistry`. It's also possible to register new chat templates for custom teacher models using the new `PromptRegistry`.

TODO: Provide an example of how to do this, preferably by linking to a script invoked by a test case that does this.

### New Top-Level API for only running Pipelines

TODO: Describe how users will directly run a `Pipeline.generate` now instead of using `generate_data` which does the entire taxonomy pre- and post-processing.

### LLMLogProbBlock

TODO: Describe what this new block does.

### LLMMessagesBlock

TODO: Describe what this new block does.

## Breaking Changes

### Pipeline configs and Prompt templates switched to Jinja

All of our [Pipeline config yamls](src/instructlab/sdg/pipelines) and [prompt template files](src/instructlab/sdg/configs) have moved to [Jinja templates](https://pypi.org/project/Jinja2/) instead of Python string `format()` calls. This brings more expressiveness into our templating language - especially for prompt templates - but does mean any variable substitutions need to be updated from single brackets to double brackets - ie `{document}` becomes `{{document}}`. This only impacts you if you were using custom pipeline config yaml files or custom prompt templates in your config blocks.

### ImportBlock removed from Pipeline blocks

Any users that were specifying custom pipeline configs (instead of using the default `full` or `simple` shipped by us) and also using the `ImportBlock` will now need to rewrite their pipelines to no longer use that block. We do not anticipate that anyone was actually using this block, but please reach out if you were so we can capture your needs in a future release.

### Deprecation of `num_instructions_to_generate`, `max-num-tokens` from `generate_data`

TODO: Not technically deprecated yet, but will need to be as users are expected to edit their pipeline yamls directly now to do things like this. We'll want to log an error message if these get passed in with a warning that they are now ignored and to edit your pipeline yaml.
