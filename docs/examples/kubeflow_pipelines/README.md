# InstructLab SDG and Kubeflow Pipelines

## Compiling SDG Pipelines to Kubeflow YAML

Run `scripts/kfp_compile.py` to from the root of this repository to
compile some example SDG pipelines to Kubeflow YAML files.

```
python scripts/kfp_compile.py
```

That outputs one or more pipeline yamls. The `e2e_demo_pipeline.yaml` is
tested to work with Kubeflow running in a local minikube, and may work
in other Kubeflow environments as well. It doesn't do any real data
generation, and just mocks out the LLM responses for the sake of
demoing.

## Running the pipeline

Open your Kubeflow Pipelines UI and create a new pipeline, uploading
the `e2e_demo_pipeline.yaml` file as the source code of that pipeline.

Then create a run of this new pipeline, using the following pipeline
parameters:
- pipeline = /testdata/mock_pipelines
- taxonomy_repo = https://github.com/RedHatOfficial/rhelai-sample-taxonomy
- teacher_model_path = /testdata/models/instructlab/granite-7b-lab
