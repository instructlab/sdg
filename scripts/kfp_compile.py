# SPDX-License-Identifier: Apache-2.0

# Third Party
import kfp

# First Party
from instructlab.sdg.kfp.pipelines import (
    e2e_pipeline,
    full_knowledge_pipeline,
)

def compile_pipeline(pipeline, yaml_path):
    kfp.compiler.Compiler().compile(pipeline, yaml_path)
    print(f"- compiled {yaml_path}")

print("Compiling Kubeflow Pipelines to YAML...")
compile_pipeline(e2e_pipeline, "e2e_demo_pipeline.yaml")
compile_pipeline(full_knowledge_pipeline, "full_knowledge.yaml")
