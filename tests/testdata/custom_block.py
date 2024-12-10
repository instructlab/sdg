# SPDX-License-Identifier: Apache-2.0

# Standard
import pathlib

# Third Party
from datasets import Dataset

# First Party
from instructlab.sdg import Block, BlockRegistry, Pipeline, PipelineContext


@BlockRegistry.register("EchoBlock")
class EchoBlock(Block):
    def generate(self, samples: Dataset):
        return samples


pipeline_context = PipelineContext(None, "mixtral", "my_model", 5)
pipeline_yaml = pathlib.Path(__file__).parent.joinpath("custom_block_pipeline.yaml")
pipeline = Pipeline.from_file(pipeline_context, pipeline_yaml)
input_ds = Dataset.from_list(
    [
        {
            "fruit": "apple",
            "color": "red",
        }
    ]
)
output_ds = pipeline.generate(input_ds)
assert len(output_ds) == 1
assert output_ds[0]["fruit"] == "apple"
assert output_ds[0]["color"] == "red"
