# SPDX-License-Identifier: Apache-2.0
# Standard
from importlib import resources
import os.path

# Third Party
from datasets import Dataset
import yaml

# Local
from . import filterblock, importblock, llmblock, utilblocks
from .logger_config import setup_logger

logger = setup_logger(__name__)


class PipelineContext:
    def __init__(
        self, client, model_family, model_id, num_instructions_to_generate
    ) -> None:
        self.client = client
        self.model_family = model_family
        self.model_id = model_id
        self.num_instructions_to_generate = num_instructions_to_generate
        # FIXME: base this on the available number of CPUs
        self.num_procs = 8


class Pipeline:
    def __init__(self, ctx, config_path, chained_blocks: list) -> None:
        """
        Initialize the Pipeline class with a configuration dictionary.
        config_dict: the run config py or yaml loaded into a dictionary
        """
        # ctx is a PipelineContext object that supplies context configuration to every block
        self.ctx = ctx
        # config_path is the path of the pipeline config file used to create this pipeline
        self.config_path = config_path
        # pipeline config is the run configuration that consists of the pipeline steps
        self.chained_blocks = chained_blocks

    @classmethod
    def from_file(cls, ctx, pipeline_yaml):
        if not os.path.isabs(pipeline_yaml):
            pipeline_yaml = os.path.join(resources.files(__package__), pipeline_yaml)
        return cls(ctx, pipeline_yaml, _parse_pipeline_config_file(pipeline_yaml))

    def _drop_duplicates(self, dataset, cols):
        """
        Drop duplicates from the dataset based on the columns provided.
        """
        df = dataset.to_pandas()
        df.drop_duplicates(subset=cols, inplace=True)
        return Dataset.from_pandas(df)

    def generate(self, dataset) -> Dataset:
        """
        Generate the dataset by running the pipeline steps.
        dataset: the input dataset
        """
        for block_prop in self.chained_blocks:
            block_name = block_prop["name"]
            block_type = _lookup_block_type(block_prop["type"])
            block_config = block_prop["config"]
            drop_columns = block_prop.get("drop_columns", [])
            gen_kwargs = block_prop.get("gen_kwargs", {})
            drop_duplicates_cols = block_prop.get("drop_duplicates", False)
            block = block_type(self.ctx, self, block_name, **block_config)

            logger.info("Running block: %s", block_name)
            logger.info(dataset)

            dataset = block.generate(dataset, **gen_kwargs)

            drop_columns_in_ds = [e for e in drop_columns if e in dataset.column_names]
            if drop_columns:
                dataset = dataset.remove_columns(drop_columns_in_ds)

            if drop_duplicates_cols:
                dataset = self._drop_duplicates(dataset, cols=drop_duplicates_cols)

        return dataset


_block_types = {
    "CombineColumnsBlock": utilblocks.CombineColumnsBlock,
    "ConditionalLLMBlock": llmblock.ConditionalLLMBlock,
    "FilterByValueBlock": filterblock.FilterByValueBlock,
    "ImportBlock": importblock.ImportBlock,
    "LLMBlock": llmblock.LLMBlock,
    "SamplePopulatorBlock": utilblocks.SamplePopulatorBlock,
    "SelectorBlock": utilblocks.SelectorBlock,
}


def _lookup_block_type(block_type):
    if not block_type in _block_types:
        raise PipelineConfigParserError("Unknown block type {block_type}")
    return _block_types[block_type]


_PIPELINE_CONFIG_PARSER_MAJOR = 1
_PIPELINE_CONFIG_PARSER_MINOR = 0


class PipelineConfigParserError(Exception):
    """An exception raised while parsing a pipline config file."""


def _parse_pipeline_config_file(pipeline_yaml):
    with open(pipeline_yaml, "r", encoding="utf-8") as pipeline_file:
        content = yaml.safe_load(pipeline_file)

    version = content["version"]
    major, minor = map(int, version.split("."))

    if major > _PIPELINE_CONFIG_PARSER_MAJOR:
        raise PipelineConfigParserError(
            "The pipeline config file format is from a future major version."
        )
    if major <= _PIPELINE_CONFIG_PARSER_MAJOR and minor > _PIPELINE_CONFIG_PARSER_MINOR:
        logger.warning(
            "The pipeline config file may have new features that will be ignored."
        )

    if not "blocks" in content:
        raise PipelineConfigParserError(
            "The pipeline config file contains no 'blocks' section"
        )

    return content["blocks"]


SIMPLE_PIPELINES_PACKAGE = "instructlab.sdg.pipelines.simple"
FULL_PIPELINES_PACKAGE = "instructlab.sdg.pipelines.full"
