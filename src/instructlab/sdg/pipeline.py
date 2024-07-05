# SPDX-License-Identifier: Apache-2.0
# Standard
from importlib import resources
import os.path

# Third Party
from datasets import Dataset
import yaml

# Local
from . import filterblock, llmblock, utilblocks
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
        self.sdg_base = resources.files(__package__)
        # FIXME: base this on the available number of CPUs
        self.num_procs = 8


class Pipeline:
    def __init__(self, ctx, chained_blocks: list) -> None:
        """
        Initialize the Pipeline class with a configuration dictionary.
        config_dict: the run config py or yaml loaded into a dictionary
        """
        # ctx is a PipelineContext object that supplies context configuration to every block
        self.ctx = ctx
        # pipeline config is the run configuration that consists of the pipeline steps
        self.chained_blocks = chained_blocks

    @classmethod
    def from_flows(cls, ctx, flows):
        block_configs = []
        for flow_path in flows:
            if not os.path.isabs(flow_path):
                flow_path = os.path.join(ctx.sdg_base, flow_path)
            block_configs.extend(parse_flow_config_file(flow_path))
        return cls(ctx, block_configs)

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
            block_type = _lookup_block_type(block_prop["block_type"])
            block_config = block_prop["block_config"]
            drop_columns = block_prop.get("drop_columns", [])
            gen_kwargs = block_prop.get("gen_kwargs", {})
            drop_duplicates_cols = block_prop.get("drop_duplicates", False)
            block = block_type(self.ctx, **block_config)

            logger.info("Running block: %s", block_config["block_name"])
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
    "LLMBlock": llmblock.LLMBlock,
    "SamplePopulatorBlock": utilblocks.SamplePopulatorBlock,
    "SelectorBlock": utilblocks.SelectorBlock,
}


def _lookup_block_type(block_type):
    if not block_type in _block_types:
        raise FlowParserError("Unknown block type {block_type}")
    return _block_types[block_type]


_FLOW_PARSER_MAJOR = 1
_FLOW_PARSER_MINOR = 0


class FlowParserError(Exception):
    """An exception raised while parsing a flow config file."""


def parse_flow_config_file(flow_path):
    with open(flow_path, "r", encoding="utf-8") as flow_file:
        content = yaml.safe_load(flow_file)

    version = content["version"]
    major, minor = map(int, version.split("."))

    if major > _FLOW_PARSER_MAJOR:
        raise FlowParserError(
            "The custom flow file format is from a future major version."
        )
    if major <= _FLOW_PARSER_MAJOR and minor > _FLOW_PARSER_MINOR:
        logger.warning(
            "The custom flow file may have new features that will be ignored."
        )

    if not "block_configs" in content:
        raise FlowParserError(
            "The custom flow file contains no 'block_configs' section"
        )

    return content["block_configs"]


MMLU_BENCH_FLOW = "flows/mmlu_bench.yaml"
SIMPLE_FREEFORM_SKILLS_FLOW = "flows/simple_freeform_skills.yaml"
SIMPLE_GROUNDED_SKILLS_FLOW = "flows/simple_grounded_skills.yaml"
SIMPLE_KNOWLEDGE_FLOW = "flows/simple_knowledge.yaml"
SYNTH_FREEFORM_SKILLS_FLOW = "flows/synth_freeform_skills.yaml"
SYNTH_GROUNDED_SKILLS_FLOW = "flows/synth_grounded_skills.yaml"
SYNTH_KNOWLEDGE_FLOW = "flows/synth_knowledge.yaml"
