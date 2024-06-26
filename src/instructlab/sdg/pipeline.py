# SPDX-License-Identifier: Apache-2.0
# Third Party
from datasets import Dataset

# Local
from .iterblock import IterBlock
from .logger_config import setup_logger

logger = setup_logger(__name__)


class Pipeline:
    def __init__(self, chained_blocks: list) -> None:
        """
        Initialize the Pipeline class with a configuration dictionary.
        config_dict: the run config py or yaml loaded into a dictionary
        """
        # pipeline config is the run configuration that consists of the pipeline steps
        self.chained_blocks = chained_blocks

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
            block_type = block_prop["block_type"]
            block_config = block_prop["block_config"]
            drop_columns = block_prop.get("drop_columns", [])
            gen_kwargs = block_prop.get("gen_kwargs", {})
            drop_duplicates_cols = block_prop.get("drop_duplicates", False)
            block = block_type(**block_config)

            if block_type == IterBlock:
                block_kwargs = block_config.pop("block_kwargs")
                block = block_type(**block_config, block_kwargs=block_kwargs)
            else:
                block = block_type(**block_config)

            logger.info("Running block: %s", block_config["block_name"])
            logger.info(dataset)

            dataset = block.generate(dataset, **gen_kwargs)

            drop_columns_in_ds = [e for e in drop_columns if e in dataset.column_names]
            if drop_columns:
                dataset = dataset.remove_columns(drop_columns_in_ds)

            if drop_duplicates_cols:
                dataset = self._drop_duplicates(dataset, cols=drop_duplicates_cols)

        return dataset
