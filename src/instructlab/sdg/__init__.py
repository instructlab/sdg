from .pipeline import Pipeline
from datasets import Dataset

class SDG:
    def __init__(self, pipelines: list[Pipeline]) -> None:
        self.pipelines = pipelines
          
    def generate(self, dataset: Dataset):
        """
        Generate the dataset by running the chained pipeline steps.
        dataset: the input dataset
        """
        for pipeline in self.pipelines:
            dataset = pipeline.generate(dataset)
        return dataset
