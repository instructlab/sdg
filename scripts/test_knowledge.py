# Standard
from importlib import resources
import operator

# Third Party
from datasets import Dataset
from openai import OpenAI

# First Party
from src.instructlab.sdg.pipeline import (
    FULL_PIPELINES_PACKAGE,
    Pipeline,
    PipelineContext,
)

# Please don't add you vLLM endpoint key here
openai_api_key = "EMPTY"
openai_api_base = "Add model endpoint here"

client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)

models = client.models.list()
teacher_model = models.data[0].id

samples = [
    {
        "icl_query_1": "what is the location of the tubal tonsils?",
        "icl_response_1": "The location of the tubal tonsils is the roof of the pharynx.",
        "icl_query_2": "How long does the adenoid grow?",
        "task_description": "Teaching about human anatomy, specifically tonsils",
        "icl_response_2": "The adenoid grows until the age of 5, starts to shrink at the age of 7 and becomes small in adulthood.",
        "icl_query_3": "What is the immune systems first line of defense against ingested or inhaled foreign pathogens?",
        "icl_response_3": "The tonsils are the immune systems first line of defense.",
        "document": "The **tonsils** are a set of lymphoid organs facing into the aerodigestive tract, which is known as Waldeyer's tonsillar ring and consists of the adenoid tonsil or pharyngeal tonsil, two tubal tonsils, two palatine tonsils, and the lingual tonsils. These organs play an important role in the immune system. When used unqualified, the term most commonly refers specifically to the palatine tonsils, which are two lymphoid organs situated at either side of the back of the human throat. The palatine tonsils and the adenoid tonsil are organs consisting of lymphoepithelial tissue located near the oropharynx and nasopharynx parts of the throat",
        "domain": "textbook",
    }
]

ds = Dataset.from_list(samples)

ctx = PipelineContext(client, "mixtral", teacher_model, 1)

with resources.path(FULL_PIPELINES_PACKAGE, "knowledge.yaml") as yaml_path:
    knowledge_pipe = Pipeline.from_file(ctx, yaml_path)

mmlubench_data = knowledge_pipe.generate(ds)

print(mmlubench_data)
print(mmlubench_data[0])
