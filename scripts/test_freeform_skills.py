# Standard
from importlib import resources

# Third Party
from datasets import Dataset
from openai import OpenAI

# First Party
from src.instructlab.sdg.pipeline import (
    FULL_PIPELINES_PACKAGE,
    Pipeline,
    PipelineContext,
)

# for vLLM endpoints, the api_key remains "EMPTY"
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
        "seed_question": """Could you help me write a formal email to inquire about the progress of my internship application?""",
        "task_description": "Writing formal emails",
        "seed_response": """Subject: Inquiry Regarding the Status of My Internship Application

Dear [Recipient's Name],

I hope this email finds you well. I am writing to inquire about the current status of my internship application with [Company Name]. I submitted my application on [date of application] for the [Internship Title] position.

I am very interested in the opportunity to learn and grow as an intern at [Company Name], and I am eager to contribute my skills and enthusiasm to your team. I understand that the internship selection process may take some time, and I appreciate your consideration of my application.

If there are any additional steps I need to take or further information you require from me, please let me know. I am more than happy to provide any necessary documentation or complete additional tasks to facilitate the decision-making process.

I am excited about the prospect of joining [Company Name] and contributing to the [specific project, team, or aspect of the company] based on my background in [mention relevant skills or experiences]. I am confident that this internship will provide me with valuable experience and growth opportunities.

Thank you for your time and consideration. I look forward to hearing from you regarding the next steps in the internship application process.

Sincerely,

[Your Full Name]

[Your Contact Information]""",
    }
]


ds = Dataset.from_list(samples)

ctx = PipelineContext(client, "mixtral", teacher_model, 1)

with resources.path(FULL_PIPELINES_PACKAGE, "freeform_skills.yaml") as yaml_path:
    skills_pipe = Pipeline.from_file(ctx, yaml_path)

gen_data = skills_pipe.generate(ds)

print(gen_data)
print(gen_data[0])
