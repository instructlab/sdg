# Third Party
from datasets import Dataset
from openai import OpenAI

# First Party
from src.instructlab.sdg import SDG
from src.instructlab.sdg.pipeline import (
    FULL_GROUNDED_SKILLS_FILE,
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
        "seed_context": """*Ms. Thompson:* Good morning, everyone. Today, we''re here to discuss
    our customer journey mapping and analysis. I believe this is crucial to understanding
    our customers'' experiences and improving our services.


    *Mr. Patel:* I agree, Lisa. We should start by identifying all touchpoints in
    our customer journey, from initial contact to post-sale support.


    *Ms. Rodriguez:* Yes, and let''s not forget about the emotional aspect of the
    journey. How do our customers feel at each stage? What are their pain points?


    *Mr. Kim:* We can use data from our CRM system to track the customer journey and
    gather insights. This will help us create a detailed, data-driven map.


    *Ms. Johnson:* Once we have the map, we can analyze it to identify areas for improvement.
    Perhaps there are steps where customers drop off or express dissatisfaction.


    *Mr. Davis:* We should also consider the customer''s perspective. Conducting interviews
    or surveys can provide valuable insights into their thoughts and feelings.


    *Ms. Brown:* Absolutely. And once we''ve identified areas for improvement, we
    can develop strategies to address them. This might involve redesigning certain
    touchpoints, enhancing our communication, or streamlining processes.


    *Mr. Smith:* And we must remember to measure the impact of any changes we make.
    By tracking key performance indicators, we can determine whether our efforts are
    successful.


    *Ms. Thompson:* Great suggestions, everyone. Let''s divide into teams to tackle
    different aspects of the customer journey. Team 1 will focus on pre-sale touchpoints,
    Team 2 on sales, and Team 3 on post-sale support.


    *Mr. Patel:* Sounds good. We''ll reconvene in a week to share our findings and
    discuss next steps.""",
        "seed_question": """Generate a company wide email based on the given meeting transcript""",
        "task_description": "Writing emails based on minutes of meeting",
        "seed_response": """Subject: Recap and Next Steps - Customer Journey Mapping and Analysis\n\
    \nDear [Company Name] Team,\n\nI hope this email finds you well. Yesterday, we\
    \ gathered to discuss our customer journey mapping and analysis, aiming to understand\
    \ our customers' experiences and identify opportunities for improvement. The discussions\
    \ were fruitful, and I want to share a brief overview of the key points and outcomes.\n\
    \n1. **Identifying Touchpoints:**\n   Mr. Patel suggested mapping all touchpoints\
    \ in our customer journey, from initial contact to post-sale support.\n\n2. **Emotional\
    \ Aspect and Pain Points:**\n   Ms. Rodriguez emphasized the importance of considering\
    \ the emotional aspect of the journey and identifying customers' pain points at\
    \ each stage.\n\n3. **Data-Driven Mapping:**\n   Mr. Kim proposed using data from\
    \ our CRM system to create a detailed, data-driven customer journey map.\n\n4.\
    \ **Customer Perspective:**\n   Ms. Johnson recommended gathering insights from\
    \ the customer's perspective through interviews or surveys.\n\n5. **Analysis and\
    \ Improvement:**\n   Ms. Brown suggested analyzing the customer journey map to\
    \ identify areas for improvement and developing strategies to address them.\n\n\
    6. **Measuring Impact:**\n   Mr. Smith stressed the need to measure the impact\
    \ of any changes made by tracking key performance indicators.\n\nTo facilitate\
    \ a comprehensive analysis, we have divided into teams to tackle different aspects\
    \ of the customer journey:\n\n* Team 1: Pre-sale touchpoints\n* Team 2: Sales\n\
    * Team 3: Post-sale support\n\nEach team will share their findings and discuss\
    \ next steps in a week.\n\nYour engagement and insights have been invaluable in\
    \ understanding our customers' experiences and identifying opportunities for improvement.\
    \ I look forward to our continued collaboration as we work towards enhancing our\
    \ services and delivering exceptional customer experiences.\n\nBest regards,\n\
    \n[Your Full Name]\n[Your Position]\n[Company Name]""",
    }
]


ds = Dataset.from_list(samples)

ctx = PipelineContext(client, "mixtral", teacher_model, 10)

skills_pipe = Pipeline.from_file(ctx, FULL_GROUNDED_SKILLS_FILE)

sdg = SDG([skills_pipe])
gen_data = sdg.generate(ds)

print(gen_data)
print(gen_data[0])
