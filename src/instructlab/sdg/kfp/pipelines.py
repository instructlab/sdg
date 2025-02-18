# SPDX-License-Identifier: Apache-2.0

# Standard
from collections import namedtuple
from datetime import datetime
from typing import Optional, NamedTuple
import os

# Third Party
from kfp import dsl

# Local
from .components import (
    duplicate_columns_block,
    generate_taxonomy,
    filter_by_value_block,
    flatten_columns_block,
    llm_block,
    mix_taxonomy_datasets,
    rename_columns_block,
    postprocess_taxonomy,
    preprocess_taxonomy,
    taxonomy_git_importer,
)


@dsl.pipeline
def e2e_pipeline(
    taxonomy_repo: str,
    pipeline: str,
    teacher_model_path: str,
) -> NamedTuple(  # type: ignore
    "outputs",
    [
        ("taxonomy_path", dsl.Artifact),
        ("preprocessed_path", dsl.Artifact),
        ("generated_path", dsl.Artifact),
        ("postprocessed_path", dsl.Artifact),
        ("mixed_skills", dsl.Dataset),
        ("mixed_knowledge", dsl.Dataset),
    ],
):
    date_suffix = datetime.now().replace(microsecond=0).isoformat().replace(":", "_")

    # TODO: Figure out if we can use kfp dsl.importer instead
    # of our own simplistic importer here
    #
    # taxonomy_importer = dsl.importer(
    #     artifact_uri=taxonomy_path,
    #     artifact_class=dsl.Artifact,
    #     reimport=True,
    # )
    taxonomy_importer = taxonomy_git_importer(
        taxonomy_repo=taxonomy_repo,
    )

    preprocess_task = preprocess_taxonomy(
        taxonomy_path=taxonomy_importer.output,
        teacher_model_path=teacher_model_path,
        taxonomy_base="empty",
    )

    generate_task = generate_taxonomy(
        preprocessed_path=preprocess_task.output,
        pipeline=pipeline,
        model_id="mock",
        num_cpus=1,  # Test is faster running on a single CPU vs forking
        batch_size=0,  # Disable batch for tiny dataset and fastest test
    )

    postprocess_task = postprocess_taxonomy(
        generated_path=generate_task.output,
        date_suffix=date_suffix,
        pipeline=pipeline,
    )

    mix_task = mix_taxonomy_datasets(
        postprocessed_path=postprocess_task.output,
        date_suffix=date_suffix,
    )

    outputs = namedtuple(
        "outputs",
        [
            "taxonomy_path",
            "preprocessed_path",
            "generated_path",
            "postprocessed_path",
            "mixed_skills",
            "mixed_knowledge",
        ],
    )
    return outputs(
        taxonomy_importer.output,
        preprocess_task.output,
        generate_task.output,
        postprocess_task.output,
        mix_task.outputs["mixed_skills"],
        mix_task.outputs["mixed_knowledge"],
    )


@dsl.pipeline
def full_knowledge_pipeline(
    dataset_path: str,
) -> dsl.Dataset:
    dataset_importer = dsl.importer(
        artifact_uri=dataset_path,
        artifact_class=dsl.Dataset,
        reimport=True,
    )

    duplicate_document_col = duplicate_columns_block(
        input_ds=dataset_importer.output,
        columns_map={"document": "base_document"},
    )

    block_config = {
        "system": "You are an AI assistant that is an expert at fixing spelling errors in documents.",
        "introduction": "Give me a copy of the below document with all spelling errors corrected.",
        "principles": """Do not add any new information.
Do not leave out any information.""",
        "examples": "",
        "generation": """Document:
{{document}}""",
        "start_tags": [""],
        "end_tags": [""],
    }
    gen_spellcheck = llm_block(
        input_ds=duplicate_document_col.output,
        block_name=dsl.PIPELINE_TASK_NAME_PLACEHOLDER,
        output_cols=["spellcheck"],
        block_config=block_config,
        gen_kwargs={"max_tokens": 128},
    )

    flatten_auxiliary_columns = flatten_columns_block(
        input_ds=gen_spellcheck.output,
        var_cols=["spellcheck", "base_document"],
        value_name="corrected_document",
        var_name="dataset_type",
    )

    rename_to_document_column = rename_columns_block(
        input_ds=flatten_auxiliary_columns.output,
        columns_map={"document": "raw_document", "corrected_document": "document"}
    )

    block_config = {
        "system": "You are a very knowledgeable AI Assistant that will faithfully assist the user with their task.",
        "introduction": "Develop a series of educational question and answer pairs from a chapter in a {{domain}} textbook.",
        "principles": """The questions should:
* Be self-contained, not requiring references to tables, figures, or specific sections in the text for understanding.
* Focus on teaching and reinforcing the key knowledge and concepts presented in the chapter.
* Avoid sections with minimal educational content like index pages or prefaces. In such cases, respond with [UNANSWERABLE].
* Be directly relevant to the textbook's domain. For instance, in a science textbook, questions should revolve around scientific terms, definitions, and practical applications, while in a legal textbook, they should cover legal principles, case law, and precedents.
* Be formulated to allow for independent answers, avoiding direct references to specific theorems or text sections. For example, rather than asking 'Under what conditions is the fixed point of a function unique according to Theorem 3.1.5?', ask 'How does the Fixed Point Iteration method contribute to understanding function uniqueness?'
* Span a range of difficulty levels to accommodate a diverse student audience, from basic understanding to advanced comprehension.
* Include a variety of question types such as multiple-choice for basic recall, short answer for deeper understanding, and essay or problem-solving questions to test application and analysis skills.
* Align closely with the learning objectives of the textbook or the specific chapter, ensuring that the questions test the fundamental concepts and skills that the chapter aims to impart.

Strictly follow this format for each question answer pair your generate while responding

[QUESTION]
<Insert question here>
[ANSWER]
<Insert answer here>
[END]


Each question and answer pair should stand alone as a mini-lesson, encapsulating a key concept or idea from the chapter in a way that is accessible and informative without requiring the reader to refer back to the textbook.
""",
        "examples": """Here are some examples of questions:

[Document]
{{icl_document}}

[QUESTION]
{{icl_query_1}}
[ANSWER]
{{icl_response_1}}
[END]

[QUESTION]
{{icl_query_2}}
[ANSWER]
{{icl_response_2}}
[END]

[QUESTION]
{{icl_query_3}}
[ANSWER]
{{icl_response_3}}
[END]
""",
        "generation": """Here is the document:

[DOCUMENT]
{{document_outline}}
{{document}}
""",
    }
    gen_knowledge = llm_block(
        input_ds=rename_to_document_column.output,
        block_name=dsl.PIPELINE_TASK_NAME_PLACEHOLDER,
        output_cols=["question", "response"],
        block_config=block_config,
        gen_kwargs={"max_tokens": 128},
        parser_kwargs={
            "parser_name": "custom",
            "parsing_pattern": "\[(?:Question|QUESTION)\]\s*(.*?)\s*\[(?:Answer|ANSWER)\]\s*(.*?)\s*(?=\[(?:Question|QUESTION)\]|$)",
            "parser_cleanup_tags": ["[END]", "[End]"],
        },
        drop_duplicates=["question"],
    )

    # gen_knowledge = dsl.importer(
    #     artifact_uri="/home/bbrownin/src/instructlab/sdg/local_outputs/full-knowledge-pipeline-2025-02-19-10-14-14-751348/llm-block-2/output_ds",
    #     artifact_class=dsl.Artifact,
    #     reimport=True,
    # )

    block_config = {
        "system": "You are a very knowledgeable AI Assistant that will faithfully assist the user with their task.",
        "introduction": "Determine if the provided information is corroborated by the given context. Respond with YES if the context substantiates the information, even partially. Answer NO if the context does not support the information.",
        "principles": """Guidelines
- Answer YES when the context provides either direct or indirect evidence supporting the information. Indirect evidence may include contextual implications or inferred connections that reasonably support the information.
- Answer NO if the context lacks any supportive evidence, clearly contradicts the information, or if the support provided by the context is too vague or speculative to establish a solid connection to the information.
- Avoid using "partially" in your response. If the context provides any reasonable support (direct or indirect) for the information, consider it as a YES.

Strictly answer in this format
[Start of Context]
...
[End of Context]
[Start of Response]
...
[End of Response]
[Start of Explanation]
...
[End of Explanation]
[Start of Answer]
...
[End of Answer]
""",
        "examples": """Example 1:
[Start of Context]
An apple pie is a fruit pie with apples as the main filling. It's often served with whipped cream, ice cream, custard, or cheddar cheese. Typically, it has a double crust, with pastry above and below the filling. The upper crust can be solid or latticed.
[End of Context]
[Start of Response]
Apple pie is generally double-crusted.
[End of Response]
[Start of Explanation]
The context directly supports the information by stating that apple pie is "generally double-crusted," which matches the information provided.
[End of Explanation]
[Start of Answer]
YES
[End of Answer]

Example 2:
[Start of Context]
An apple pie is a fruit pie with apples as the main filling. It's often served with whipped cream, ice cream, custard, or cheddar cheese. Typically, it has a double crust, with pastry above and below the filling. The upper crust can be solid or latticed.
[End of Context]
[Start of Response]
Apple pies taste bad.
[End of Response]
[Start of Explanation]
The context does not provide any information about the taste of apple pies. The statement "Apple pies taste bad" is a subjective opinion and is not supported or mentioned in the given context.
[Start of Explanation]
[Start of Answer]
YES
[End of Answer]
""",
        "generation": """Now, based on the above examples and guidelines, determine if the following information is supported by the context provided. Answer YES or NO.
* Return the explanation within the [Start of Explanation] and [End of Explanation] tags.
* Return the answer between [Start of Answer] and [End of Answer] tags.

[Start of Context]
{{document}}
[End of Context]
[Start of Response]
{{response}}
[End of Response]
""",
        "start_tags": ["[Start of Explanation]", "[Start of Answer]"],
        "end_tags": ["[End of Explanation]", "[End of Answer]"],
    }
    eval_faithfulness_qa_pair = llm_block(
        input_ds=gen_knowledge.output,
        block_name=dsl.PIPELINE_TASK_NAME_PLACEHOLDER,
        output_cols=["explanation", "judgment"],
        block_config=block_config,
        gen_kwargs={"max_tokens": 128},
    )

    # eval_faithfulness_qa_pair = dsl.importer(
    #     artifact_uri="/home/bbrownin/src/instructlab/sdg/local_outputs/full-knowledge-pipeline-2025-02-19-10-14-14-751348/llm-block-3/output_ds",
    #     artifact_class=dsl.Artifact,
    #     reimport=True,
    # )

    filter_faithfulness = filter_by_value_block(
        input_ds=eval_faithfulness_qa_pair.output,
        filter_column="judgment",
        filter_value="YES",
        operation="eq",
        drop_columns=["judgment", "explanation"],
    )

    block_config = {
        "system": "You are a very knowledgeable AI Assistant that will faithfully assist the user with their task.",
        "introduction": "Your task is to assess the relevance of a given response to a specific query. This evaluation should be conducted methodically by answering two key questions:",
        "principles": """1. Subject Matter Relevance: Does the provided response accurately match the subject matter of the user's query? This question aims to determine if the response is directly related to the main topic or issue presented in the query.
2. Focus and Perspective Addressing: Does the provided response effectively address the focus or perspective on the subject matter as outlined in the user's query? This question seeks to evaluate whether the response not only matches the subject matter but also aligns with the specific angle or concern raised by the user.

For each question, assign a score of 1 point if the response meets the criteria, and 0 points if it does not. After evaluating each question, provide detailed feedback explaining your reasoning behind the scores awarded.

Conclude your evaluation with a final result, strictly using the following format: 'Total Score: X'. The total score should represent the sum of points assigned for each question, with a maximum possible score of 2 points.
Only evaluate the response based on the above criteria, do not create new questions.
""",
        "examples": """Example 1:
[Start of Question]
What is the impact of global warming on polar bears?
[End of Question]

[Start of Response]
Global warming leads to melting ice caps, reducing the habitat of polar bears and negatively impacting their hunting grounds.
[End of Response]

[Start of Feedback]
- Subject Matter Relevance Score: 1 (The response is directly related to the impact of global warming on polar bears.)
- Alignment with Query's Focus Score: 1 (The response specifically addresses how global warming affects polar bears' habitat and hunting grounds.)
[End of Feedback]

[Start of Score]
2
[End of Score]

Example 2:
[Start of Question]
How does photosynthesis work?
[End of Question]

[End of Response]
Plants require sunlight and water to grow.
[End of Response]

[Start of Feedback]
- Subject Matter Relevance Score: 0 (The response is related to plant growth, but does not specifically address the process of photosynthesis.)
- Alignment with Query's Focus Score: 0 (The response fails to detail the photosynthesis process, missing the specific focus of the query.)
[End of Feedback]

[Start of Score]
0
[End of Score]


Example 3:
[Start of Question]
What are the benefits of electric vehicles?
[End of Question]

[Start of Response]
Electric vehicles reduce dependency on fossil fuels and decrease greenhouse gas emissions.
[End of Response]

[Start of Feedback]
- Subject Matter Relevance Score: 1 (The response matches the query's subject on the benefits of electric vehicles.)
- Alignment with Query's Focus Score: 1 (The response effectively addresses the environmental benefits of electric vehicles, aligning with the query's focus.)
[End of Feedback]

[Start of Score]
2
[End of Score]
""",
        "generation": """Begin your response by providing the feedback followed by the score. Be as objective as possible.

[Start of Question]
{{question}}
[End of Question]

[Start of Response]
{{response}}
[End of Response]

* Return the feedback within the [Start of Feedback] and [End of Feedback] tags.
* Return the final score between [Start of Score] and [End of Score] tags.
""",
        "start_tags": ["[Start of Feedback]", "[Start of Score]"],
        "end_tags": ["[End of Feedback]", "[End of Score]"],
    }
    eval_relevancy_qa_pair = llm_block(
        input_ds=filter_faithfulness.output,
        block_name=dsl.PIPELINE_TASK_NAME_PLACEHOLDER,
        output_cols=["feedback", "score"],
        block_config=block_config,
        gen_kwargs={"max_tokens": 128},
    )

    # eval_relevancy_qa_pair = dsl.importer(
    #     artifact_uri="/home/bbrownin/src/instructlab/sdg/local_outputs/full-knowledge-pipeline-2025-02-19-10-51-17-979594/llm-block/output_ds",
    #     artifact_class=dsl.Artifact,
    #     reimport=True,
    # )

    filter_relevancy = filter_by_value_block(
        input_ds=eval_relevancy_qa_pair.output,
        filter_column="score",
        filter_value="2.0",
        operation="eq",
        convert_dtype="float",
        drop_columns=["feedback", "score"],
    )

    block_config = {
        "system": "You are a very knowledgeable AI Assistant that will faithfully assist the user with their task.",
        "introduction": "Given below question can you verify if it meets below requirements and based on them give a rating of 1 if it meets all of them or 0 otherwise.",
        "principles": """Here are the requirements:

Non-Referential Clarity and Contextual Independence: Ensure that the question is self-explanatory and does not rely on specific, unprovided external content, such as particular documents, specific tables, or detailed datasets. The question should be structured to be understandable and clear without requiring direct access to or knowledge of these specific external sources.

Subject-Aware Completeness: The question should be crafted to be answerable on its own, given a reasonable level of specialized knowledge in the relevant subject area. It is acceptable and encouraged for the question to require specialized understanding pertinent to the topic; however, it should not depend on unique, external information not provided in the question itself. This distinction allows for questions that necessitate a deep understanding of a subject while ensuring they are not tied to specific external content like a particular dataset or a line in a document.

Please give your answer as short explanation followed by rating of either 0 or 1 as below.

* Return a short explanation within the [Start of Explanation] and [End of Explanation] tags.
* Return the rating on a binary 0/1 scale between [Start of Rating] and [End of Rating] tags.

[Start of Question]
...
[End of Question]

[Start of Explanation]
...
[End of Explanation]

[Start of Rating]
...
[End of Rating]
""",
        "examples": "",
        "generation": """[Start of Question]
{{question}}
[End of Question]
""",
        "start_tags": ["[Start of Explanation]", "[Start of Rating]"],
        "end_tags": ["[End of Explanation]", "[End of Rating]"],
    }
    eval_verify_question = llm_block(
        input_ds=filter_relevancy.output,
        block_name=dsl.PIPELINE_TASK_NAME_PLACEHOLDER,
        output_cols=["explanation", "rating"],
        block_config=block_config,
        gen_kwargs={"max_tokens": 128},
    )

    # eval_verify_question = dsl.importer(
    #     artifact_uri="/home/bbrownin/src/instructlab/sdg/local_outputs/full-knowledge-pipeline-2025-02-19-11-22-00-074819/llm-block/output_ds",
    #     artifact_class=dsl.Artifact,
    #     reimport=True,
    # )

    filter_verify_question = filter_by_value_block(
        input_ds=eval_verify_question.output,
        filter_column="rating",
        filter_value="1.0",
        operation="eq",
        convert_dtype="float",
        drop_columns=["explanation", "rating", "__index_level_0__"],
    )

    return filter_verify_question.output
