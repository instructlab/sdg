# SDG FAQs

## Where can I find more details about the synthetic data generation algorithm?

The [Large-Scale Alignment for ChatBots](https://arxiv.org/pdf/2403.01081) research paper does an excellent job of explaining the SDG process in section 3.2.

## How do I estimate the amount of samples a given leaf node will produce in the SDG process for training?

For each question and answer pair in the taxonomy leaf node in a skill file: an estimated 30 synthetic samples will be produced in the training dataset.  

For each knowledge leaf node: the formula to estimate the number of produced synthetic samples in the training dataset is:

```text
(total cumulative size of knowledge documents / max document chunk size) * number of qna pairs in the knowledge file leaf node * 30 synthetic samples per qna pair
```

For example: let’s say the total size of the knowledge markdown files in the knowledge directory are 1 MB in size and there are 15 question and answer pairs in the knowledge leaf node file. You can estimate the total number of knowledge synthetic samples for that leaf node to be:

```text
1MB * 1048576 bytes / 4 bytes per token / 1.3 tokens per word / 1000 word document chunks ~= 202 chunks

202 chunks * 15 qna pairs * 30 samples per pair = 90900 samples
```

Note that this is just an estimation technique: the exact number is non-deterministic since the results are ultimately evaluated against a judge model and results that do not pass a score metric are dropped from the overall set.

## How many seed_examples can InstructLab process in a knowledge leaf node

There is no known limit to the number of seed example entries for a knowledge leaf node. There must be a minimum of 5 seed examples. These parameters can be seen in the taxonomy [knowledge schema](https://github.com/instructlab/schema/blob/main/src/instructlab/schema/v3/knowledge.json).

## How many qna pairs can be listed for a given seed_example in a knowledge leaf node

**Exactly** 3 QNA pairs must be listed for a given seed_example. If more is specified they will be ignored and not processed by the appropriate prompt. This can be seen by looking at the [prompt files in SDG](/src/instructlab/sdg/configs/knowledge/simple_generate_qa.yaml#L21-L28).

## How long can a given seed_example be for a knowledge leaf node?

Cumulatively: the context, and all qna pairs must be able to fit in the context window of the teacher model being used along with the total document chunk processed with the context. For full scale SDG workloads using the Mixtral-8x7B-Instruct-v0.1 the context window is 32768 tokens. The document chunk size is  approximately 1000 words. Therefore: with an average of 1.3 tokens per word: you can estimate that the cumulative size of the context and qna pairs must be under  (32768 - (1.3 * 1000)) / 1.3 = 24206 words. Note that additional words outside the seed example are put in the prompt as well so you cannot go up to the 24206 word mark. It is recommended to leave a minimum of 1000 additional words for the associated prompt: meaning you should target cumulatively the entire seed example to have a maximum of 23000 words.

## How many seed_examples can InstructLab process in a skills leaf node?

There is no known limit to the number of seed example entries for a knowledge leaf node. There must be a minimum of 5 seed examples. These parameters can be seen in the taxonomy [skills schema](https://github.com/instructlab/schema/blob/main/src/instructlab/schema/v3/compositional_skills.json#L31>).

## How long can a given seed_example be for a skill leaf node?

Cumulatively: the context, and associated qna pair must be able to fit in the context window of the teacher model being used. For full scale SDG workloads using the Mixtral-8x7B-Instruct-v0.1 the context window is 32768 tokens. Therefore: with an average of 1.3 tokens per word: you can estimate that the cumulative size of the context and qna pairs must be under: 32768 / 1.3 = 25206 words. Note that additional words outside the seed example are put in the prompt as well so you cannot go up to the 25206 word mark. It is recommended to leave a minimum of 1000 additional words for the associated prompt: meaning you should target cumulatively the entire seed example to have a maximum of 24000 words.

Note free form skills do not specify a context and therefore the 24000 words are available to be used for the qna pairs.

## Can I mix and match grounded skills (skills with context) and freeform skills within a leaf node?

No: within a given leaf node: there must be uniformity in the types of skills defined. They all must either be freeform skills or grounded skills.
