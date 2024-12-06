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
