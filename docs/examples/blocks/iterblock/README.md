# IterBlock

`IterBlock` is used to run multiple iterations of another `Block`,
such as to call an `LLMBlock` multiple times generating a new sample
from each iteration.

A simple example of its usage is shown in
[pipeline.yaml](pipeline.yaml), where we use it to call the
`DuplicateColumnsBlock` 5 times for every input sample, which results
in us generating 5 output samples per each input samples with the
specified column duplicated in each output sample.

Assuming you have SDG installed, you can run that example with a
command like:

```shell
python -m instructlab.sdg.cli.run_pipeline --pipeline pipeline.yaml --input input.jsonl --output output.jsonl
```
