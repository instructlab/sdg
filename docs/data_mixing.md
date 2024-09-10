# Data Mixing

As one of the last steps in data generation, the SDG library can optionally mix multiple datasets into a single output dataset in proportions specified by a recipe yaml file. The current implementation is designed to be used with mostly static recipes, that get used by default for every `ilab data generate` run. There is not yet an easy way to specify the recipe to use with each generation run, but we do make it possible to change the default recipe used for skills and/or knowledge data generation.

The primary intended use of this is to specify an optional pre-generated dataset maintained by the InstructLab community that can improve training results when attempting to teach new skills to a model. This process is a bit manual for now, and the steps to do that are documented below.

## Using InstructLab Community Pre-generated Dataset

To use the [InstructLab Community pre-generated dataset](https://huggingface.co/datasets/instructlab/InstructLabCommunity) with all skills training, we first need to create a default recipe that specifies this dataset to include when mixing generated skills data. This recipe will get automatically picked up if placed in a `default_data_recipes/skills.yaml` subfolder and file under one of several possible locations - `'/home/<user>/.local/share/instructlab/sdg'`, `'/usr/local/share/instructlab/sdg'`, or `'/usr/share/instructlab/sdg'`. The exact list of possible locations is platform-dependent, and can be enumerated by a Python command like below:

```python
python3 -c "import os; from xdg_base_dirs import xdg_data_home, xdg_data_dirs; data_dirs = [os.path.join(xdg_data_home(), 'instructlab', 'sdg')] + [os.path.join(dir, 'instructlab', 'sdg') for dir in xdg_data_dirs()]; print(data_dirs)"
```

For this example, we'll assume you want to place to default data recipe under the `~/.local/share/instructlab/sdg/` platform directory.

Ensure that directory exists and create the recipe yaml file:

```shell
mkdir -p ~/.local/share/instructlab/sdg/default_data_recipes/
cat <<EOF > ~/.local/share/instructlab/sdg/default_data_recipes/skills.yaml
datasets:
  - path: instructlab_community.jsonl
    sampling_size: 1.0
EOF
```

Next, download the `instructlab_community.jsonl` file from <https://huggingface.co/datasets/instructlab/InstructLabCommunity/tree/main> and place it in `~/.local/share/instructlab/datasets/`, where the recipe we wrote above will pick it up. If you prefer to place this pre-generated dataset in a different location, you can specify the absolute path to that different location in your recipe yaml file instead of using relative paths as shown here.

Then, during your next `ilab data generate`, you should see output near the end like:

```log
INFO 2024-08-06 16:08:42,069 instructlab.sdg.datamixing:123: Loading dataset from /home/user/.local/share/instructlab/datasets/instructlab_community.jsonl ...
Generating train split: 13863 examples [00:00, 185935.73 examples/s]
INFO 2024-08-06 16:08:42,414 instructlab.sdg.datamixing:125: Dataset columns: ['messages', 'metadata', 'id']
INFO 2024-08-06 16:08:42,414 instructlab.sdg.datamixing:126: Dataset loaded with 13863 samples
```

Your resulting `skills_train_*.jsonl` file will now contain the additional 13k+ examples from the pre-computed dataset, which should ensure your subsequent skills training doesn't regress in already-learned skills while being taught the new skill.
