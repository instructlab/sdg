#!/usr/bin/env python

# Standard
import glob
import json
import sys

# Third Party
from jsonschema import validate
import jsonschema
import yaml


def validate_yaml_file(yaml_file, schema):
    with open(yaml_file, "r") as file:
        pipeline = yaml.safe_load(file)

    try:
        validate(instance=pipeline, schema=schema)
        print(f"Validation successful for {yaml_file}.")
    except jsonschema.exceptions.ValidationError as err:
        print(f"Validation failed for {yaml_file}:", err)
        return False
    return True


def main():
    schema_path = "src/instructlab/sdg/pipelines/schema/v1.json"
    with open(schema_path, "r") as file:
        schema = json.load(file)

    yaml_files = glob.glob("src/instructlab/sdg/pipelines/**/*.yaml", recursive=True)
    all_valid = True
    for yaml_file in yaml_files:
        print("=======================================================")
        print("=== Validating", yaml_file)
        print("=======================================================")
        if not validate_yaml_file(yaml_file, schema):
            all_valid = False

    return 1 if not all_valid else 0


if __name__ == "__main__":
    sys.exit(main())
