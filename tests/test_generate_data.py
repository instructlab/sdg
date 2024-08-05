"""
Unit tests for the top-level generate_data module.
"""

# Standard
from contextlib import contextmanager
from unittest.mock import MagicMock, patch
import glob
import json
import os
import shutil
import tempfile
import unittest

# Third Party
from datasets import load_dataset
import pytest

# First Party
from instructlab.sdg.generate_data import _context_init, generate_data
from instructlab.sdg.llmblock import LLMBlock
from instructlab.sdg.pipeline import PipelineContext

TEST_VALID_COMPOSITIONAL_SKILL_YAML = """created_by: rafael-vasquez
version: 1
seed_examples:
- answer: "Sure thing!"
  context: "This is a valid YAML."
  question: "Can you help me debug this failing unit test?"
- answer: "answer2"
  context: "context2"
  question: "question2"
- answer: "answer3"
  context: "context3"
  question: "question3"
- answer: "answer4"
  context: "context4"
  question: "question4"
- answer: "answer5"
  context: "context5"
  question: "question5"
task_description: 'This is a task'
"""

TEST_VALID_KNOWLEDGE_SKILL_YAML = """created_by: lukeinglis
domain: anatomy_tonsil
version: 3
seed_examples:
  - context: |
      ## Structure
      Humans are born with four types of tonsils: the pharyngeal tonsil, two
      tubal tonsils, two palatine tonsils, and the lingual tonsils.[1]

      <table>
      <thead>
      <tr class="header">
      <th><p>Type</p></th>
      <th><p><a href="Epithelium" title="wikilink">Epithelium</a></p></th>
      <th><p><a href=":wikt:capsule" title="wikilink">Capsule</a></p></th>
      <th><p><a href="Tonsillar_crypts" title="wikilink">Crypts</a></p></th>
      <th><p>Location</p></th>
      </tr>
      </thead>
      <tbody>
      <tr class="odd">
      <td><p><a href="Adenoid" title="wikilink">Pharyngeal tonsil</a> (also
      termed "adenoid")</p></td>
      <td><p><a href="pseudostratified_epithelium" title="wikilink">Ciliated
      pseudostratified columnar</a> (<a href="respiratory_epithelium"
      title="wikilink">respiratory epithelium</a>)</p></td>
      <td><p>Incompletely encapsulated</p></td>
      <td><p>Small folds—sometimes described as crypts<a href="#fn1"
      class="footnote-ref" id="fnref1"
      role="doc-noteref"><sup>1</sup></a></p></td>
      <td><p>Roof of <a href="pharynx" title="wikilink">pharynx</a></p></td>
      </tr>
      <tr class="even">
      <td><p><a href="Tubal_tonsils" title="wikilink">Tubal tonsils</a></p></td>
      <td><p>Ciliated pseudostratified columnar (respiratory epithelium)</p></td>
      <td><p>Not encapsulated</p></td>
      <td><p>No crypts</p></td>
      <td><p>Roof of pharynx</p></td>
      </tr>
      <tr class="odd">
      <td><p><a href="Palatine_tonsils" title="wikilink">Palatine tonsils</a></p></td>
      <td><p>Stratified squamous epithelium</p></td>
      <td><p>Fully encapsulated</p></td>
      <td><p>Multiple deep crypts</p></td>
      <td><p>Each side of the throat at the back of the mouth</p></td>
      </tr>

    questions_and_answers:
      - question: What is the location of the tubal tonsils?
        answer: The location of the tubal tonsils is the roof of the pharynx.
      - question: |
          Compare the epithelial types, encapsulation, and presence of
          crypts in the pharyngeal, tubal, and palatine tonsils according to the
          table provided.
        answer: |
          The pharyngeal tonsil features ciliated pseudostratified columnar
          epithelium and is incompletely encapsulated with small folds sometimes
          described as crypts. The tubal tonsils also have ciliated
          pseudostratified columnar epithelium but are not encapsulated and do
          not possess crypts. In contrast, the palatine tonsils are covered with
          stratified squamous epithelium, are fully encapsulated, and contain
          multiple deep crypts. These structural differences are indicative of
          their varied anatomical locations and potentially their distinct
          functions within the immune system.
      - question: What type of epithelium is found in the pharyngeal tonsil?
        answer: |
          The type of epithelium found in the pharyngeal tonsil is ciliated
          pseudostratified columnar (respiratory epithelium).


  - context: |
      The **tonsils** are a set of [lymphoid](Lymphatic_system "wikilink")
      organs facing into the aerodigestive tract, which is known as
      [Waldeyer's tonsillar ring](Waldeyer's_tonsillar_ring "wikilink") and
      consists of the [adenoid tonsil](adenoid "wikilink") (or pharyngeal
      tonsil), two [tubal tonsils](tubal_tonsil "wikilink"), two [palatine
      tonsils](palatine_tonsil "wikilink"), and the [lingual
      tonsils](lingual_tonsil "wikilink"). These organs play an important role
      in the immune system.

    questions_and_answers:
      - question: What is the immune system's first line of defense?
        answer: |
          The tonsils are the immune system's first line of defense against
          ingested or inhaled foreign pathogens.
      - question: What is Waldeyer's tonsillar ring?
        answer: |
          Waldeyer's tonsillar ring is a set of lymphoid organs facing into the
          aerodigestive tract, consisting of the adenoid tonsil, two tubal
          tonsils, two palatine tonsils, and the lingual tonsils.
      - question: How many tubal tonsils are part of Waldeyer's tonsillar ring?
        answer: There are two tubal tonsils as part of Waldeyer's tonsillar ring.

  - context: |
      The palatine tonsils tend to reach their largest size in [puberty](puberty
      "wikilink"), and they gradually undergo [atrophy](atrophy "wikilink")
      thereafter. However, they are largest relative to the diameter of the
      throat in young children. In adults, each palatine tonsil normally
      measures up to 2.5 cm in length, 2.0 cm in width and 1.2 cm in
      thickness.[2]

    questions_and_answers:
      - question: When do the palatine tonsils tend to reach their largest size?
        answer: The palatine tonsils tend to reach their largest size in puberty.
      - question: What are the typical dimensions of an adult palatine tonsil?
        answer: |
          In adults, each palatine tonsil normally measures up to 2.5 cm in
          length, 2.0 cm in width, and 1.2 cm in thickness.
      - question: How do the palatine tonsils change in size with age?
        answer: |
          The palatine tonsils tend to gradually undergo atrophy after puberty,
          becoming smaller in size compared to their dimensions in young
          children.

  - context: |
      The tonsils are immunocompetent organs that serve as the immune system's
      first line of defense against ingested or inhaled foreign pathogens, and
      as such frequently engorge with blood to assist in immune responses to
      common illnesses such as the common cold. The tonsils have on their
      surface specialized antigen capture cells called [microfold
      cells](microfold_cell "wikilink") (M cells) that allow for the uptake of
      antigens produced by pathogens. These M cells then alert the B cells and T
      cells in the tonsil that a pathogen is present and an immune response is
      stimulated.[3] B cells are activated and proliferate in areas called
      germinal centers in the tonsil. These germinal centers are places where B
      memory cells are created and [secretory antibody (IgA)](Immunoglobulin_A
      "wikilink") is produced.

    questions_and_answers:
      - question: |
          What are the specialized antigen capture cells on the surface of the
          tonsils called?
        answer: |
          The specialized antigen capture cells on the surface of the tonsils
          are called microfold cells (M cells).
      - question: What is the role of microfold cells in the tonsils?
        answer: |
          Microfold cells (M cells) allow for the uptake of antigens produced by
          pathogens. They alert the B cells and T cells in the tonsil that a
          pathogen is present, stimulating an immune response.
      - question: Where do B cells proliferate in the tonsils?
        answer: B cells proliferate in areas called germinal centers in the tonsils.

  - context: |
      A [tonsillolith](tonsillolith "wikilink") (also known as a "tonsil stone")
      is material that accumulates on the palatine tonsil. This can reach the
      size of a [peppercorn](peppercorn "wikilink") and is white or cream in
      color. The main substance is mostly [calcium](calcium "wikilink"), but it
      has a strong unpleasant odor because of [hydrogen
      sulfide](hydrogen_sulfide "wikilink") and [methyl
      mercaptan](methyl_mercaptan "wikilink") and other chemicals.[6]

    questions_and_answers:
      - question: What is a tonsillolith?
        answer: |
          A tonsillolith (tonsil stone) is material that accumulates on the
          palatine tonsil, reaching the size of a peppercorn and having a white
          or cream color. It contains calcium and has a strong unpleasant odor
          due to hydrogen sulfide, methyl mercaptan, and other chemicals.
      - question: What is the main substance found in a tonsillolith?
        answer: The main substance found in a tonsillolith is mostly calcium.
      - question: Why do tonsilloliths have a strong unpleasant odor?
        answer: |
          Tonsilloliths have a strong unpleasant odor due to hydrogen sulfide,
          methyl mercaptan, and other chemicals.

document_outline: |
  Overview of Human tonsils, describing their types, locations, structure,
  function, and clinical significance, with a specific focus on their role in
  the immune system and related health issues.

document:
  repo: https://github.com/luke-inglis/il-anatomy-knowledge
  commit: cc7c6ca
  patterns:
    - anatomy1.md
"""

TEST_TAXONOMY_BASE = "main"

TEST_CUSTOM_YAML_RULES = b"""extends: relaxed
rules:
  line-length:
    max: 180
"""

TEST_DATA_DIR = os.path.join(os.path.dirname(__file__), "testdata")


def _noop_llmblock_generate(self, samples):
    """Generate mock output based on input samples.

    Simply return the seed question and response from the input sample,
    joined using '?' and with an integer discriminator.

    _get_question_hack() and _get_response_hack() is the code that later
    splits these using the '?' separator.

    Return 10 output samples per input samples, since the LLMBlock in the
    simple pipeline is configured with 'n: scaled' and we pass
    num_instructions_to_generate=10 to generate_data.
    """

    def strip_q(q):
        return q.strip().rstrip("?")

    output = []
    for sample in samples:
        for i in range(10):
            if "domain" in sample:  # knowledge
                output.append(
                    sample["icl_document"]
                    + f" (q{i}) "
                    + strip_q(sample["icl_query_1"])
                    + f" ? (a{i}) "
                    + sample["icl_response_1"]
                )
            else:
                output.append(
                    sample["seed_context"]
                    + f" (q{i}) "
                    + strip_q(sample["seed_question"])
                    + f" ? (a{i}) "
                    + sample["seed_response"]
                )
    return output


@patch.object(LLMBlock, "_generate", _noop_llmblock_generate)
class TestGenerateCompositionalData(unittest.TestCase):
    @pytest.fixture(autouse=True)
    def _init_taxonomy(self, taxonomy_dir):
        self.test_taxonomy = taxonomy_dir

    def setUp(self):
        self.tmp_path = tempfile.TemporaryDirectory().name
        tracked_compositional_file = os.path.join(
            "compositional_skills", "tracked", "qna.yaml"
        )
        untracked_compositional_file = os.path.join(
            "compositional_skills", "new", "qna.yaml"
        )
        self.test_taxonomy.add_tracked(
            tracked_compositional_file, TEST_VALID_COMPOSITIONAL_SKILL_YAML
        )
        self.test_taxonomy.create_untracked(
            untracked_compositional_file, TEST_VALID_COMPOSITIONAL_SKILL_YAML
        )

    def test_generate(self):
        with patch("logging.Logger.info") as mocked_logger:
            generate_data(
                mocked_logger,
                model_family="merlinite",
                model_name="models/merlinite-7b-lab-Q4_K_M.gguf",
                num_instructions_to_generate=10,
                taxonomy=self.test_taxonomy.root,
                taxonomy_base=TEST_TAXONOMY_BASE,
                output_dir=self.tmp_path,
                yaml_rules=TEST_CUSTOM_YAML_RULES,
                client=MagicMock(),
                pipeline="simple",
            )

            node_file = os.path.join(
                "node_datasets_*", "compositional_skills_new.jsonl"
            )
            for name in [
                "test_*.jsonl",
                "train_*.jsonl",
                "messages_*.jsonl",
                "skills_recipe_*.yaml",
                "skills_train_*.jsonl",
                node_file,
            ]:
                file_name = os.path.join(self.tmp_path, name)
                print(f"Testing that generated file ({file_name}) exists")
                files = glob.glob(file_name)
                assert len(files) == 1

            # Test contents of generated files for contributed context
            for name in ["test_*.jsonl", "train_*.jsonl", "skills_train_*.jsonl"]:
                file_name = os.path.join(self.tmp_path, name)
                print(f"Testing contents of ({file_name})")
                files = glob.glob(file_name)
                with open(files[0], "r", encoding="utf-8") as jsonfile:
                    data_as_str = jsonfile.read()
                    generated_content_exists = False
                    if "This is a valid YAM" in data_as_str:
                        generated_content_exists = True
                    else:
                        print(f'"This is a valid YAM" not in data: {data_as_str}')
                    assert generated_content_exists is True

    def teardown(self) -> None:
        """Recursively remove the temporary repository and all of its
        subdirectories and files.
        """
        shutil.rmtree(self.tmp_path)
        return

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.teardown()


@patch.object(LLMBlock, "_generate", _noop_llmblock_generate)
class TestGenerateKnowledgeData(unittest.TestCase):
    @pytest.fixture(autouse=True)
    def _init_taxonomy(self, taxonomy_dir):
        self.test_taxonomy = taxonomy_dir

    def setUp(self):
        self.tmp_path = tempfile.TemporaryDirectory().name
        tracked_knowledge_file = os.path.join("knowledge  ", "tracked", "qna.yaml")
        untracked_knowledge_file = os.path.join("knowledge", "new", "qna.yaml")
        self.test_taxonomy.add_tracked(
            tracked_knowledge_file, TEST_VALID_KNOWLEDGE_SKILL_YAML
        )
        self.test_taxonomy.create_untracked(
            untracked_knowledge_file, TEST_VALID_KNOWLEDGE_SKILL_YAML
        )

    def test_generate(self):
        with patch("logging.Logger.info") as mocked_logger:
            generate_data(
                mocked_logger,
                model_family="merlinite",
                model_name="models/merlinite-7b-lab-Q4_K_M.gguf",
                num_instructions_to_generate=10,
                taxonomy=self.test_taxonomy.root,
                taxonomy_base=TEST_TAXONOMY_BASE,
                output_dir=self.tmp_path,
                yaml_rules=TEST_CUSTOM_YAML_RULES,
                chunk_word_count=1000,
                server_ctx_size=4096,
                client=MagicMock(),
                pipeline="simple",
            )
            for name in [
                "test_*.jsonl",
                "train_*.jsonl",
                "messages_*.jsonl",
                "skills_recipe_*.yaml",
                "skills_train_*.jsonl",
                "knowledge_recipe_*.yaml",
                "knowledge_train_*.jsonl",
            ]:
                file_name = os.path.join(self.tmp_path, name)
                print(f"Testing that generated file ({file_name}) exists")
                files = glob.glob(file_name)
                assert len(files) == 1

            for name in [
                "knowledge_new_p07.jsonl",
                "knowledge_new_p10.jsonl",
                "knowledge_new_task.yaml",
                "mmlubench_knowledge_new.jsonl",
            ]:
                file_name = os.path.join(self.tmp_path, "node_datasets_*", name)
                print(f"Testing that generated file ({file_name}) exists")
                files = glob.glob(file_name)
                assert len(files) == 1

            # Test contents of generated files for contributed context
            for name in [
                "test_*.jsonl",
                "train_*.jsonl",
                "skills_train_*.jsonl",
                "knowledge_train_*.jsonl",
            ]:
                file_name = os.path.join(self.tmp_path, name)
                print(f"Testing contents of ({file_name})")
                files = glob.glob(file_name)
                with open(files[0], "r", encoding="utf-8") as jsonfile:
                    data_as_str = jsonfile.read()
                    generated_content_exists = False
                    if "tonsil" in data_as_str:
                        generated_content_exists = True
                    else:
                        print(f"tonsil not in data: {data_as_str}")
                    assert generated_content_exists is True

    def teardown(self) -> None:
        """Recursively remove the temporary repository and all of its
        subdirectories and files.
        """
        shutil.rmtree(self.tmp_path)
        return

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.teardown()


def test_context_init_batch_size_optional():
    """Test that the _context_init function can handle a missing batch size by
    delegating to the default in PipelineContext.
    """
    ctx = _context_init(
        None,
        "mixtral",
        "foo.bar",
        1,
        "/checkpoint/dir",
        1,
        batch_size=None,
        batch_num_workers=None,
    )
    assert ctx.batch_size == PipelineContext.DEFAULT_BATCH_SIZE


def test_context_init_batch_size_optional():
    """Test that the _context_init function can handle a passed batch size"""
    ctx = _context_init(
        None,
        "mixtral",
        "foo.bar",
        1,
        "/checkpoint/dir",
        1,
        batch_size=20,
        batch_num_workers=32,
    )
    assert ctx.batch_size == 20
