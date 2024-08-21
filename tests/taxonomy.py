# SPDX-License-Identifier: Apache-2.0

# Standard
from pathlib import Path
from typing import Any, Dict, List
import shutil

# Third Party
import git
import yaml


class MockTaxonomy:
    INIT_COMMIT_FILE = "README.md"

    def __init__(self, path: Path) -> None:
        self.root = path
        self._repo = git.Repo.init(path, initial_branch="main")
        with open(path / self.INIT_COMMIT_FILE, "wb"):
            pass
        self._repo.index.add([self.INIT_COMMIT_FILE])
        self._repo.index.commit("Initial commit")

    @property
    def untracked_files(self) -> List[str]:
        """List untracked files in the repository"""
        return self._repo.untracked_files

    def create_untracked(self, rel_path: str, contents: Dict[str, Any]) -> Path:
        """Create a new untracked file in the repository.

        Args:
            rel_path (str): Relative path (from repository root) to the file.
            contents (Dict[str, Any]): Object to be written to the file.
        Returns:
            file_path: The path to the created file.
        """
        taxonomy_path = Path(rel_path)
        assert not taxonomy_path.is_absolute()
        file_path = self.root.joinpath(taxonomy_path)
        file_path.parent.mkdir(exist_ok=True, parents=True)
        with file_path.open(mode="w", encoding="utf-8") as fp:
            yaml.dump(contents, fp)
        return file_path

    def add_tracked(self, rel_path, contents: Dict[str, Any]) -> Path:
        """Add a new tracked file to the repository (and commit it).

        Args:
            rel_path (str): Relative path (from repository root) to the file.
            contents (Dict[str, Any]): Object to be written to the file.
        Returns:
            file_path: The path to the added file.
        """
        file_path = self.create_untracked(rel_path, contents)
        self._repo.index.add([rel_path])
        self._repo.index.commit("new commit")
        return file_path

    def teardown(self) -> None:
        """Recursively remove the temporary repository and all of its
        subdirectories and files.
        """
        shutil.rmtree(self.root)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.teardown()
