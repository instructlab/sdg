## v0.6.2

### Fixes

* Fixed a bug in our version specification of `docling` and `docling_parse` dependencies that were causing new installs of InstructLab to pull in incompatible versions of these. We also fixed a similar bug in the `mypy` dependency, but that one only impacts developers of SDG as opposed to users of InstructLab.

## v0.6.1

### Fixes

* Fixed a bug where generating data from a taxonomy with 2 or more changed knowledge leaf nodes would fail with a message about a destination path `already exists and is not an empty directory`

## v0.6.0

### Features

* Small knowledge datasets will automatically get upsampled during final data mixing based on the length of any precomputed skills datasets used during data mixing. This avoids issues where very large precomputed skills datasets were swamping the comparatively minor number of knowledge samples, resulting in lower than optimal knowledge retention during multiphase training. If a large precomputed dataset isn't in use during mixing (which is how things operate by default), this change is a no-op.
* When chunking PDF documents, we'll now look for the docling models on-disk in `$XDG_DATA_HOME/instructlab/sdg/models` (as well as `$XDG_DATA_DIRS` with the same `instructlab/sdg/models` subdirectory). If they are not found on disk, they'll automatically be downloaded from HuggingFace.
* When chunking PDF documents with Docling, we'll automatically configure Docling to use `tesserocr` if a working implementation is found instead of relying on `easyocr`. We fallback to `easyocr` if Tesseract is not properly configured for use by `tesserocr`.

### Breaking Changes

* Teacher model tokenizers are loaded from the local teacher model on-disk and not downloaded automatically from HuggingFace. The typical workflows in use so far expect the teacher model to exist on-disk, and this enforces that at least its tokenizer exists.
