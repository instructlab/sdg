# Local Development Setup

## Fork and clone the SDG repository locally

Follow GitHub's documentation for [forking a
repository](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/working-with-forks/fork-a-repo)
to learn how to fork and clone the
[instructlab/sdg](https://github.com/instructlab/sdg) repository to
your local machine. If you don't want to submit pull requests to SDG,
you can skip the forking step and directly clone the instructlab/sdg
repository. If you intend to submit a pull request, it's best to fork
first.

The subsequent instructions assume you have cloned SDG to a local
directory and are executing these commands from within that
directory. For example, it assumes you've done something like this:

```shell
git clone git@github.com:instructlab/sdg.git
cd sdg
```

## Setup a Python virtual environment and install InstructLab

```shell
python3 -m venv --upgrade-deps venv
. venv/bin/activate
pip install git+https://github.com/instructlab/instructlab@main
```

At this point, you have InstructLab installed and a working `ilab`
CLI. However, it is using a released version of SDG and not your
locally cloned one.

## Overwrite SDG with your locally cloned one

Now let's override the SDG version installed with our locally cloned
one. We'll use the `-e` flag to `pip install` so that we can edit the
SDG files and changes immediately take effect, without having to run
`pip install` again after each edit. We also install the development
requirements, necessary to run tests.

```shell
pip install -e .
pip install -r requirements-dev.txt
```

This may give a warning from pip about its dependency resolver and
instructlab-sdg versions. You can ignore that, as we are explicitly
using an unreleased local version of instructlab-sdg here.

## Test SDG

At this point, you should be able to run SDG tests locally as a sanity
check to ensure your local development environment is setup. The first
time you run them may take a bit as `tox` installs the necessary
development requirements. Subsequent runs will be faster.

### Unit Tests

These are relatively fast to run, with each testing a small section of
the SDG code.

```shell
tox -e py3-unit
```

### Functional Tests

These take a bit longer to run, and test larger functional areas of
the SDG code.

```shell
tox -e py3-functional
```

### Manual Testing / usage

You can also invoke the `ilab` CLI or use the SDG Python APIs directly
to test things locally. Details on how to use the `ilab` CLI are
maintained in the [upstream InstructLab
documentation](https://github.com/instructlab/instructlab).

## Running code formatting and linting checks locally

To run the same checks locally that our CI system uses to evaluate PRs
for linting errors, unused imports, code style, etc:

```shell
tox -e ruff -- check
tox -e lint
tox -e mypy
tox -e validate-pipelines
```

Also run the unit and functional tests, mentioned above, before
opening a pull request to ensure the tests pass with your changes.

## Proposing changes to the project

We follow a standard GitHub [fork and pull
model](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/getting-started/about-collaborative-development-models#fork-and-pull-model)
to allow anyone to fork our repository and propose changes via a pull
request. If you're unfamiliar with how to submit a pull request, the
GitHub documentation on [Collaborating with pull
requests](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests)
is a good guide to follow.
