Contributing to nikamap
=======================

Thank you for contributing to ``nikamap``.

This project provides tools to read and analyse NIKA2 continuum maps. Contributions
that improve scientific correctness, package stability, documentation, and test
coverage are all useful.

Ways to contribute
------------------

You can contribute by:

- reporting bugs or regressions;
- proposing or implementing new features;
- improving tests;
- clarifying documentation and examples;
- improving compatibility with supported Python, Astropy, and Photutils versions.

Before opening a change, check whether a similar issue or merge request already exists.
When you report a bug, include the package version, Python version, and a minimal example
that reproduces the problem.

Development setup
-----------------

``nikamap`` requires Python 3.10 or later.

1. Clone the repository.
2. Create and activate a virtual environment.
3. Install the package in editable mode with the development extras you need.

Example:

.. code:: bash

    python -m venv .venv
    source .venv/bin/activate
    python -m pip install --upgrade pip
    python -m pip install -e ".[test,docs]"

The optional ``stacking`` extra can also be installed when working on reprojection-related
features:

.. code:: bash

    python -m pip install -e ".[test,docs,stacking]"

Running tests
-------------

For a quick local test run:

.. code:: bash

    pytest --pyargs nikamap

The repository also defines a tox matrix to test multiple Python and Photutils combinations:

.. code:: bash

    tox -e py311-photutils230

You can list the available environments in ``tox.ini``. When fixing a compatibility issue,
prefer running the tox environment that matches the affected dependency combination.


Linting with Ruff
-----------------

Ruff is run in CI (GitLab job ``ruff``) and should be clean for contributions.
Run it locally before opening a merge request:

.. code:: bash

    python -m pip install ruff
    ruff check .

To automatically fix what Ruff can fix safely:

.. code:: bash

    ruff check . --fix

Documentation
-------------

The documentation sources live in ``doc/`` and examples are under ``examples/``.
To build the HTML documentation locally:

.. code:: bash

    tox -e build_docs

If your change modifies public behaviour, update the relevant documentation and examples.

Code and test expectations
--------------------------

- Keep changes focused and avoid unrelated refactors.
- Follow the existing coding style in the modified files.
- Add or update tests for behaviour changes and bug fixes.
- Preserve scientific metadata when applicable, especially units, WCS information, and beam handling.
- Be mindful of compatibility with the supported Photutils versions exercised by ``tox``.

Submitting changes
------------------

When opening a contribution, include:

- a short description of the problem being solved;
- the approach taken;
- any limitations or follow-up work;
- test coverage for the change.

If the change affects users, update ``CHANGELOG.md`` when appropriate.

Small, well-scoped contributions are easier to review and merge.