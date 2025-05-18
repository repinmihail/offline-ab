#!/usr/bin/env bash

# create conda env
CONDA_ENV_NAME=`basename "$PWD"`

echo -e "Activate base environment ..."
conda activate base

echo -e "Remove $CONDA_ENV_NAME environment ..."
conda env remove -y -n $CONDA_ENV_NAME

echo -e "Create $CONDA_ENV_NAME environment ..."
conda create -y -n $CONDA_ENV_NAME python=3.10

echo -e "Activate $CONDA_ENV_NAME environment ..."
conda activate $CONDA_ENV_NAME

# poetry
rm poetry.lock
export POETRY_HOME="~/poetry"
export POETRY_VERSION="1.8.3"
export PYTHON_KEYRING_BACKEND=keyring.backends.null.Keyring
export PATH="$HOME/poetry/bin:$PATH"
curl -sSL https://install.python-poetry.org | python3 - --version $POETRY_VERSION
poetry install
rm poetry.lock

# clean
unset CONDA_ENV_NAME
