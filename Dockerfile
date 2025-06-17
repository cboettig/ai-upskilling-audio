# FROM quay.io/jupyter/pytorch-notebook:cuda12-ubuntu-24.04
FROM rocker/ml:cuda

# Install Poetry for package management
RUN curl -sSL https://install.python-poetry.org | python3 -

# Install all dependencies specified in the poetry configs.
RUN poetry install
