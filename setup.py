# setup.py
from setuptools import setup, find_packages

setup(
    name="futures_data_lake",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "polars",
        "pandas",
        "pyarrow",
    ],
)