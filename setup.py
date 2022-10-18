from setuptools import setup, find_packages

from pathlib import Path
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(
    name="h5torch",
    python_requires=">3.9.0",
    packages=find_packages(),
    license="LICENSE.txt",
    description="h5torch",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Gaetan De Waele",
    author_email="gaetandewaele@hotmail.com",
    url="https://github.com/gdewael/h5torch",
    install_requires=[
        "numpy",
        "h5py",
        "torch",
        "scipy"

    ],
)