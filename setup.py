from pathlib import Path
from setuptools import setup, find_packages

here = Path(__file__).parent
readme = (here / "README.md").read_text(encoding="utf-8") if (here / "README.md").exists() else ""

setup(
    name="hyperspace",
    version="0.1.0",
    description="Hyperspace: lightweight utilities for ...",
    long_description=readme,
    long_description_content_type="text/markdown",
    author="Shay Snyder",
    author_email="ssnyde9@gmu.edu",
    url="https://github.com/shaymeister/HyperSpace",
    packages=find_packages(exclude=("tests", "docs")),
    include_package_data=True,
    python_requires=">=3.8",
    install_requires=[
        "IPython",
        "ipykernel",
        "matplotlib",
        "scipy",
        "torch",
        "torchvision",
    ],
    extras_require={
        "dev": ["pytest", "flake8"],
        "docs": ["sphinx"],
    },
    entry_points={
        "console_scripts": [
            # "hyperspace=hyperspace.cli:main",
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3 :: Only",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    license="MIT",
)