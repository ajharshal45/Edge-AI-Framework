"""
EdgeFlow Framework — pip setup
================================
Install:
    pip install -e .        (editable / dev mode)
    pip install .           (production install)

After install, the CLI is available as:
    edgeflow train --data data.csv --target esp32
"""

from setuptools import setup, find_packages
import os

# Read long description from root README
here = os.path.abspath(os.path.dirname(__file__))
readme_path = os.path.join(here, "..", "README.md")
try:
    with open(readme_path, "r", encoding="utf-8") as f:
        long_description = f.read()
except FileNotFoundError:
    long_description = "EdgeFlow — Train on your laptop. Deploy to any IoT device. Run forever offline."

setup(
    name="edgeflow",
    version="1.0.0",
    author="VIT Pune EDI Group E-16",
    author_email="edgeflow@example.com",
    description=(
        "Open-source Edge AI framework for deploying ML models to IoT devices "
        "with zero cloud dependency."
    ),
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ajharshal45/Edge-AI-Framework",
    license="MIT",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "scikit-learn>=1.0",
        "pandas>=1.3",
        "numpy>=1.21",
    ],
    entry_points={
        "console_scripts": [
            "edgeflow=edgeflow.cli.commands:main",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Embedded Systems",
    ],
    keywords=[
        "edge-ai", "iot", "esp32", "raspberry-pi",
        "machine-learning", "embedded", "offline-ai",
    ],
    include_package_data=True,
)
