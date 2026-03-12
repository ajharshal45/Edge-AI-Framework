"""
EdgeAI Framework - Setup Configuration
Lightweight Edge AI Framework for Any Device
"""

from setuptools import setup, find_packages

setup(
    name="edgeai",
    version="1.0.0",
    description="Lightweight Edge AI Framework for Any Device",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    author="EdgeAI Team",
    author_email="team@edgeai.dev",
    url="https://github.com/edgeai/edgeai",
    license="MIT",

    packages=find_packages(),
    include_package_data=True,
    package_data={
        "edgeai": ["models/pretrained/*.pkl"],
    },

    python_requires=">=3.8",
    install_requires=[
        "scikit-learn>=1.0",
        "pandas>=1.3",
        "numpy>=1.21",
        "psutil>=5.8",
        "matplotlib>=3.4",
    ],

    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Operating System :: OS Independent",
    ],

    keywords="edge-ai iot machine-learning inference offline lightweight",

    entry_points={
        "console_scripts": [
            "edgeai=edgeai.EdgeAI:main",
        ],
    },
)
