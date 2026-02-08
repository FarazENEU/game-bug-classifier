from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="game-bug-classifier",
    version="0.1.0",
    author="Faraz",
    description="Fine-tuned LLM for game bug report classification and triage",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/game-bug-classifier",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "bug-classifier-train=scripts.train:main",
            "bug-classifier-eval=scripts.evaluate:main",
            "bug-classifier-predict=scripts.inference:main",
        ],
    },
)
