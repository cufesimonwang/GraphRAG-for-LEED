from setuptools import setup, find_packages

setup(
    name="graphrag",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "langchain>=0.1.0",
        "openai>=1.0.0",
        "pydantic>=2.0.0",
        "pyyaml>=6.0.0",
        "networkx>=3.0.0",
        "pytest>=7.0.0",
    ],
    python_requires=">=3.8",
    author="Simon Wang",
    author_email="cufesimonwang@gmail.com",
    description="A GraphRAG system for LEED documentation",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/cufesimonwang/GraphRAG-for-LEED",
    project_urls={
        "Bug Tracker": "https://github.com/cufesimonwang/GraphRAG-for-LEED/issues",
        "Documentation": "https://github.com/cufesimonwang/GraphRAG-for-LEED#readme",
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
) 