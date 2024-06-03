from setuptools import setup, find_packages

setup(
    name="DREDDA",
    version="0.1.0",  # Adjust as per your versioning
    author="Zhongxiao Li",
    author_email="lzx325@outlook.com",
    description="Drug Repositioning through Expression Data Domain Adaptation",
    keywords="drug repositioning, domain adaptation",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/lzx325/DREDDA",
    packages=["dredda"],
    classifiers=[
        "Programming Language :: Python :: 3.8",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Development Status :: 3 - Alpha"
    ],
    python_requires=">=3.8",
    install_requires=[
        "torch==1.6.0",
        "torchvision==0.7.0",
        "numpy==1.20.2",
        "scipy==1.6.2",
        "pandas==1.2.4",
        "tables==3.6.1",
        "h5py==2.10.0",
        "PyYAML==6.0",
        "scikit-learn==1.1.0",
        "requests>=2.25",
        "tqdm",
    ],
    extras_require={
        "dev": [
            "pytest>=3.7",
            "check-manifest",
            "twine",
        ],
    },
)
