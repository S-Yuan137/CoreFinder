from setuptools import setup, find_packages

setup(
    name="corefinder",
    version="0.1.2",
    author="Shibo Yuan",
    author_email="shibo_yuan@outlook.com",
    description="A package for finding cores in 3D simulation data.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="",
    packages=find_packages(),
    # packages=find_packages(exclude=["tests", "tests.*", "test_*"]),
    install_requires=[
        "numpy>=2.0.0",
        "scikit-image>=0.24.0",
        "connected-components-3d>=3.18.0",
        "h5py>=3.10.0",
    ],
    python_requires=">=3.9",
    extras_require={
        "optional": [
            "matplotlib>=3.9.0",
            "scikit-learn>=1.5.0",
        ],
    },
    license="MIT",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Operating System :: OS Independent",
    ],
)
