from setuptools import setup, find_packages

with open("README.md", "r") as f:
    long_description = f.read()

setup(
    name="rcbench",
    version="0.0.10",
    description="Reservoir computing benchmark toolkit",
    #package_dir={"":"rcbench"},
    packages=find_packages(),
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/nanotechdave/RCbench",
    author="Davide Pilati",
    author_email="davide.pilati@polito.it",
    license="MIT",
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.9",
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
        "Operating System :: OS Independent",
    ],
    install_requires=[
        "numpy",
        "scipy",
        "matplotlib",
        "scikit-learn",
    ],
    extras_require={
        'test': [
            'pytest',
            'pytest-cov',
        ],
    },
    python_requires=">=3.9",
    include_package_data=True,
)
