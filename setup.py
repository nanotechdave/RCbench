from setuptools import setup, Extension, find_packages
from Cython.Build import cythonize
import numpy

extensions = [
    Extension("rcbench.tasks.c_metrics", ["rcbench/tasks/c_metrics.pyx"], include_dirs=[numpy.get_include()])
]

with open("RCbench/README.md", "r") as f:
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
    ext_modules=cythonize(extensions),
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.9",
        "Operating System :: OS Independent",
    ],
    install_requires=[
        "numpy",
        "scipy",
        "matplotlib",
        "scikit-learn",
        "cython",
    ],
    python_requires=">=3.9",
    include_package_data=True,
)
