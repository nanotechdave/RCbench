from setuptools import setup, Extension, find_packages
from Cython.Build import cythonize
import numpy

extensions = [
    Extension("rcda.tasks.c_metrics", ["rcda/tasks/c_metrics.pyx"],
              include_dirs=[numpy.get_include()])
]

with open("rcda/README.md", "r") as f:
    long_description = f.read()

setup(
    name="rcda",
    version="0.0.10",
    description="Physical reservoir computing benchmark toolkit",
    package_dir={"":"rcda"},
    packages=find_packages(where="rcda"),
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/nanotechdave/RCDA",
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
