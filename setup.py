from setuptools import setup, find_packages


with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="pycausalgps", 
    version="0.0.2",
    author="Naeem Khoshnevis, Xiao Wu, Danielle Braun",
    author_email="nkhoshnevis@g.harvard.edu, xw2892@cumc.columbia.edu, dbraun@mail.harvard.edu",
    maintainer="Naeem Khoshnevis",
    maintainer_email = "nkhoshnevis@g.harvard.edu",
    description="Matching on generalized propensity scores with continuous exposures",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/NSAPH-Software/pycausalgps",
    license="GPL-3",
    packages=find_packages(exclude=["docs*", "tests*", "scripts*",
                                     "notebooks*"]),
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Natural Language :: English",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3 :: Only",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: Implementation :: CPython",
        "Programming Language :: Python :: Implementation :: PyPy",
    ],
    python_requires='>=3.7',
)
