import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="cogvis",
    version="0.0.1",
    description="A CVS framework for specialised CVS development",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/TimothySimons/CVS_framework",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
