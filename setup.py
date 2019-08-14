import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="loras",
    version="0.0.7-beta",
    author="Saptarshi Bej, Narek Davtyan",
    author_email="davtyannarek@hotmail.com",
    description="A small package for LoRAS oversampling",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/narek-davtyan/LoRAS",
    packages=['loras'],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=[
          'scikit-learn', 'pandas', 'numpy'
    ],
)
