import setuptools

with open("README.rst", "r") as fh:
    long_description = fh.read()

print(setuptools.find_packages())

setuptools.setup(
    name="tomotk", # Replace with your own username
    version="0.0.1",
    author="Mark Wolfman",
    author_email="wolfman@anl.gov",
    description="Tools for working with reconstructed tomography data.",
    long_description=long_description,
    long_description_content_type="text/x-rst",
    url="",
    packages=setuptools.find_packages(),
    classifiers=[
        # "Programming Language :: Python :: 3",
        # "License :: OSI Approved :: MIT License",
        # "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
