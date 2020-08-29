import setuptools
from blackbox import version

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="blackbox",
    version=version.__version__,
    author="Gabriel Martín Blázquez",
    author_email="gmartin_b@usal.es",
    description="A package that implements several algorithms to detect anomalies",
    long_description_content_type="text/markdown",
    url="https://github.com/gabrielmbmb/intry4.0-backend-anomaly",
    packages=setuptools.find_packages(),
    data_files=[],
    classifiers=[
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Operating System :: OS Independent",
        "Intended Audience :: Developers",
        "Framework :: Flask",
    ],
    python_requires=">=3.6",
    keywords="anomaly-detection",
    project_urls={
        "Bug Reports": "https://github.com/gabrielmbmb/intry4.0-backend-anomalyissues",
        "Source": "https://github.com/gabrielmbmb/intry4.0-backend-anomaly",
    },
)
