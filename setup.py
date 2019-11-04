import setuptools
from blackbox import version
with open('README.md', 'r') as fh:
    long_description = fh.read()

setuptools.setup(
    name="platinum-anomaly-detection",
    version=version.__version__,
    author="Gabriel Martín Blázquez",
    author_email="gmartin_b@usal.es",
    description="A package that implements several algorithms to detect anomalies",
    long_description_content_type="text/markdown",
    url="https://github.com/bisite/PLATINUM-anomaly-detection",
    packages=setuptools.find_packages(),
    data_files=[],
    entry_points={
        'console_scripts': [
            'blackbox = blackbox.api.api:run_api',
            'blackbox-tools = tools.blackbox_tools:tools'
        ]
    },
    classifiers=[
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Operating System :: OS Independent",
        "Intended Audience :: Developers",
        "Framework :: Flask",
    ],
    python_requires='>=3.6',
    keywords='PLATINUM, anomaly-detection',
    project_urls={
        'Bug Reports': 'https://github.com/bisite/PLATINUM-anomaly-detection/issues',
        'Source': 'https://github.com/bisite/PLATINUM-anomaly-detection',
    }
)