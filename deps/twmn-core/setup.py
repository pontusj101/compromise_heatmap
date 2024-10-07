from setuptools import setup, find_packages

setup(
    name='epic_importer',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'docopt',
        'jinja2',
    ],
)
