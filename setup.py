from setuptools import find_namespace_packages, setup

setup(
    name="heatmap",
    version="0.1",
    packages=find_namespace_packages(where=".", include="heatmap.*"),
    install_requires=[],
    entry_points={
        "console_scripts": [
            "heatmap=heatmap.__main__:main",
        ],
    },
)
