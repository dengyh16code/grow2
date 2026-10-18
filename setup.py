from setuptools import setup, find_packages

setup(
    name="improvised_tooluse",
    version="0.1.0",
    packages=find_packages(),
    # config.yaml is deliberately not packaged -- it lives outside the package
    # and is passed in by the entry points (see improvised_tooluse/config.py).
    package_data={"improvised_tooluse": ["prompt/*.txt"]},
    install_requires=["pyyaml"],
)
