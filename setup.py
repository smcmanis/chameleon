from setuptools import setup, find_packages


setup(
    name="chameleon",
    packages=find_packages(),
    package_dir={'chameleon': 'chameleon'},
    entry_points={
        "console_scripts": [
            "chameleon-data = chameleon.scripts.data:cli",
            "chameleon = chameleon.scripts.pipe:cli"
        ]
    },
    install_requires=[
        "click>=7.1.1",
        "numpy>=1.18.1",
        "pandas>=1.0.1",
        "scikit-learn>=0.22.1"
    ]
)

