from distutils.core import setup

INSTALL_REQUIRES = [
    "pandas",
    "numpy",
    "pre-commit",
    "voluptuous",
    "pathlib",
    "argparse",
]

setup(
    name="DETR-Tensorflow",
    version="1.0",
    author="Auvisus GmbH",
    packages=["detr_models.detr", "detr_models.backbone", "detr_models.transformer"],
    install_requires=INSTALL_REQUIRES,
)
