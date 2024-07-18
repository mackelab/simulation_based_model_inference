from setuptools import setup

setup(
    name="GrassmannBinaryDistribution",  # project name
    version="0.0",
    description="Simulation-based model inference",
    url="https://github.com/mackelab/simulation_based_model_inference",
    author="Cornelius Schroeder",
    author_email="cornelius.schroeder@uni-tuebingen.de",
    license="MIT",
    packages=["sbmi"],  # actual package name (to import package)
    zip_safe=False,
)
