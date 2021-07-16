from setuptools import setup, find_packages
import os

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="tf-image",
    version="0.2.0",
    description="Image augmentation operations for TensorFlow 2+.",
    url="https://github.com/Ximilar-com/tf-image",
    author="Ximilar.com Team, Michal Lukac, Libor Vanek, ...",
    author_email="tech@ximilar.com",
    license="MIT",
    packages=find_packages(),
    keywords="machine learning, multimedia, image",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.6",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    include_package_data=True,
    zip_safe=False,
    namespace_packages=["tf_image"],
    long_description=long_description,
    long_description_content_type="text/markdown",
)
