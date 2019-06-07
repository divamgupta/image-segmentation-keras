from setuptools import find_packages, setup


setup(name="keras_segmentation",
      version="0.2.0",
      description="Image Segmentation toolkit for keras",
      author="Divam Gupta",
      author_email='divamgupta@gmail.com',
      platforms=["any"],  # or more specific, e.g. "win32", "cygwin", "osx"
      license="MIT",
      url="https://github.com/divamgupta/image-segmentation-keras",
      packages=find_packages(),
)