from setuptools import find_packages, setup


setup(name="keras_segmentation",
      version="0.3.0",
      description="Image Segmentation toolkit for keras",
      author="Divam Gupta",
      author_email='divamgupta@gmail.com',
      platforms=["any"],  # or more specific, e.g. "win32", "cygwin", "osx"
      license="MIT",
      url="https://github.com/divamgupta/image-segmentation-keras",
      packages=find_packages(),
      entry_points={
            'console_scripts': [
                  'keras_segmentation = keras_segmentation.__main__:main'
            ]
      }
      )
