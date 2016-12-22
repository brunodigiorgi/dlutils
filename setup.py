from setuptools import setup, find_packages

install_requires = [
    'numpy',
]

tests_require = [
    'pytest',
]

setup(name='dlutils',
      version='0.1',
      description='Deep Learning Utilities',
      author="Bruno Di Giorgi",
      author_email='bruno@brunodigiorgi.it',
      url='https://github.com/brunodigiorgi/dlutils',
      license="GPLv2",
      packages=find_packages(),
      package_data={'': ['*.html_template']},
      zip_safe=False,
      install_requires=install_requires,
      extras_require={
          'testing': tests_require,
      },
      )
