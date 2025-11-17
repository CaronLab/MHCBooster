from setuptools import setup
from mhcbooster import __version__ as version

def read_requirements(filename="requirements.txt"):
    with open(filename, "r") as f:
        return [line.strip() for line in f.readlines() if line.strip() and not line.startswith("#")]

setup(
    name='mhcbooster',
    version=str(version),
    description='',
    long_description=open('README.md').read(),
    long_description_content_type="text/markdown",
    url='https://github.com/caronlab/mhcbooster',
    author='Ruimin Wang',
    author_email='ruimin.wang@yale.edu',
    entry_points={
        'console_scripts': ['mhcbooster = mhcbooster.interface.mhcbooster_cli:run',
                            'mhcbooster-gui = mhcbooster.interface.mhcbooster_gui:run',
                            'mhcbooster-package-installer = mhcbooster.utils.package_installer:install']
    },
    classifiers=[
        "Programming Language :: Python :: 3.10",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
    ],
    packages=['mhcbooster', 'mhcbooster.adapter', 'mhcbooster.interface', 'mhcbooster.model', 'mhcbooster.predictors',
              'mhcbooster.report', 'mhcbooster.utils',
              'mhcbooster.third_party'],
    python_requires='==3.10.*',
    install_requires=read_requirements(),
    include_package_data=True,
    license='GPL-3.0'
)
