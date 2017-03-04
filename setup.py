from setuptools import setup, find_packages

config = {
    'name': 'deep_rf',
    'version': '0.0.1',
    'url': 'https://github.com/solstat/deep-rf',
    'description': 'A Python module for deep reinforcement learning with Tensorflow.',
    'author': 'Solstat',
    'license': 'BSD',
    'packages': find_packages()
}

setup(**config)
