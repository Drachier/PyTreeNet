"""
Some general constants for the project, such as file paths.
"""
from os import path

DIRECTORY = path.dirname(path.abspath(__file__))
REPO_ROOT = path.dirname(DIRECTORY)
CONFIG_FILE = path.join(REPO_ROOT, 'config.ini')
