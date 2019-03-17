import os


def make_dir(*subpaths):
  path = os.path.join(*subpaths)
  path = os.path.abspath(path)
  os.makedirs(path, exist_ok=True)
  return path
