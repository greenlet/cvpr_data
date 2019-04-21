import os
import shutil
import urllib.request
import zipfile


def make_dir(*subpaths):
  path = os.path.join(*subpaths)
  path = os.path.abspath(path)
  os.makedirs(path, exist_ok=True)
  return path


def clear_dir(*subpaths):
  root_path = os.path.join(*subpaths)
  for name in os.listdir(root_path):
    df_path = os.path.join(root_path, name)
    if os.path.isfile(df_path):
      os.unlink(df_path)
    else:
      shutil.rmtree(df_path)


def download_file(src_url, dst_path):
  if not os.path.exists(dst_path):
    print('Downloading {} into {}'.format(src_url, dst_path))
    with urllib.request.urlopen(src_url) as response, open(dst_path, 'wb') as out_file:
      shutil.copyfileobj(response, out_file)
      return True
  return False


def extract(arch_path, out_path, file_name=None):
  with zipfile.ZipFile(arch_path) as zip_ref:
    if not file_name:
      zip_ref.extractall(out_path)
    else:
      zip_ref.extract(file_name, out_path)

