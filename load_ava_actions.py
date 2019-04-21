import os
import argparse
import urllib.request
import subprocess
import re
from datetime import datetime
import utils


CUSTOM_VIDEO_ID_REGEX = re.compile(r'^v_(.+)\.(\w+)$')
CVDF_VIDEO_ID_REGEX = re.compile(r'<Key>(trainval|test)/([a-zA-Z0-9\-_]+)\.(\w+)</Key>')


def get_ids(path):
  file_names = os.listdir(path)
  ids = set()
  for fname in file_names:
    m = CUSTOM_VIDEO_ID_REGEX.match(fname)
    if m:
      ids.add(m.group(1))
  return ids


def read_ids(file_path):
  is_csv = file_path.endswith('.csv')
  with open(file_path) as f:
    lines = f.readlines()
    video_ids = set()
    for line in lines:
      if is_csv:
        video_id = line.split(',')[0]
      else:
        video_id = line.rstrip()
      video_ids.add(video_id)
    return video_ids


def load_lists(opt):
  arch_name = 'ava_v{}.zip'.format(opt.version)
  arch_path = os.path.join(opt.out_path, arch_name)
  arch_url = 'https://research.google.com/ava/download/{}'.format(arch_name)
  if utils.download_file(arch_url, arch_path):
    utils.extract(arch_path, opt.out_path)

  train_video_ids, val_video_ids, test_video_ids = None, None, None
  if opt.type is None or opt.type == 'train':
    ids_file_path = os.path.join(opt.out_path, 'ava_train_v{}.csv'.format(opt.version))
    train_video_ids = read_ids(ids_file_path)
  if opt.type is None or opt.type == 'validation':
    ids_file_path = os.path.join(opt.out_path, 'ava_val_v{}.csv'.format(opt.version))
    val_video_ids = read_ids(ids_file_path)
  if opt.type is None or opt.type == 'test':
    ids_file_path = os.path.join(opt.out_path, 'ava_test_v{}.txt'.format(opt.version))
    test_video_ids = read_ids(ids_file_path)

  ts_file_name = 'ava_included_timestamps_v{}.txt'.format(opt.version)
  ts_file_path = os.path.join(opt.out_path, ts_file_name)
  with open(ts_file_path) as f:
    lines = f.readlines()
    timestamps = int(lines[0]), int(lines[-1])

  return train_video_ids, val_video_ids, test_video_ids, timestamps


def load_cvdf_list(opt):
  base_url = 'https://s3.amazonaws.com/ava-dataset'
  print('Loading files list from {}'.format(base_url))
  xml_path = os.path.join(opt.out_path, 'cvdf.xml')
  res = {}
  with urllib.request.urlopen(base_url) as response, open(xml_path, 'w+') as fout:
    xml_str = response.read().decode('utf-8')
    matches = CVDF_VIDEO_ID_REGEX.findall(xml_str)
    for m in matches:
      res[m[1]] = {
        'url': '{}/{}/{}.{}'.format(base_url, m[0], m[1], m[2]),
        'ext': m[2]
      }
    fout.write(xml_str)
  return res


def rewrite_timestamps_file(opt, name_base, delta):
  def decrement(ts_str):
    ts = int(ts_str)
    ts -= delta
    return '{:04}'.format(ts)

  file_name_in = name_base.format('ava_', '_v{}'.format(opt.version))
  file_name_out = name_base.format('short_', '')
  path_in = os.path.join(opt.out_path, file_name_in)
  path_out = os.path.join(opt.out_path, file_name_out)
  if os.path.exists(path_out):
    return

  print('{} --> {}'.format(file_name_in, file_name_out))
  with open(path_in) as fin, open(path_out, 'w+') as fout:
    for line_in in fin.readlines():
      parts = line_in.strip().split(',')
      if len(parts) == 1:
        line_out = decrement(parts[0])
      else:
        parts[1] = decrement(parts[1])
        line_out = ','.join(parts)
      fout.write(line_out + '\n')


def rewrite_timestamps(opt, delta):
  rewrite_timestamps_file(opt, '{}included_timestamps{}.txt', delta)
  rewrite_timestamps_file(opt, '{}train{}.csv', delta)
  rewrite_timestamps_file(opt, '{}train_excluded_timestamps{}.csv', delta)
  rewrite_timestamps_file(opt, '{}val{}.csv', delta)
  rewrite_timestamps_file(opt, '{}val_excluded_timestamps{}.csv', delta)
  rewrite_timestamps_file(opt, '{}test_excluded_timestamps{}.csv', delta)


def load_files(opt, ids_to_load, data_type, timestamps, cvdf_id2info, log_file):
  out_dir = src_dir = os.path.join(opt.out_path, 'src_{}'.format(data_type))
  utils.make_dir(src_dir)
  if opt.shorten:
    # Each timestamp is a middle of a 3-second short action
    # located at [timestamp - 1.5, timestamp + 1.5] period
    ts_start = timestamps[0] - 1.5
    ts_stop = 1.5 + (timestamps[1] - timestamps[0]) + 1.5
    shorten_cmd_fmt = 'ffmpeg -loglevel warning -ss {} -i {{}} -t {} -c copy {{}}'.format(
      ts_start, ts_stop)
    out_dir = short_dir = os.path.join(opt.out_path, 'short_{}'.format(data_type))
    utils.make_dir(short_dir)
  file_names = os.listdir(out_dir)
  # Removing 'v_' prefix and '.mp4' suffix
  ids_to_load -= get_ids(out_dir)
  if not len(ids_to_load):
    print('Nothing to load into {}'.format(out_dir))
    return ids_to_load
  print('Start loading {} files into {}'.format(len(ids_to_load), out_dir))

  def process(cmd):
    print(cmd)
    log_file.write(cmd + '\n')
    log_file.flush()
    res = subprocess.run(cmd, stderr=log_file, shell=True)
    print('--> ' + ('FAIL' if res.returncode else 'SUCCESS'))
    return res

  for video_id in sorted(ids_to_load):
    ytb_file_name = 'v_{}.mp4'.format(video_id)
    ytb_file_path = os.path.join(src_dir, ytb_file_name)
    ytb_cmd = 'youtube-dl -f best -f mp4 "https://youtube.com/watch?v={}" -o "{}"'.format(
      video_id, ytb_file_path)
    
    cvdf_file_path = None
    if video_id in cvdf_id2info:
      cvdf_file_name = 'v_{}.{}'.format(video_id, cvdf_id2info[video_id]['ext'])
      cvdf_file_path = os.path.join(src_dir, cvdf_file_name)
      cvdf_cmd = 'youtube-dl "{}" -o "{}"'.format(
        cvdf_id2info[video_id]['url'], cvdf_file_path)

    if not os.path.exists(ytb_file_path):
      if cvdf_file_path:
        if not os.path.exists(cvdf_file_path):
          process(ytb_cmd)
          if not os.path.exists(ytb_file_path):
            process(cvdf_cmd)
      else:
        process(ytb_cmd)

    src_file_path = None
    src_file_name = None
    if os.path.exists(ytb_file_path):
      src_file_path, src_file_name = ytb_file_path, ytb_file_name
    elif os.path.exists(cvdf_file_path):
      src_file_path, src_file_name = cvdf_file_path, cvdf_file_name

    if opt.shorten and src_file_path:
      short_file_path = os.path.join(short_dir, src_file_name)
      if not os.path.exists(short_file_path):
        shorten_cmd = shorten_cmd_fmt.format(src_file_path, short_file_path)
        process(shorten_cmd)

  ids_left = ids_to_load - get_ids(out_dir)
  return ids_left


def run(opt):
  t1 = datetime.now()

  opt.out_path = os.path.join(opt.data_path, 'AVA_Actions_v{}'.format(opt.version))
  opt.out_path = utils.make_dir(opt.out_path)
  train_video_ids, val_video_ids, test_video_ids, timestamps = load_lists(opt)
  cvdf_id2info = load_cvdf_list(opt)
  if opt.shorten:
    rewrite_timestamps(opt, timestamps[0] - 1)
  log_file = open(os.path.join(opt.out_path, 'load.log'), 'w+')
  ids_left = set()
  if train_video_ids:
    not_loaded = load_files(opt, train_video_ids, 'train', timestamps, cvdf_id2info, log_file)
    ids_left = ids_left.union(not_loaded)
  if val_video_ids:
    not_loaded = load_files(opt, val_video_ids, 'val', timestamps, cvdf_id2info, log_file)
    ids_left = ids_left.union(not_loaded)
  if test_video_ids:
    not_loaded = load_files(opt, test_video_ids, 'test', timestamps, cvdf_id2info, log_file)
    ids_left = ids_left.union(not_loaded)

  not_loaded_path = os.path.join(opt.out_path, 'not_loaded.txt')
  n_left = len(ids_left)
  if n_left == 0 and os.path.exists(not_loaded_path):
    os.remove(not_loaded_path)
  elif n_left > 0:
    print('There are {} files not loaded. Writing them down to {}'.format(
      n_left, not_loaded_path))
    with open(not_loaded_path, 'w+') as f:
      f.write('\n'.join(sorted(ids_left)))
  
  print('\nProcessing time: {}'.format(datetime.now() - t1))


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='AVA Actions dataset loader')
  parser.add_argument('--version', type=str, default='2.2',
    help='AVA Actions dataset version')
  parser.add_argument('--data-path', type=str, required=True,
    help='Data directory path. All new data will be put in subdirectory DATA_PATH/AVA_Actions_<version-suffix>')
  parser.add_argument('--type', type=str, choices=['train', 'validation', 'test'],
    help='Dataset type. If not chosen, all videos will be loaded. \
      Downloaded files will be put into subdirectories src_train, src_val, src_test respectively')
  parser.add_argument('--shorten', action='store_true',
    help='Boolean flag. If present, indicates whether to shorten source video files. \
      New video files will be generated into subdirectories short_train, short_val, short_test respectively. \
      All CSV, TXT data containing timestamps will be adjusted accordingly and put into files with prefix "short_"')
  opt = parser.parse_args()

  run(opt)


