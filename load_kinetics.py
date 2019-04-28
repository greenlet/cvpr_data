import argparse
import cv2
from datetime import datetime
import json
from multiprocessing import Process, Queue, current_process
import numpy as np
import os
import pandas as pd
import queue
import re
import shutil
import signal
import subprocess as sbpr
import sys
import traceback
import time
import urllib
import youtube_dl
from youtube_dl import DownloadError

import utils


KIN_VERSION = 600
KIN_DIR_NAME = 'Kinetics_{}'.format(KIN_VERSION)
KIN_PARAMS = {
  'train': {
    'arch_url': 'https://deepmind.com/documents/193/kinetics_600_train%20(1).zip',
    'arch_name': 'kinetics_train.zip',
    'csv_name': 'kinetics_train.csv',
  },
  'val': {
    'arch_url': 'https://deepmind.com/documents/194/kinetics_600_val%20(1).zip',
    'arch_name': 'kinetics_val.zip',
    'csv_name': 'kinetics_val.csv',
  },
  'test': {
    'arch_url': 'https://deepmind.com/documents/232/kinetics_600_test%20(2).zip',
    'arch_name': 'kinetics_test.zip',
    'csv_name': 'kinetics_600_test.csv',
  },
  'holdout_test': {
    'arch_url': 'https://deepmind.com/documents/194/kinetics_600_val%20(1).zip',
    'arch_name': 'https://deepmind.com/documents/231/kinetics_600_holdout_test.zip',
    'csv_name': 'kinetics_600_holdout_test.csv',
  },
  'readme': 'https://deepmind.com/documents/197/kinetics_600_readme%20(1).txt',
}

VIDEO_FILE_REGEXP = re.compile(r'^(.{11})_(\d+)_(\d+)\.mp4$')
ADM_STOP_NOW = 'ADM_STOP_NOW'
ADM_FINISH_TASKS_AND_STOP = 'ADM_FINISH_TASKS_AND_STOP'
CMD_YDL_EXTRACT_INFO = 'CMD_YDL_EXTRACT_INFO'
CMD_YDL_DOWNLOAD = 'CMD_YDL_DOWNLOAD'
STATUS_EXEC_SUCCESS = 'STATUS_EXEC_SUCCESS'
STATUS_EXEC_ERROR = 'STATUS_EXEC_ERROR'
STATUS_EXEC_EXCEPTION = 'STATUS_EXCEPTION'
STATUS_EXEC_TIMEOUT = 'STATUS_EXEC_TIMEOUT'


def read_dataframe(out_path, split):
  arch_url = KIN_PARAMS[split]['arch_url']
  arch_path = os.path.join(out_path, KIN_PARAMS[split]['arch_name'])
  csv_name = KIN_PARAMS[split]['csv_name']
  csv_path = os.path.join(out_path, csv_name)
  if utils.download_file(arch_url, arch_path):
    utils.extract(arch_path, out_path, csv_name)
  df = pd.read_csv(csv_path)
  if 'label' in df.columns:
    df['label'] = df['label'].astype('category')
  return df


def save_labels(df, out_path):
  if 'label' in df.columns:
    file_path = os.path.join(out_path, 'labels.txt')
    if not os.path.exists(file_path):
      labels = sorted(df['label'].cat.categories.tolist())
      f = open(file_path, 'w+')
      f.write('\n'.join(labels))


def make_directories(out_path, split, df):
  tmp_path = os.path.join(out_path, 'tmp')
  utils.make_dir(tmp_path)
  utils.clear_dir(tmp_path)
  split_path = os.path.join(out_path, split)
  utils.make_dir(split_path)
  if 'label' in df.columns:
    label_cat = df.label.astype('category')
    for label in label_cat.cat.categories:
      label_path = os.path.join(split_path, label)
      utils.make_dir(label_path)
  return tmp_path, split_path


def get_failed_video_ids(out_path, split, level):
  file_name_fmt = '{}_{}_{}.txt'
  ids_file_name = file_name_fmt.format(split, level, 'ids')
  log_file_name = file_name_fmt.format(split, level, 'log')
  ids_file_path = os.path.join(out_path, ids_file_name)
  log_file_path = os.path.join(out_path, log_file_name)
  ids_file = open(ids_file_path, 'a+', encoding='utf-8')
  log_file = open(log_file_path, 'a+', encoding='utf-8')
  ids_file.seek(0)
  ids = set([l.rstrip() for l in ids_file.readlines()])
  return ids, ids_file, log_file


def filter_failed_video_ids(out_path, split, df):
  error_ids, error_ids_file, error_log_file = get_failed_video_ids(out_path, split, 'error')
  warn_ids, warn_ids_file, warn_log_file = get_failed_video_ids(out_path, split, 'warn')
  print('Error video ids: {}'.format(len(error_ids)))
  print('Warn video ids: {}'.format(len(warn_ids)))
  failed_ids = error_ids.union(warn_ids)
  if len(failed_ids):
    n_before = df.shape[0]
    df = df[~df.youtube_id.isin(failed_ids)]
    n_after = df.shape[0]
    print('Dataset size: {} --> {}'.format(n_before, n_after))
  return failed_ids, error_ids_file, error_log_file, warn_ids_file, warn_log_file, df


def get_loaded_pairs(path):
  res = {}
  n = 0
  for rootdir, subdirs, fnames in os.walk(path):
    for fname in fnames:
      m = VIDEO_FILE_REGEXP.match(fname)
      if m:
        video_id, time_start  = m.group(1), int(m.group(2))
        res.setdefault(video_id, []).append(time_start)
        n += 1
  return res, n


def filter_loaded_pairs(split_path, df):
  loaded, n_loaded = get_loaded_pairs(split_path)
  print('Loaded files: {}'.format(n_loaded))
  if n_loaded:
    df1 = df[['youtube_id', 'time_start']]
    n = df1.shape[0]
    cond_loaded = np.zeros((n,), dtype=np.bool)
    vals = df1.values
    for i in range(n):
      if vals[i, 0] in loaded and vals[i, 1] in loaded[vals[i, 0]]:
        cond_loaded[i] = True
    n_before = df.shape[0]
    df = df[~cond_loaded]
    n_after = df.shape[0]
    print('Dataset size: {} --> {}'.format(n_before, n_after))
  return df


def task_iterator(tmp_path, split_path, df, batch_size):
  label = ''
  has_label = 'label' in df.columns
  fout_path = split_path
  batch = []
  for i in df.index:
    if has_label:
      label = df.loc[i, 'label']
      fout_path = os.path.join(split_path, label)
    youtube_id, time_start, time_end = df.loc[i, 'youtube_id'], df.loc[i, 'time_start'], df.loc[i, 'time_end']
    batch.append((youtube_id, time_start, time_end, fout_path, tmp_path))
    if len(batch) == batch_size:
      yield batch
      batch = []
  if len(batch):
    yield batch


def check_queue(q, timeout=1):
    try:
      if timeout == 0:
        res = q.get(False)
      else:
        res = q.get(True, timeout)
      return res
    except queue.Empty:
      return None


def pprint(*args, **kwargs):
  print('[{}]'.format(current_process().name), *args, **kwargs)


class ExecError(Exception):
  def __init__(self, status, cmd, output):
    super().__init__('{} {} {}'.format(status, cmd, output))
    self.status = status
    self.cmd = cmd
    self.output = output


def exec_ydl_cmd(ydl, cmd, url):
  pprint('{} {}'.format(cmd, url))
  try:
    if cmd == CMD_YDL_DOWNLOAD:
      return ydl.download([url])
    elif cmd == CMD_YDL_EXTRACT_INFO:
      return ydl.extract_info(url, download=False)
  except DownloadError as err:
    raise ExecError(STATUS_EXEC_ERROR, cmd, str(err)) 


def exec_shell_cmd(cmd, timeout=40, attempts=1, noexcept=False):
  if type(cmd) != str:
    cmd = ' '.join(cmd)
  args = {}
  if timeout:
    args['timeout'] = timeout
  output = ''
  for i in range(attempts):
    try:
      pprint(cmd)
      res = sbpr.run(cmd, stdout=sbpr.PIPE, stderr=sbpr.PIPE, encoding='utf-8', **args)
      if res.returncode == 0:
        output = (res.stdout or '').strip()
        if noexcept:
          return STATUS_EXEC_SUCCESS, cmd, output
        return output
      # Error occured
      output = ''
      if res.stderr:
        output = res.stderr.strip()
      else:
        output = 'Exited with code: {}'.format(res.returncode)
      if noexcept:
        return STATUS_EXEC_ERROR, cmd, output
      raise ExecError(STATUS_EXEC_ERROR, cmd, output)
    except sbpr.TimeoutExpired as err:
      pprint('Attempt #{}. Timeout reached {}'.format(i, timeout))
      output += err.output
  if noexcept:
    return STATUS_EXEC_TIMEOUT, cmd, output
  raise ExecError(STATUS_EXEC_TIMEOUT, cmd, output)


def get_ydl_url(ydl_info):
  formats = {fmt['format_id']:fmt for fmt in ydl_info['formats']}
  if '18' in formats:
    return formats['18']['url']
  raise ExecError(STATUS_EXEC_ERROR, 'get_ydl_url', json.dumps(formats))


def is_valid_video(file_path):
  cap = cv2.VideoCapture(file_path)
  width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
  height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
  frames_num = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
  return width > 0 and height > 0 and frames_num > 1


def move_video(tmp_file_path, out_file_path):
  if not os.path.exists(tmp_file_path):
    raise ExecError(STATUS_EXEC_ERROR, 'copy_video', 'File "{}" not found'.format(tmp_file_path))
  if not is_valid_video(tmp_file_path):
    raise ExecError(STATUS_EXEC_ERROR, 'copy_video', 'File "{}" is not valid video'.format(tmp_file_path))
  shutil.move(tmp_file_path, out_file_path)


def exec_task(task):
  youtube_id, time_start, time_end, fout_path, tmp_path = task
  file_name_tmpl = '%(id)s_{:06}_{:06}.mp4'.format(time_start, time_end)
  ydl_opts = {
    'format': '18',
    'outtmpl': os.path.join(tmp_path, file_name_tmpl),
    'quiet': True,
    'no_warnings': True,
    'socket_timeout': 40,
    'retries': 1,
  }
  ydl = youtube_dl.YoutubeDL(ydl_opts)
  
  url = 'https://www.youtube.com/watch?v={}'.format(youtube_id)
  ydl_info = exec_ydl_cmd(ydl, CMD_YDL_EXTRACT_INFO, url)

  file_name = file_name_tmpl % {'id': youtube_id}
  tmp_file_path = os.path.join(tmp_path, file_name)
  out_file_path = os.path.join(fout_path, file_name)

  def make_ffmpeg_cmd(src_path, dst_path):
    return [
      'ffmpeg -y', # Overwrite
      '-ss {}'.format(time_start),
      '-t {}'.format(time_end - time_start),
      '-i "{}"'.format(src_path),
      '-c:v libx264 -preset ultrafast',
      '-c:a aac',
      '-threads 1 -loglevel panic',
      '"{}"'.format(dst_path)
    ]

  if ydl_info['duration'] <= 10:
    pprint('Duration: {}'.format(ydl_info['duration']))
    exec_ydl_cmd(ydl, CMD_YDL_DOWNLOAD, url)
  else:
    ydl_url = get_ydl_url(ydl_info)
    ffmpeg_cmd = make_ffmpeg_cmd(ydl_url, tmp_file_path)
    status, cmd, output = exec_shell_cmd(ffmpeg_cmd, noexcept=True)
    # Sometimes FFMPEG fails downloading from valid URL
    faulty = False
    if status != STATUS_EXEC_SUCCESS or not is_valid_video(tmp_file_path):
      if status != STATUS_EXEC_SUCCESS:
        pprint('{}. {}\n\tOuput:{}'.format(status, cmd, output))
      exec_ydl_cmd(ydl, CMD_YDL_DOWNLOAD, url)
      tmp_long_file_path = os.path.join(tmp_path, 'long_{}'.format(file_name))
      shutil.move(tmp_file_path, tmp_long_file_path)
      ffmpeg_cmd = make_ffmpeg_cmd(tmp_long_file_path, tmp_file_path)
      exec_shell_cmd(ffmpeg_cmd)
      os.remove(tmp_long_file_path)

  move_video(tmp_file_path, out_file_path)


def download(queue_adm, queue_in, queue_out):
  signal.signal(signal.SIGINT, signal.SIG_IGN)
  process_name = current_process().name

  def _exec():
    task = check_queue(queue_in)
    if task:
      youtube_id = task[0]
      try:
        exec_task(task)
        res = (process_name, STATUS_EXEC_SUCCESS, youtube_id, None, None)
      except ExecError as ex:
        res = (process_name, ex.status, youtube_id, ex.cmd, ex.output)
      except Exception as ex:
        res = (process_name, STATUS_EXEC_EXCEPTION, youtube_id, None, traceback.format_exc())
      queue_out.put_nowait(res)
      return res

  try:
    do_not_stop, res = True, None
    while do_not_stop or res:
      res = _exec()
      command = check_queue(queue_adm, 0)
      if command:
        pprint(command)
        if command == ADM_STOP_NOW:
          # sys.exit(0)
          os._exit(0)
        elif command == ADM_FINISH_TASKS_AND_STOP:
          do_not_stop = False
  except Exception as ex:
    traceback.print_exc()
    sys.exit(0)


def get_dataset(out_path, split, num_jobs):
  print('Get "{}" dataset into {}'.format(split, out_path))
  
  # Read CSV into Dataframe
  df = read_dataframe(out_path, split)
  save_labels(df, out_path)

  # Make subdirectories
  tmp_path, split_path = make_directories(out_path, split, df)
  
  # Filter previous failed attempts out
  failed_ids, ferr_ids, ferr_log, fwarn_ids, fwarn_log, df = filter_failed_video_ids(out_path, split, df)

  # Filter already loaded pairs (youtube_id, time_start) out
  df = filter_loaded_pairs(split_path, df)
  # df = df[:20]
  N = df.shape[0]
  print('Tasks total:', N)

  # Starting parallel workers
  pool = []
  queue_in, queue_out = Queue(), Queue()
  queues_adm = []
  print('Starting processes')
  for i in range(num_jobs):
    queue_adm = Queue()
    queues_adm.append(queue_adm)
    process = Process(target=download, args=(queue_adm, queue_in, queue_out))
    process.start()
    pool.append(process)

  process_exit = False

  def sig_handler(sig, frame):
    print('Smooth exit on Ctrl+C')
    nonlocal process_exit
    process_exit = True
  signal.signal(signal.SIGINT, sig_handler)

  def is_blocked_or_unavailable(message):
    if 'video is unavailable' in message or \
        'is no longer available' in message or \
        'is not available' in message or \
        'who has blocked it' in message or \
        'have blocked it' in message or \
        'video has been removed' in message or \
        'account associated with this video has been terminated' in message:
      return True
    
    return False

  def put_line(fid, s):
    fid.write('{}\n'.format(s))
    fid.flush()
  
  def put_lines(youtube_id, log_line, is_err):
    file_ids, file_log = (ferr_ids, ferr_log) if is_err else (fwarn_ids, fwarn_log)
    put_line(file_ids, youtube_id)
    put_line(file_log, log_line)

  def send_adm_command_and_close(cmd):
    for i in range(num_jobs):
      print('Job {:02}. Send {} and close adm queue'.format(i, cmd))
      queues_adm[i].put(cmd)
      queues_adm[i].close()
    for i in range(num_jobs):
      queues_adm[i].join_thread()
    # queue_in.close()
    # queue_out.close()

  try:
    batch_size = num_jobs
    task_it = task_iterator(tmp_path, split_path, df, batch_size)
    n_sent, n_received = 0, 0

    def receive(n=None, strict=False):
      nonlocal n_received
      n = n if n else n_sent - n_received
      i = 0
      while i < n:
        if process_exit:
          return
        res = check_queue(queue_out)
        if res:
          print('Task res:', res)
          process_name, status, youtube_id, cmd, output = res
          log_line = '[{}] {} {} {}\n\tOutput: {}'.format(process_name, status, youtube_id, cmd, output or None)
          print('Result. {}'.format(log_line))
          if status != STATUS_EXEC_SUCCESS:
            failed_ids.add(youtube_id)
            is_err = status == STATUS_EXEC_ERROR and cmd == CMD_YDL_EXTRACT_INFO and \
              is_blocked_or_unavailable(output)
            put_lines(youtube_id, log_line, is_err)
          n_received += 1
          i += 1
        elif not strict and queue_in.qsize() <= num_jobs:
          return

    for batch in task_it:
      for task in batch:
        if task[0] not in failed_ids:
          queue_in.put_nowait(task)
          n_sent +=1
      if n_sent > batch_size:
        receive(batch_size)

    send_adm_command_and_close(ADM_FINISH_TASKS_AND_STOP)

    # Receiving rest so queue_out will be empty and can safely join worker processes
    receive(None, True)
  except Exception as e:
    if not process_exit:
      traceback.print_exc()
    send_adm_command_and_close(ADM_STOP_NOW)


  print('Joining processes')
  for process in pool:
    process.join()

  signal.signal(signal.SIGINT, signal.SIG_DFL)


def run(opt):
  t1 = datetime.now()

  out_path = os.path.join(opt.data_path, KIN_DIR_NAME)
  splits = [opt.split] if opt.split else ['train', 'val', 'test']
  for split in splits:
    get_dataset(out_path, split, opt.num_jobs)

  print('\nProcessing time: {}'.format(datetime.now() - t1))


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Kinetics {} dataset loader'.format(KIN_VERSION))
  parser.add_argument('--data-path', type=str, required=True,
    help='Data directory path. All new data will be put in subdirectory DATA_PATH/{}'.format(KIN_DIR_NAME))
  parser.add_argument('--split', type=str, choices=['train', 'val', 'test', 'holdout_test'],
    help='Dataset split type. If not chosen, "train", then "val", then "test" sets will be loaded. \
      Downloaded files will be put into subdirectories "train", "val", "test", "holdout_test" respectively')
  parser.add_argument('--num-jobs', type=int, default=12,
    help='Number of simultaneous downloading processes')
  opt = parser.parse_args()

  run(opt)

