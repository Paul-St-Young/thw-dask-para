import os
import subprocess as sp

def check_dir_before(mkdir):
  def wrapper(dirname):
    if not os.path.isdir(dirname):
      mkdir(dirname)
  return wrapper

@check_dir_before
def mkdir(x):
  sp.check_call(['mkdir', '-p', x])

def skip_exist_file(write_file):
  def wrapper(fout, *args, **kwargs):
    if not os.path.isfile(fout):
      return write_file(fout, *args, **kwargs)
    else:
      msg = '%s exists' % fout
      print(msg)
  return wrapper

def cache(write_file):
  def wrapper(fout, *args, **kwargs):
    cache_dir = os.path.dirname(fout)
    if cache_dir != '':
      mkdir(cache_dir)
    return skip_exist_file(write_file)(fout, *args, **kwargs)
  return wrapper
