#!/usr/bin/env python3
import numpy as np
from pylib.sugar import cache
from pylib.sk import legal_kvecs, Sk, shavg

@cache
def cache_sofk(fout, fjson, Sk, nx, verbose=True, para=True):
  if para:
    import dask
    from dask.distributed import Client, progress
    client = Client(processes=False)
    Sk = dask.delayed(Sk)

  import pandas as pd
  mdf = pd.read_json(fjson)
  box = mdf.iloc[0]['box']
  kvecs = legal_kvecs(nx, box)
  nk = len(kvecs)

  if verbose and (not para):
    from progressbar import ProgressBar, Bar, ETA
    widgets = [Bar('>'), ETA()]
    bar = ProgressBar(widgets=widgets, maxval=len(mdf))
    bar.start()

  skl = []
  sk2l = []
  isk = 0
  for label, row in mdf.iterrows():
    com = np.array(row['positions'])
    sk = Sk(kvecs, com)
    skl.append(sk)
    sk2l.append(sk**2)

    isk += 1
    if verbose and (not para):
      bar.update(isk)
  skm = np.mean(skl, axis=0)
  ske = (np.mean(sk2l, axis=0)-skm**2)**0.5/len(skl)**0.5

  if para:
    skm, ske = dask.persist(skm, ske)
    if verbose:
      progress(skm, ske)
    skm, ske = dask.compute(skm, ske)
    client.shutdown()

  # spherical average
  uk, uskm, uske = shavg(kvecs, skm, ske)
  np.savetxt(fout, np.array([uk, uskm, uske]).T)

if __name__ == '__main__':
  from argparse import ArgumentParser
  parser = ArgumentParser()
  parser.add_argument('fjson',
    help='database containing ["box", "positions"]')
  args = parser.add_argument('--nx', type=int, default=14,
    help='number of shells to use along each direction')
  args = parser.add_argument('--serial', action='store_true',
    help='run in serial, i.e. disable dask parallelization')
  args = parser.add_argument('--no_progress', action='store_false',
    help='show no progress')
  args = parser.parse_args()
  para = not args.serial
  verbose = ~args.no_progress
  if verbose:
    msg = 'executing in paralell: %s' % para
    print(msg)

  fjson = args.fjson
  fdat = fjson.replace('-h2.json', '-h2sk.dat')
  if verbose:
    from time import time
    tstart = time()
  cache_sofk(fdat, fjson, Sk, verbose=verbose, nx=args.nx, para=para)
  if verbose:
    tend = time()
    telapsed = tend - tstart
    msg = 'finished in %8.6f s' % telapsed
    print(msg)
# end __main__
