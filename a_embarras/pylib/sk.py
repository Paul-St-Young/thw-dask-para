import numpy as np

def legal_kvecs(nx, box):
  ndim = len(box)
  nlist0 = np.arange(nx+1)
  nlist = np.arange(-nx, nx+1)
  nX, nY, nZ = np.meshgrid(nlist0, nlist, nlist, indexing='ij')
  kvecs = np.stack((nX, nY, nZ), axis=-1).reshape((-1, 3)).astype(float)
  for idim in range(ndim):
    kvecs[:, idim] *= 2*np.pi/box[idim]
  kmags = np.linalg.norm(kvecs, axis=-1)
  sel = kmags > 0
  return kvecs[sel]

def rhok(kvecs, pos):
  if kvecs.ndim == 1:
    kvecs = kvecs[np.newaxis, :]
  # dot the last axis of kvecs and pos
  exponentials = np.exp(-1j*np.inner(kvecs, pos))
  # sum over position index
  rhok = np.sum(exponentials, axis=-1)
  return rhok

def Sk(kvecs, pos):
  natom = pos.shape[-2]
  rho = rhok(kvecs, pos)
  sk = rho*rho.conj()/natom
  if sk.ndim >= 2:
    sk = sk.mean(axis=-1)
  return sk.real

def kshell_sels(kmags, zoom):
  kints = np.round(kmags*zoom).astype(int)
  unique_kints = np.unique(kints)
  nsh = len(unique_kints)
  sels = []
  for ish in range(nsh):
    kint = unique_kints[ish]  # shell integer label
    sel = kints == kint       # select this shell
    sels.append(sel)
  return sels

def shavg(kvecs, dskm, dske, zoom=100.):
  """ Shell average S(k), including error bar
  hint: if your S(k) data has no error, then pass in dske=np.zeros(nk).

  Args:
    kvecs (np.array): kvectors, shape (nk, ndim)
    sk (np.array): S(k), shape (nk,)
    zoom (float, optional): control resolution of kshells,
     higher zoom will result in more kshells, default is 100.
  Return:
    (np.array, np.array, np.array): (uk, uskm, uske), shell-averaged k, S(k)
     mean and S(k) error
  """
  # determine kshells by rounding kvecs
  kmags = np.linalg.norm(kvecs, axis=-1)
  sels = kshell_sels(kmags, zoom)
  nsh = len(sels)
  # loop over each shell and average
  uk = np.zeros(nsh)
  uskm = np.zeros(nsh)
  uske = np.zeros(nsh)
  for ish, sel in enumerate(sels):
    uk[ish] = np.mean(kmags[sel])
    uskm[ish] = np.mean(dskm[sel])
    uske[ish] = np.sqrt(np.sum(dske[sel]**2))/len(dske[sel])
  return uk, uskm, uske
