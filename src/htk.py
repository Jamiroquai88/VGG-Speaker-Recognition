import re
import struct
import numpy as np

WAVEFORM = 0
LPC = 1
LPCREFC = 2
LPCEPSTRA = 3
LPCDELCEP = 4
IREFC = 5
MFCC = 6
FBANK = 7
MELSPEC = 8
USER = 9
DISCRETE = 10
PLP = 11
ANON = 12

_E = 100  # has energy
_N = 200  # absolute energy supressed
_D = 400  # has delta coefficients
_A = 1000  # has acceleration coefficients
_C = 2000  # is compressed
_Z = 4000  # has zero mean static coef.
_K = 10000  # has CRC checksum
_0 = 20000  # has 0th cepstral coef.
_V = 40000  # has VQ data
_T = 100000  # has third differential coef.

parms16bit = [WAVEFORM, IREFC, DISCRETE]

# BSAPI
HTK_HEADER_SIZE = 12
HTK_EXHEADER_SIZE = 12
HTK_HAS_EXT_HEADER = -1
SIZEOF_FLOAT = 4


def readhtk(file, return_parmKind_and_sampPeriod=False):
    """ Read htk feature file
     Input:
         file: file name or file-like object.
     Outputs:
          m  - data: column vector for waveforms, one row per frame for other types
          sampPeriod - frame rate [seconds]
          parmKind
    """
    try:
        fh = open(file,'rb')
    except TypeError:
        fh = file
    try:
        nSamples, sampPeriod, sampSize, parmKind = struct.unpack(">IIHH", fh.read(12))
        m = np.frombuffer(fh.read(nSamples*sampSize), 'i1')
        pk = parmKind & 0x3f
        if pk in parms16bit:
            m = m.view('>h').reshape(nSamples,sampSize//2)
        elif parmKind & _C:
            scale, bias = m[:sampSize*4].view('>f').reshape(2,sampSize//2)
            m = (m.view('>h').reshape(nSamples,sampSize//2)[4:] + bias) / scale
        else:
            m = m.view('>f').reshape(nSamples,sampSize//4)
        if pk == IREFC:
            m = m / 32767.0
        if pk == WAVEFORM:
            m = m.ravel()
        if parmKind & _K:
            fh.read(1)
    finally:
        if fh is not file: fh.close()
    return m if not return_parmKind_and_sampPeriod else (m, parmKind, sampPeriod/1e7)


def readhtk_segment(file, start, end, return_parmKind_and_sampPeriod=False):
    """ Read segment from htk feature file
     Input:
         file - file name or file-like object alowing to seek in the file
         start, end - only frames in the range start:end are extracted. 
         If start is negative or when end points behind the end of the feature
         matrix, the first or/and the last frame are repeated as required
         to always return end-start frames.
     Outputs:
          m  - column vector for waveforms, one row per frame for other types
          sampPeriod - frame rate [seconds]
          parmKind
    """
    try:
        fh = open(file,'rb')
    except TypeError:
        fh = file
    try:
        fh.seek(0)
        nSamples, sampPeriod, sampSize, parmKind = struct.unpack(">IIHH", fh.read(12))
        pk = parmKind & 0x3f
        if parmKind & _C:
            scale, bias = np.fromfile(fh, '>f', sampSize).reshape(2,sampSize/2)
            nSamples -= 4
        s, e = max(0, start), min(nSamples, end)
        fh.seek(s*sampSize, 1)
        dtype, bytes = ('>h', 2) if parmKind & _C or pk in parms16bit else ('>f', 4)
        m = np.fromfile(fh, dtype, (e-s)*sampSize/bytes).reshape(e-s,sampSize/bytes)
        if parmKind & _C:
            m = (m + bias) / scale
        if pk == IREFC:
            m = m / 32767.0
        if pk == WAVEFORM:
            m = m.ravel()
    finally:
        if fh is not file: fh.close()
    if start != s or end != e: # repeat first or/and last frame as required
      m = np.r_[np.repeat(m[[0]], s-start, axis=0), m, np.repeat(m[[-1]], end-e, axis=0)]
    return m if not return_parmKind_and_sampPeriod else (m, parmKind, sampPeriod/1e7)


def readhtk_header(file):
    with  open(file,'rb') as fh:
        nSamples, sampPeriod, sampSize, parmKind = struct.unpack(">IIHH", fh.read(12))
        if parmKind & _C: nSamples -= 4
    return nSamples, sampPeriod/1e7, sampSize, parmKind


def writehtk(file_name, m, parmKind=USER, sampPeriod=0.01):
    """ Write htk feature file
     Input:
         file_name
          m  - data: vector for waveforms, one row per frame for other types
          sampPeriod - frame rate [seconds]
          parmKind
    """
    pk = parmKind & 0x3f
    parmKind &= ~_K # clear unsupported CRC bit
    m = np.atleast_2d(m)
    if pk == WAVEFORM:
        m = m.reshape(-1,1)
    with open(file_name,'wb') as fh:
        fh.write(struct.pack(">IIHH", len(m)+(4 if parmKind & _C else 0), int(sampPeriod*1e7),
            m.shape[1] * (2 if (pk in parms16bit or  parmKind & _C) else 4), parmKind))
        if pk == IREFC:
            m = m * 32767.0
        if pk in parms16bit:
            m = m.astype('>h')
        elif parmKind & _C:
            mmax, mmin = m.max(axis=0), m.min(axis=0)
            mmax[mmax==mmin] += 32767
            mmin[mmax==mmin] -= 32767 # to avoid division by zero for constant coefficients
            scale= 2*32767./(mmax-mmin)
            bias = 0.5*scale*(mmax+mmin)
            m = m * scale - bias
            scale.astype('>f').tofile(fh)
            bias.astype('>f').tofile(fh)
            m = m.astype('>h')
        else:
            m = m.astype('>f')

        m.tofile(fh)
