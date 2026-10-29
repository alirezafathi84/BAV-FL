"""Pairwise-mask secure aggregation (Bonawitz et al., 2017 -- semi-honest, no-dropout core).

Real cryptography (not a PRNG stand-in):
  * X25519 Diffie-Hellman key agreement between every client pair,
  * HKDF-SHA256 derives a fresh per-round, per-pair seed,
  * AES-256-CTR expands the seed into a uniform mask in Z_{2^32},
  * updates are fixed-point encoded in Z_{2^32}; masks cancel exactly in the sum.

Clients whose update fails verification are excluded *before* masking (the
accepted set is re-keyed for the round), mirroring the two-phase design of
verifiable SecAgg systems such as RoFL / EIFFeL.
"""
import time
import numpy as np
from cryptography.hazmat.primitives.asymmetric.x25519 import X25519PrivateKey
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes

MOD_BITS = 32
MOD = 1 << MOD_BITS
FRAC_BITS = 20                      # fixed-point precision 2^-20


def encode(x):
    v = np.round(np.asarray(x, float) * (1 << FRAC_BITS)).astype(np.int64)
    return np.mod(v, MOD).astype(np.uint64)


def decode(v):
    v = np.asarray(v, np.int64)
    v = np.where(v >= MOD // 2, v - MOD, v)
    return v.astype(float) / (1 << FRAC_BITS)


def _prg(seed, n):
    enc = Cipher(algorithms.AES(seed), modes.CTR(b"\x00" * 16)).encryptor()
    raw = enc.update(b"\x00" * (4 * n)) + enc.finalize()
    return np.frombuffer(raw, dtype="<u4").astype(np.uint64)


class SecureAggregator:
    def __init__(self, client_ids):
        self.ids = list(client_ids)
        self.sk = {c: X25519PrivateKey.generate() for c in self.ids}
        self.pk = {c: self.sk[c].public_key() for c in self.ids}
        self.timing = []

    def _pair_seed(self, i, j, rnd):
        shared = self.sk[i].exchange(self.pk[j])
        a, b = sorted([str(i), str(j)])
        info = f"bavfl-secagg|{a}|{b}|round{rnd}".encode()
        return HKDF(algorithm=hashes.SHA256(), length=32, salt=None, info=info).derive(shared)

    def mask(self, cid, vec, accepted, rnd):
        """Client-side: returns the masked encoding y_i = enc(x_i) + sum_j ±PRG(s_ij)."""
        y = encode(vec)
        for j in accepted:
            if j == cid:
                continue
            m = _prg(self._pair_seed(cid, j, rnd), len(y))
            y = (y + m) % MOD if str(cid) < str(j) else (y + MOD - m) % MOD
        return y

    def aggregate(self, contributions, rnd):
        """contributions: {cid: weighted update vector}. Returns (sum, masked views)."""
        t0 = time.perf_counter()
        accepted = list(contributions)
        masked = {c: self.mask(c, contributions[c], accepted, rnd) for c in accepted}
        tot = np.zeros_like(next(iter(masked.values())))
        for c in accepted:
            tot = (tot + masked[c]) % MOD
        self.timing.append(time.perf_counter() - t0)
        return decode(tot), masked
