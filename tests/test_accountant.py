import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from bavfl.privacy import _rdp_poisson_gaussian, ORDERS, eps_from_rdp
from dp_accounting.rdp import rdp_privacy_accountant as r
for q, s in [(0.08, 1.2), (0.18, 0.7), (0.05, 3.0), (0.5, 1.0), (1.0, 2.0)]:
    mine = _rdp_poisson_gaussian(q, s)
    ref = np.asarray(r._compute_rdp_poisson_subsampled_gaussian(q, s, ORDERS))
    assert np.allclose(mine, ref, rtol=1e-9), (q, s)
    assert abs(eps_from_rdp(mine * 300, 1e-4) - r.compute_epsilon(ORDERS, ref * 300, 1e-4)[0]) < 1e-9
print("accountant matches dp_accounting: OK")
