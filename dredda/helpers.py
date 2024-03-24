import contextlib
import numpy as np
import torch


@contextlib.contextmanager
def temp_seed(seed):
    """Temporarily set the random seed of Numpy."""
    state = np.random.get_state()
    np.random.seed(seed)
    try:
        yield
    finally:
        np.random.set_state(state)


def clone_state(net):
    """Clone the state of a PyTorch model."""
    params_dict = dict()
    for name, param in net.state_dict().items():
        params_dict[name] = param.detach().clone()
    return params_dict


def check_state_equivalence(param_dict1, param_dict2):
    """Check if the states of two PyTorch models are equivalent."""
    for name, param in param_dict1.items():
        if torch.allclose(param, param_dict2[name]):
            print("%s: same" % (name))
        else:
            max_diff = torch.max(torch.abs(param - param_dict2[name]))
            print("%s: different, max difference: %.5f" % (name, max_diff))


def iprint(s="", *args, ident=0, **kwargs):
    if not isinstance(s, str):
        s = str(s)
    print("\t" * ident + s, *args, **kwargs)


def iprint_dict(d, keys=None, ident=0):
    keys = keys if keys is not None else d.keys()
    for k in keys:
        v = d[k]
        if isinstance(v, dict):
            iprint(k + ":", ident=ident)
            iprint_dict(v, ident=ident + 1)
        else:
            iprint(f"{k}: {v}", ident=ident)
