"""Backend-independent coupled-response entry points."""


def _provider(backend, device):
    if backend != "numpy" or device != "cpu":
        raise NotImplementedError(
            "mobile response qualification currently numpy/cpu only; no implicit fallback"
        )
    from .backends.numpy import mobile_response
    return mobile_response


def mobile_features(trace):
    return _provider("numpy", "cpu").mobile_features(trace)


def fit_mobile_context(config, records, family, *, backend="numpy", device="cpu"):
    return _provider(backend, device).fit_mobile_context(config, records, family)


def validate_mobile_response(config, mechanism, records, *, backend="numpy", device="cpu"):
    return _provider(backend, device).validate_mobile_response(config, mechanism, records)
