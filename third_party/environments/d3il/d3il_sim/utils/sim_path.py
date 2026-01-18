import os

D3IL_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir)
)


def d3il_path(*args) -> str:
    """
    Abstraction from os.path.join()
    Builds absolute paths from relative path strings with environments/d3il as root.
    If args already contains an absolute path, it is used as root for the subsequent joins
    Args:
        *args:

    Returns:
        absolute path

    """
    return os.path.abspath(os.path.join(D3IL_DIR, *args))
