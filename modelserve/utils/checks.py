import re
import subprocess
from importlib import metadata
import torch

def parse_version(version='0.0.0') -> tuple:
    """
    Convert a version string to a tuple of integers, ignoring any extra non-numeric string attached to the version. This
    function replaces deprecated 'pkg_resources.parse_version(v)'.

    Args:
        version (str): Version string, i.e. '2.0.1+cpu'

    Returns:
        (tuple): Tuple of integers representing the numeric part of the version and the extra string, i.e. (2, 0, 1)
    """
    try:
        return tuple(map(int, re.findall(r'\d+', version)[:3]))  # '2.0.1+cpu' -> (2, 0, 1)
    except Exception as e:
        print(e)
        return 0, 0, 0


def is_ascii(s) -> bool:
    """
    Check if a string is composed of only ASCII characters.

    Args:
        s (str): String to be checked.

    Returns:
        bool: True if the string is composed only of ASCII characters, False otherwise.
    """
    # Convert list, tuple, None, etc. to string
    s = str(s)

    # Check if the string is composed of only ASCII characters
    return all(ord(c) < 128 for c in s)


def check_version(current: str = '0.0.0',
                  required: str = '0.0.0',
                  name: str = 'version',
                  hard: bool = False,
                  verbose: bool = False,
                  msg: str = '') -> bool:
    """
    Check current version against the required version or range.

    Args:
        current (str): Current version or package name to get version from.
        required (str): Required version or range (in pip-style format).
        name (str, optional): Name to be used in warning message.
        hard (bool, optional): If True, raise an AssertionError if the requirement is not met.
        verbose (bool, optional): If True, print warning message if requirement is not met.
        msg (str, optional): Extra message to display if verbose.

    Returns:
        (bool): True if requirement is met, False otherwise.

    Example:
        ```python
        # Check if current version is exactly 22.04
        check_version(current='22.04', required='==22.04')

        # Check if current version is greater than or equal to 22.04
        check_version(current='22.10', required='22.04')  # assumes '>=' inequality if none passed

        # Check if current version is less than or equal to 22.04
        check_version(current='22.04', required='<=22.04')

        # Check if current version is between 20.04 (inclusive) and 22.04 (exclusive)
        check_version(current='21.10', required='>20.04,<22.04')
        ```
    """
    if not current:  # if current is '' or None
        return True
    elif not current[0].isdigit():  # current is package name rather than version string, i.e. current='ultralytics'
        try:
            name = current  # assigned package name to 'name' arg
            current = metadata.version(current)  # get version string from package name
        except metadata.PackageNotFoundError:
            if hard:
                raise ModuleNotFoundError(f'WARNING ⚠️ {current} package is required but not installed')
            else:
                return False

    if not required:  # if required is '' or None
        return True

    op = ''
    version = ''
    result = True
    c = parse_version(current)  # '1.2.3' -> (1, 2, 3)
    for r in required.strip(',').split(','):
        op, version = re.match(r'([^0-9]*)([\d.]+)', r).groups()  # split '>=22.04' -> ('>=', '22.04')
        v = parse_version(version)  # '1.2.3' -> (1, 2, 3)
        if op == '==' and c != v:
            result = False
        elif op == '!=' and c == v:
            result = False
        elif op in ('>=', '') and not (c >= v):  # if no constraint passed assume '>=required'
            result = False
        elif op == '<=' and not (c <= v):
            result = False
        elif op == '>' and not (c > v):
            result = False
        elif op == '<' and not (c < v):
            result = False
    if not result:
        warning = f'WARNING ⚠️ {name}{op}{version} is required, but {name}=={current} is currently installed {msg}'
        if hard:
            raise ModuleNotFoundError(warning)  # assert version requirements met
    return result


def check_torchvision():
    """
    Checks the installed versions of PyTorch and Torchvision to ensure they're compatible.

    This function checks the installed versions of PyTorch and Torchvision, and warns if they're incompatible according
    to the provided compatibility table based on:
    https://github.com/pytorch/vision#installation.

    The compatibility table is a dictionary where the keys are PyTorch versions and the values are lists of compatible
    Torchvision versions.
    """

    import torchvision

    # Compatibility table
    compatibility_table = {'2.0': ['0.15'], '1.13': ['0.14'], '1.12': ['0.13']}

    # Extract only the major and minor versions
    v_torch = '.'.join(torch.__version__.split('+')[0].split('.')[:2])
    v_torchvision = '.'.join(torchvision.__version__.split('+')[0].split('.')[:2])

    if v_torch in compatibility_table:
        compatible_versions = compatibility_table[v_torch]
        if all(v_torchvision != v for v in compatible_versions):
            print(f'WARNING ⚠️ torchvision=={v_torchvision} is incompatible with torch=={v_torch}.\n'
                  f"Run 'pip install torchvision=={compatible_versions[0]}' to fix torchvision or "
                  "'pip install -U torch torchvision' to update both.\n"
                  'For a full compatibility table see https://github.com/pytorch/vision#installation')

def cuda_device_count() -> int:
    """
    Get the number of NVIDIA GPUs available in the environment.

    Returns:
        (int): The number of NVIDIA GPUs available.
    """
    try:
        # Run the nvidia-smi command and capture its output
        output = subprocess.check_output(['nvidia-smi', '--query-gpu=count', '--format=csv,noheader,nounits'],
                                         encoding='utf-8')

        # Take the first line and strip any leading/trailing white space
        first_line = output.strip().split('\n')[0]

        return int(first_line)
    except (subprocess.CalledProcessError, FileNotFoundError, ValueError):
        # If the command fails, nvidia-smi is not found, or output is not an integer, assume no GPUs are available
        return 0


def cuda_is_available() -> bool:
    """
    Check if CUDA is available in the environment.

    Returns:
        (bool): True if one or more NVIDIA GPUs are available, False otherwise.
    """
    return cuda_device_count() > 0
