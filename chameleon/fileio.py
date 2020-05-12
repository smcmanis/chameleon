
import os.path
from glob import glob
from typing import List


def datafiles(directory: str) -> List[str]:
    """ all pkl files within a directory."""
    glob_pattern = os.path.join(directory, "*.pkl")
    return [os.path.basename(x) for x in glob(glob_pattern, recursive=False)]