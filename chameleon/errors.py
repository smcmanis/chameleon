import logging
import sys
from typing import Any, Callable, List, Tuple

import numpy as np

log = logging.getLogger(__name__)


class Error(Exception):
    """Base class for exceptions in Chameleon."""

    message = "Chameleon error."


def catch_and_exit(f: Callable) -> Callable:
    """Decorate function to exit program if it throws an Error."""
    def wrapped(*args: Any, **kwargs: Any) -> None:
        try:
            f(*args, **kwargs)
        except Error as e:
            log.error(str(e))
            sys.exit()

    return wrapped

class NonBinaryTargets(Error):
    "More than two distinct values detected in target file"
    message = "The target has more than 2 different values (i.e. not a binary class)"