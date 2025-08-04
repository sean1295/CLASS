import os
import sys
import contextlib


class suppress_output(contextlib.ContextDecorator):
    def __init__(self, stdout=True, stderr=True):
        self.stdout = stdout
        self.stderr = stderr
        self._original_stdout = None
        self._original_stderr = None
        self._devnull = None

    def __enter__(self):
        # Open devnull
        self._devnull = open(os.devnull, "w")

        # Suppress stdout if specified
        if self.stdout:
            self._original_stdout = sys.stdout
            sys.stdout = self._devnull

        # Suppress stderr if specified
        if self.stderr:
            self._original_stderr = sys.stderr
            sys.stderr = self._devnull

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Restore stdout if it was suppressed
        if self.stdout and self._original_stdout:
            sys.stdout = self._original_stdout

        # Restore stderr if it was suppressed
        if self.stderr and self._original_stderr:
            sys.stderr = self._original_stderr

        # Close devnull
        if self._devnull:
            self._devnull.close()

        # Propagate exceptions
        return False


# Define a resolver with an `if` condition
def conditional_resolver(condition, true_value, false_value):
    return true_value if eval(condition) else false_value
