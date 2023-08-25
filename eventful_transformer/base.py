from collections import defaultdict
from sys import stdout

from torch import nn as nn


class Counts(defaultdict):
    """
    A utility class for counting operations. Essentially, a dict with
    arithmetic operations on values.
    """

    def __init__(self, *args, **kwargs):
        if len(args) > 0 or len(kwargs) > 0:
            super().__init__(*args, **kwargs)
        else:
            super().__init__(int)

    def __add__(self, other):
        result = self.copy()
        if isinstance(other, Counts):
            for key, value in other.items():
                result[key] += value
        else:
            for key, value in result.items():
                result[key] += other
        return result

    def __mul__(self, other):
        result = self.copy()
        for key in result:
            result[key] *= other
        return result

    def __neg__(self):
        result = self.copy()
        for key, value in result.items():
            result[key] = -value
        return result

    def __radd__(self, other):
        return self.__add__(other)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __rsub__(self, other):
        return self.__neg__().__add__(other)

    def __sub__(self, other):
        return self.__add__(-other)

    def __truediv__(self, other):
        return self.__mul__(1.0 / other)

    def csv_header(self):
        """
        Generates a CSV header line from the keys.
        """
        return dict_csv_header(self)

    def csv_line(self):
        """
        Generates a CSV data line from the values.
        """
        return dict_csv_line(self)

    def pretty_print(self, indent=4, value_format=".3e", file=stdout, flush=False):
        """
        Prints the count data in a human-readable format.

        :param indent: Number of spaces to use for indents
        :param value_format: Number format for count values
        :param file: File where output should be printed (default is
        stdout)
        :param flush: Whether to flush the output buffer
        """
        print(dict_string(self, indent, value_format), file=file, flush=flush)


class ExtendedModule(nn.Module):
    """
    An extended nn.Module that adds tooling for useful features:
    operation counting, state resets, and sub-module filtering.
    """

    def __init__(self):
        super().__init__()
        self.count_mode = False
        self.counts = Counts()

    def clear_counts(self):
        """
        Resets the operation counts for this module and all
        ExtendedModule submodules.
        """
        for module in self.extended_modules():
            module.counts.clear()

    def counting(self, mode=True):
        """
        Sets the operation counting mode (enables counting by default).

        :param mode: A True/False counting mode
        """
        for module in self.extended_modules():
            module.count_mode = mode

    def extended_modules(self):
        """
        Enumerates all ExtendedModule submodules.
        """
        return self.modules_of_type(ExtendedModule)

    def modules_of_type(self, module_type):
        """
        Enumerates all submodules of the specified type.

        :param module_type: Enumerate children of this class
        :return: An iterator of children of the specified class
        """
        return filter(lambda x: isinstance(x, module_type), self.modules())

    def no_counting(self):
        """
        Disables operation counting.
        """
        self.counting(mode=False)

    def reset(self):
        """
        Resets extra state for this module and all submodules.
        """
        for module in self.extended_modules():
            module.reset_self()

    def reset_self(self):
        """
        Resets extra state in this module (but not in submodules). Child
        classes can define reset logic by overriding this method.
        """
        pass

    def total_counts(self):
        """
        Returns a sum of operation counts over this module and all
        ExtendedModule submodules.
        """
        return sum(x.counts for x in self.extended_modules())


def numeric_tuple(x, length):
    """
    Expands a single numeric value (int, float, complex, or bool) into a
    tuple of a specified length. If the value is not of the specified
    types, does nothing.

    :param x: The input value
    :param length: The length of tuple to return if x is of the
    specified types
    """
    return (x,) * length if isinstance(x, (int, float, complex, bool)) else tuple(x)


def dict_csv_header(x):
    """
    Returns a CSV-header string containing the keys of a dict.

    :param x: A dict
    """
    return ",".join(k for k in sorted(x.keys()))


def dict_csv_line(x):
    """
    Returns a CSV-content string containing the values of a dict.

    :param x: A dict
    """
    return ",".join(f"{x[k]:g}" for k in sorted(x.keys()))


def dict_string(x, indent=4, value_format=".4g"):
    """
    Returns a human-readable string for a dict.

    :param indent: Number of spaces to use for indents
    :param value_format: Number format for count values
    """
    lines = []
    key_length = max(len(str(key)) for key in x.keys())
    format_str = " " * indent + f"{{:<{key_length + 1}}} {{:{value_format}}}"
    for key in sorted(x.keys()):
        lines.append(format_str.format(f"{key}:", x[key]))
    return "\n".join(lines)
