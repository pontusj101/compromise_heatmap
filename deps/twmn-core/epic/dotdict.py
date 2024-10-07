#!/usr/bin/env python

"""Module with dictionary wrapper classes."""


from __future__ import annotations

from copy import deepcopy
from typing import Any, Callable


class DotDict(dict):
    """A dict whose keys can be accessed as object properties.

    Example:
    m = DotDict({'first_name': 'Eduardo'}, last_name='Pool', age=24, sports=['Soccer'])
    assert m.first_name == 'Eduardo'
    """

    __slots__ = ()

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initialize a dot-traversable dict.

        :param args: dictionaries to convert to DotDict
        :param kwargs: key-value pairs to add to the DotDict
        """
        super().__init__(*args, **kwargs)

        (args := list(args or [])).append(kwargs)

        for arg in args:
            if isinstance(arg, dict):
                for k, v in arg.items():
                    if isinstance(v, dict):
                        v = DotDict(v)
                    if isinstance(v, list):
                        v = DotList(*v)

                    self[k] = v

    def __getattr__(self, attr):
        """Return dict values using the supplied attribute as key.

        :param attr: the key to return from the dict
        """
        if attr in self:
            return self[attr]

        raise AttributeError(f"'{self.__class__.__name__}' object has no property '{attr}'")

    def __setattr__(self, attr: str, value: Any) -> None:
        """Set a dict entry for the used attribute.

        :param attr: the dict key to set
        :param value: the value to set the key to
        """
        self.__setitem__(attr, value)

    def __delattr__(self, item: Any) -> None:
        """Allow deleting dict keys using `delattr`.

        :param item: the dict key to remove
        """
        self.__delitem__(item)

    def __deepcopy__(self, memo: dict = None) -> DotDict:
        """Return a copy of the DotDict and of any of its values that are objects.

        :param memo: internally required arg
        """
        # https://stackoverflow.com/questions/49901590
        return DotDict(deepcopy(dict(self), memo=memo))

    def __str__(self) -> str:
        """Return a string representation of the DotDict."""
        return dict.__str__(self)

    def __repr__(self) -> str:
        """Return a developer-friendly representation of the DotDict."""
        return dict.__repr__(self)

    def __dir__(self) -> list[str]:
        return dict.__dir__(self) + list(self.keys())


class DotList(list):
    """A list whose items can be traversed like object attributes."""

    def __init__(self, *args: Any) -> None:
        """Initialize a dot-traversable list.

        :param args: items to put into the list
        """
        items: Any = []

        for item in args:
            if isinstance(item, dict):
                items.append(DotDict(item))
            elif isinstance(item, list):
                items.append(DotList(*item))
            else:
                items.append(item)

        super().__init__(items)

    # TODO implement dot support for indices!


class DefaultDotDict(DotDict):
    """A DefaultDotDict is to a DotDict what a defaultdict is to a dict."""

    __slots__ = ('__default_factory__', )

    def __init__(self, default_factory: Callable, *args: Any, **kwargs: Any) -> None:
        """Initialize a DefaultDotDict.

        :param default_factory: the Callable to be called when a key is missing
        """
        dict.__init__(*args, **kwargs)

        (args := list(args or [])).append(kwargs)

        for arg in args:
            if isinstance(arg, dict):
                for k, v in arg.items():
                    if isinstance(v, dict):
                        v = DefaultDotDict(default_factory, v)
                    if isinstance(v, list):
                        v = DefaultDotList(default_factory, *v)

                    self[k] = v

        self.__default_factory__ = default_factory

    def __setattr__(self, attr: str, value: Any) -> None:
        """Set the attribute as if it was a key in the dictionary."""
        if attr == '__default_factory__':
            object.__setattr__(self, '__default_factory__', value)
            return

        super().__setitem__(attr, value)

    def __setitem__(self, key: str, value: Any) -> None:
        """Set the item of key in the dictionary.

        If the key is default_factory this throws an error since the Callable should
        only be set at object creation.
        """
        if key == "__default_factory__":
            raise KeyError(
                "__default_factory__ is a reserved key"
            )

        super().__setitem__(key, value)

    def __getattr__(self, attr: str) -> Any:
        """Return the attribute as if it was a key in the dictionary."""
        if attr in ('__dict__', '__weakref__'):  # use of __slots__ disables these
            super().__getattr__(attr)  # raise AttributeError

        return self[attr]

    def __missing__(self, key: Any) -> Any:
        """Return the factory default value when the key not in the dict."""
        if key in ('__dict__', '__weakref__', '__default_factory__'):
            raise KeyError(key)

        self.__setitem__(key, self.__default_factory__())
        return self[key]


class DefaultDotList(list):
    """A list that recursively turns dict objects into DefaultDotDict objects."""

    def __init__(self, default_factory: Callable, *args: Any) -> None:
        """Recursively turn dict objects into DefaultDotDict objects."""
        items: Any = []

        for item in args:
            if isinstance(item, dict):
                items.append(DefaultDotDict(default_factory, item))
            elif isinstance(item, list):
                items.append(DefaultDotList(default_factory, *item))
            else:
                items.append(item)

        super().__init__(items)

        # TODO implement dot notation for indices
