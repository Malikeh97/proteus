"""One registry flavour, used by every plug-in point in the package.

Adding a component is a file plus a decorator; nothing else in the tree changes:

    from proteus.menu.wrappers import WRAPPERS
    from proteus.menu.wrappers.base import Wrapper

    @WRAPPERS.register("my_filter")
    class MyFilter(Wrapper):
        ...

The YAML `type:` field is the registry key, so the new component is reachable
from configs immediately.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Generic, TypeVar

T = TypeVar("T")


class Registry(Generic[T]):
    def __init__(self, kind: str) -> None:
        self._kind = kind
        self._entries: dict[str, type[T]] = {}

    def register(self, name: str) -> Callable[[type[T]], type[T]]:
        def decorator(cls: type[T]) -> type[T]:
            if name in self._entries:
                raise ValueError(f"{self._kind} '{name}' is already registered")
            self._entries[name] = cls
            return cls

        return decorator

    def get(self, name: str) -> type[T]:
        if name not in self._entries:
            raise ValueError(
                f"Unknown {self._kind} '{name}'. Registered: {', '.join(self.names()) or '(none)'}"
            )
        return self._entries[name]

    def create(self, name: str, *args, **kwargs) -> T:
        return self.get(name)(*args, **kwargs)

    def names(self) -> list[str]:
        return sorted(self._entries)

    def __contains__(self, name: str) -> bool:
        return name in self._entries
