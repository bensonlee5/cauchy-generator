"""RNG helpers for seeded, component-level reproducibility."""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field
from typing import Any

import torch

from dagzoo.runtime_profiling import record_runtime_profile_metric

SEED32_MIN = 0
SEED32_MAX = (2**32) - 1
_AMBIENT_NONCE_MARKER = "__ambient_nonce__"
_AMBIENT_NONCE_WORDS = 4


def validate_seed32(seed: int, *, field_name: str = "seed") -> int:
    """Validate an external seed against the supported unsigned 32-bit range."""

    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError(
            f"{field_name} must be an integer in [{SEED32_MIN}, {SEED32_MAX}], got {seed!r}."
        )
    if seed < SEED32_MIN or seed > SEED32_MAX:
        raise ValueError(
            f"{field_name} must be an integer in [{SEED32_MIN}, {SEED32_MAX}], got {seed!r}."
        )
    return int(seed)


def derive_seed(base_seed: int, *components: str | int) -> int:
    """Derive a deterministic 32-bit seed from a base seed and components."""

    h = _seed_hasher_with_components(base_seed, components)
    return int.from_bytes(h.digest(), "little") % SEED32_MAX


def _seed_hasher_with_components(
    base_seed: int,
    components: tuple[str | int, ...],
) -> Any:
    """Return one seeded blake2s hasher initialized for `components`."""

    hasher = hashlib.blake2s(digest_size=8)
    hasher.update(str(base_seed).encode("utf-8"))
    for component in components:
        hasher.update(b"|")
        hasher.update(str(component).encode("utf-8"))
    return hasher


@dataclass(slots=True, frozen=True)
class KeyedRng:
    """A keyed RNG namespace rooted at one deterministic base seed."""

    seed: int
    path: tuple[str | int, ...] = ()
    _ambient_nonce: tuple[int, ...] = field(default=(), repr=False)
    _child_cache: dict[tuple[str | int, ...], "KeyedRng"] = field(
        default_factory=dict,
        repr=False,
        compare=False,
    )
    _child_seed_cache: dict[tuple[str | int, ...], int] = field(
        default_factory=dict,
        repr=False,
        compare=False,
    )
    _generator_template_cache: dict[tuple[str, tuple[str | int, ...]], torch.Generator] = field(
        default_factory=dict,
        repr=False,
        compare=False,
    )
    _seed_hasher_template: Any | None = field(
        default=None,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        """Normalize public path input to an immutable tuple."""

        path = self.path
        normalized = (path,) if isinstance(path, str | int) else tuple(path)
        object.__setattr__(self, "path", normalized)
        object.__setattr__(
            self,
            "_ambient_nonce",
            tuple(int(component) for component in self._ambient_nonce),
        )
        if self._seed_hasher_template is None:
            ambient_components: tuple[str | int, ...] = ()
            if self._ambient_nonce:
                ambient_components = (_AMBIENT_NONCE_MARKER, *self._ambient_nonce)
            object.__setattr__(
                self,
                "_seed_hasher_template",
                _seed_hasher_with_components(
                    self.seed,
                    ambient_components + self.path,
                ),
            )

    def keyed(self, *components: str | int) -> "KeyedRng":
        """Return a child namespace with the provided semantic path appended."""

        normalized = tuple(components)
        if not normalized:
            return self
        cached = self._child_cache.get(normalized)
        if cached is not None:
            return cached
        template = self._seed_hasher_template
        if template is None:
            raise RuntimeError("KeyedRng seed hasher template was not initialized.")
        child_template = template.copy()
        for component in normalized:
            child_template.update(b"|")
            child_template.update(str(component).encode("utf-8"))
        child = KeyedRng(
            seed=self.seed,
            path=self.path + normalized,
            _ambient_nonce=self._ambient_nonce,
            _seed_hasher_template=child_template,
        )
        self._child_cache[normalized] = child
        return child

    def child_seed(self, *components: str | int) -> int:
        """Return a deterministic seed for this namespace and child components."""

        normalized = tuple(components)
        cached = self._child_seed_cache.get(normalized)
        if cached is not None:
            return cached
        template = self._seed_hasher_template
        if template is None:
            raise RuntimeError("KeyedRng seed hasher template was not initialized.")
        hasher = template.copy()
        for component in normalized:
            hasher.update(b"|")
            hasher.update(str(component).encode("utf-8"))
        derived = int.from_bytes(hasher.digest(), "little") % SEED32_MAX
        self._child_seed_cache[normalized] = derived
        return derived

    def torch_rng(self, *components: str | int, device: str = "cpu") -> torch.Generator:
        """Return a torch Generator for this namespace and child components."""

        normalized = tuple(components)
        cache_key = (str(device), normalized)
        start = time.perf_counter()
        template = self._generator_template_cache.get(cache_key)
        if template is None:
            template = torch.Generator(device=device)
            template.manual_seed(self.child_seed(*normalized))
            self._generator_template_cache[cache_key] = template
        generator = template.clone_state()
        record_runtime_profile_metric(
            "profile_rng_torch_generator_elapsed_seconds",
            time.perf_counter() - start,
        )
        return generator


def keyed_rng_from_generator(generator: torch.Generator, *components: str | int) -> KeyedRng:
    """Consume ambient generator state and convert it into a keyed RNG root."""

    words = tuple(
        int(value)
        for value in torch.randint(
            0,
            SEED32_MAX + 1,
            (_AMBIENT_NONCE_WORDS,),
            generator=generator,
            device=str(generator.device),
        ).tolist()
    )
    return KeyedRng(
        validate_seed32(words[0]),
        _ambient_nonce=words[1:],
    ).keyed(*components)
