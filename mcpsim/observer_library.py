"""Built-in observers, opt-in by name: ``observers: [{use: fabrication_auditor}, ...]``.

Plain declarations (the same mappings a scenario file would hold) so :mod:`mcpsim.scenario` can
resolve ``use:`` entries without importing the observer runner. The library is filled in by the
Informant-Report Method's fourth stage; see :mod:`mcpsim.observers` for the models and the DSL.
"""

from __future__ import annotations

from typing import Any

BUILTIN_OBSERVERS: dict[str, dict[str, Any]] = {}
