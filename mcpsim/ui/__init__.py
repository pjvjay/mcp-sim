"""``mcpsim ui``: a local test runner over scenario files and run directories (contract D).

:func:`mcpsim.ui.app.create_app` builds the Starlette app and :func:`mcpsim.ui.app.serve` runs
it; the page itself is ``static/index.html`` with ``app.js`` and ``app.css`` (no build step).
Starlette and uvicorn are imported only here, so the rest of the CLI never needs them.
"""
