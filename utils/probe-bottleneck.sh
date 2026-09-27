#!/usr/bin/env bash
# Report which file the name `bottleneck` resolves to, and whether it carries a
# __version__. pandas raises ImportError("Can't determine version for
# bottleneck") when it does not, which takes smolvla's CPU reference down.
#
# Run inside the activated venv. Takes a label for the log.

LABEL="${1:-probe}"

echo "::group::bottleneck -- $LABEL"

python - <<'PY'
import importlib.util
import sys

spec = importlib.util.find_spec("bottleneck")
if spec is None:
    print("find_spec: not found")
else:
    print("find_spec origin :", spec.origin)
    print("find_spec loader :", type(spec.loader).__name__)
    print("submodule paths  :", list(spec.submodule_search_locations or []))

try:
    import bottleneck as b
except Exception as e:
    print("import failed    :", type(e).__name__, e)
else:
    print("__file__         :", getattr(b, "__file__", None))
    print("__version__      :", getattr(b, "__version__", "NO __version__"))
    print("__path__         :", list(getattr(b, "__path__", []) or []))

print("sys.path:")
for p in sys.path:
    print("   ", p or "<cwd>")
PY

pip show bottleneck 2>/dev/null || echo "pip show: not installed by pip"

# A namespace package is the usual way a name imports with no __version__: any
# directory on sys.path called bottleneck will do it.
python -c "import sys; print('\n'.join(sys.path))" | while read -r d; do
  [ -n "$d" ] && [ -e "$d/bottleneck" ] && ls -la "$d/bottleneck" | head -20
done

echo "::endgroup::"
