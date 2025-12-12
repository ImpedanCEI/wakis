#!/usr/bin/env bash
set -euo pipefail
IFS=$'\n\t'

# Extract project metadata from pyproject.toml (PEP 621)
NAME="wakis"
VER=$(python -c "from wakis._version import __version__; print(__version__)")


echo "========================================================================"
echo "Tagging $NAME v$VER"
echo "========================================================================"

git tag -a "v$VER" -m "Release $VER"
git push origin "v$VER"

echo "========================================================================"
echo "Building $NAME v$VER"
echo "========================================================================"

# Clean old artifacts
rm -rf dist build *.egg-info

# Build wheel + sdist (PEP 517)
python -m build

echo "========================================================================"
echo "Uploading $NAME v$VER to PyPI"
echo "========================================================================"

twine upload dist/*


rm -r dist/ *.egg-info/ build/