#!/bin/bash
set -euo pipefail; IFS=$'\n\t'

NAME="wakis"
VER=$(grep __version__ wakis/_version.py| cut -d '=' -f2 | xargs)

echo "========================================================================"
echo "Tagging $NAME v$VER"
echo "========================================================================"

git tag -a v$VER
git push origin v$VER

echo "========================================================================"
echo "Releasing $NAME v$VER on PyPI"
echo "========================================================================"

python setup.py sdist
twine upload dist/*
rm -r dist/ *.egg-info