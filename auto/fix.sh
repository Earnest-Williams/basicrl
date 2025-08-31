#!/usr/bin/env bash

set -e  # Exit on first error

echo "🔍 Ruff (initial check):"
ruff check . || true

echo "🎨 Black pass 1:"
black .

echo "🔍 Ruff (after black):"
ruff check . || true

echo "🛠️ Autopep8 pass:"
find . -type f -name "*.py" -exec autopep8 --in-place --aggressive --aggressive {} +

echo "🎨 Black pass 2:"
black .

echo "📦 isort pass:"
isort . --recursive

echo "🎯 Final Ruff fix:"
ruff check . --fix

echo "✅ Done. Final Ruff report:"
ruff check .
