#!/bin/bash
# Quick verification script for release automation setup

set -e

echo "🧪 Testing Release Automation Setup"
echo "===================================="
echo ""

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m'

# Check files exist
echo "1. Checking required files..."
files_to_check=(
    "NEXT_VERSION"
    ".github/workflows/ci-dev-testpypi.yml"
    ".github/workflows/release-tags.yml"
)

for file in "${files_to_check[@]}"; do
    if [ -f "$file" ]; then
        echo -e "  ${GREEN}✓${NC} $file"
    else
        echo -e "  ${RED}✗${NC} $file MISSING"
        exit 1
    fi
done

# Check pyproject.toml configuration
echo ""
echo "2. Validating pyproject.toml..."

PACKAGE_NAME=$(python3 - <<'PY'
from __future__ import annotations

import re
from pathlib import Path

text = Path("pyproject.toml").read_text(encoding="utf-8")
m = re.search(r'^name\s*=\s*"([^"]+)"\s*$', text, flags=re.MULTILINE)
if not m:
    raise SystemExit("ERROR: Could not detect project.name from pyproject.toml")
print(m.group(1))
PY
)

IMPORT_NAME=$(python3 - <<'PY'
from __future__ import annotations

from pathlib import Path

src = Path("src")
for child in src.iterdir():
    if child.is_dir() and (child / "__init__.py").is_file():
        print(child.name)
        raise SystemExit(0)
raise SystemExit("ERROR: Could not detect an importable package under src/")
PY
)

# Check dynamic version
if grep -q 'dynamic.*=.*\[.*"version".*\]' pyproject.toml; then
    echo -e "  ${GREEN}✓${NC} dynamic version configured"
else
    echo -e "  ${RED}✗${NC} dynamic version NOT configured"
    exit 1
fi

# Check setuptools-scm
if grep -q 'setuptools-scm' pyproject.toml; then
    echo -e "  ${GREEN}✓${NC} setuptools-scm in build-system"
else
    echo -e "  ${RED}✗${NC} setuptools-scm NOT in build-system"
    exit 1
fi

# Check local_scheme
if grep -q 'local_scheme.*=.*"no-local-version"' pyproject.toml; then
    echo -e "  ${GREEN}✓${NC} local_scheme = no-local-version"
else
    echo -e "  ${RED}✗${NC} local_scheme NOT configured"
    exit 1
fi

# Check src layout
if grep -q 'where.*=.*\[.*"src".*\]' pyproject.toml; then
    echo -e "  ${GREEN}✓${NC} src layout configured"
else
    echo -e "  ${RED}✗${NC} src layout NOT configured"
    exit 1
fi

# Check package structure
echo ""
echo "3. Validating package structure..."

if [ -d "src/${IMPORT_NAME}" ]; then
    echo -e "  ${GREEN}✓${NC} Package directory exists"
else
    echo -e "  ${RED}✗${NC} Package directory NOT found"
    exit 1
fi

if [ -f "src/${IMPORT_NAME}/__init__.py" ]; then
    echo -e "  ${GREEN}✓${NC} Package __init__.py exists"
else
    echo -e "  ${RED}✗${NC} Package __init__.py NOT found"
    exit 1
fi

# Check version imports in __init__.py
if grep -q "__version__" "src/${IMPORT_NAME}/__init__.py"; then
    echo -e "  ${GREEN}✓${NC} Version exports in __init__.py"
else
    echo -e "  ${RED}✗${NC} Version exports NOT in __init__.py"
    exit 1
fi

# Test setuptools-scm
echo ""
echo "4. Testing setuptools-scm..."

if command -v python3 &> /dev/null; then
    VERSION=$(python3 -m setuptools_scm 2>&1 || echo "ERROR")
    if [ "$VERSION" != "ERROR" ]; then
        echo -e "  ${GREEN}✓${NC} setuptools-scm works: $VERSION"
    else
        echo -e "  ${RED}✗${NC} setuptools-scm failed"
        exit 1
    fi
else
    echo "  ⚠️  Python not found, skipping"
fi

# Test build
echo ""
echo "5. Testing package build..."

if [ -d "dist" ]; then
    rm -rf dist/
fi

NORMALIZED_NAME=$(echo "$PACKAGE_NAME" | tr '[:lower:]' '[:upper:]' | tr '.-' '_')
ENV_VAR_NAME="SETUPTOOLS_SCM_PRETEND_VERSION_FOR_${NORMALIZED_NAME}"
export "$ENV_VAR_NAME"=0.1.0.dev999
python3 -m build > /dev/null 2>&1 || {
    echo -e "  ${RED}✗${NC} Build failed"
    exit 1
}

if [ -f "dist/"*.whl ] && [ -f "dist/"*.tar.gz ]; then
    echo -e "  ${GREEN}✓${NC} Build successful"
    ls -lh dist/
else
    echo -e "  ${RED}✗${NC} Build artifacts missing"
    exit 1
fi

# Check wheel contents
echo ""
echo "6. Verifying wheel contents..."

WHEEL=$(ls dist/*.whl)
CONTENTS=$(python3 -m zipfile -l "$WHEEL" | grep "${IMPORT_NAME}/")

if echo "$CONTENTS" | grep -q "__init__.py"; then
    echo -e "  ${GREEN}✓${NC} __init__.py in wheel"
else
    echo -e "  ${RED}✗${NC} __init__.py NOT in wheel"
    exit 1
fi

if echo "$CONTENTS" | grep -q "_version.py"; then
    echo -e "  ${GREEN}✓${NC} _version.py in wheel"
else
    echo -e "  ${RED}✗${NC} _version.py NOT in wheel"
    exit 1
fi


# Summary
echo ""
echo "===================================="
echo -e "${GREEN}✅ All checks passed!${NC}"
echo ""
echo "Next steps:"
echo "  1. Configure GitHub secrets (TESTPYPI_API_TOKEN, PYPI_API_TOKEN)"
echo "  2. Push to dev branch to test TestPyPI workflow"
echo "  3. Create tag (e.g., v0.1.0rc1) to test PyPI workflow"
echo ""
echo "See RELEASE_GUIDE.md for detailed instructions."

