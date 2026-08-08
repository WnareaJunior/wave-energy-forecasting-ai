#!/usr/bin/env bash
# Set up and run the NDBC buoy pilot on a fresh Linux machine.
#
#   bash scripts/run_pilot.sh                       # default: station 46041, 2015-2023
#   bash scripts/run_pilot.sh 46087 2016-2023       # another station / range
#
# Creates a virtualenv in .venv, installs the minimal dependency set, downloads
# the buoy record (a few hundred KB per station-year, cached in data/raw/ndbc),
# and runs the full model ladder. Safe to re-run: cached downloads are reused.

set -euo pipefail

STATION="${1:-46041}"
YEARS="${2:-2015-2023}"

cd "$(dirname "$0")/.."
echo "Repository: $(pwd)"

# --- Python version check -------------------------------------------------
PYTHON="${PYTHON:-python3}"
if ! command -v "$PYTHON" >/dev/null 2>&1; then
    echo "ERROR: $PYTHON not found. Install it with:" >&2
    echo "  sudo apt install -y python3 python3-venv python3-pip" >&2
    exit 1
fi

PY_VERSION="$($PYTHON -c 'import sys; print("%d.%d" % sys.version_info[:2])')"
echo "Python: $PY_VERSION ($($PYTHON -c 'import sys; print(sys.executable)'))"

if ! "$PYTHON" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 9) else 1)'; then
    echo "ERROR: Python 3.9 or newer required, found $PY_VERSION." >&2
    echo "On Debian/Ubuntu: sudo apt install -y python3.11 python3.11-venv" >&2
    echo "then re-run with: PYTHON=python3.11 bash scripts/run_pilot.sh" >&2
    exit 1
fi

# --- Virtualenv -----------------------------------------------------------
if [ ! -d .venv ]; then
    echo "Creating virtualenv in .venv"
    "$PYTHON" -m venv .venv || {
        echo "ERROR: venv creation failed. Install the venv module with:" >&2
        echo "  sudo apt install -y python3-venv" >&2
        exit 1
    }
fi
# shellcheck disable=SC1091
source .venv/bin/activate

echo "Installing dependencies (a few minutes on a slow machine)"
pip install --quiet --upgrade pip
pip install --quiet -r requirements-pilot.txt

# --- Connectivity check ---------------------------------------------------
# Fail here with a clear message rather than deep inside the experiment.
echo "Checking access to ndbc.noaa.gov"
python - <<'PY'
import sys
import requests

url = "https://www.ndbc.noaa.gov/data/historical/stdmet/46041h2020.txt.gz"
try:
    r = requests.get(url, timeout=30, stream=True)
    r.close()
except Exception as e:
    sys.exit(f"ERROR: cannot reach ndbc.noaa.gov: {e}")

if not r.ok:
    sys.exit(f"ERROR: ndbc.noaa.gov returned HTTP {r.status_code}")
print("  reachable")
PY

# --- Run ------------------------------------------------------------------
echo
echo "Running pilot: station $STATION, years $YEARS"
echo
python experiments/pilot_ndbc.py --station "$STATION" --years "$YEARS"

echo
echo "Results CSV: results/pilot/pilot_${STATION}_results.csv"
echo "Downloaded data cached in: data/raw/ndbc/"
