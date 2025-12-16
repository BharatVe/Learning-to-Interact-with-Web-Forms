#!/usr/bin/env bash
set -e
PACKAGES=(python3 python3-venv python3-pip curl)
sudo apt-get update
sudo apt-get install -y "${PACKAGES[@]}"
VENV_DIR="$HOME/gf_form_agent_env"
if [ ! -d "$VENV_DIR" ]; then
  python3 -m venv "$VENV_DIR"
fi
source "$VENV_DIR/bin/activate"
pip install --upgrade pip
pip install --upgrade playwright
python -m playwright install chromium
deactivate
printf 'Environment ready. Activate with: source ~/gf_form_agent_env/bin/activate\n'
