#!/usr/bin/env bash
# Install Kimi Code CLI (terminal agent)
# Docs: https://www.kimi.com/code/docs/kimi-code-cli/getting-started.html
set -euo pipefail

if command -v kimi >/dev/null 2>&1; then
  echo "kimi already installed: $(kimi --version 2>&1 || kimi -V)"
  exit 0
fi

echo "Installing Kimi Code CLI via official install script..."
curl -LsSf https://code.kimi.com/install.sh | bash

if ! command -v kimi >/dev/null 2>&1; then
  export PATH="$HOME/.local/bin:$PATH"
fi

echo ""
echo "Installed: $(kimi --version 2>&1 || kimi -V 2>&1 || echo 'run: source ~/.zshrc')"
echo ""
echo "Next steps:"
echo "  1. kimi login          # OAuth (Kimi Code membership) — see docs"
echo "  2. cd benchmarks/spreadsheet"
echo "  3. skillopt optimize config.kimi_code.yaml"
echo ""
echo "Or use API key in ~/.kimi/config.toml (Kimi Code platform, api.kimi.com/coding/v1)"
