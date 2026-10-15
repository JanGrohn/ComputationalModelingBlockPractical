#!/usr/bin/env bash
# Export the slide decks to PDF, one page per slide.
#
#   ./docs/export-pdf.sh                 # every deck that exists
#   ./docs/export-pdf.sh session1        # just one
#
# DeckTape drives a real browser and screenshots each slide, which is the only
# route that reliably gives one page per slide: reveal.js's own ?print-pdf mode
# rearranges the DOM with JavaScript and does not always finish before the
# browser prints.
#
# The ?export flag swaps each figure's live sliders for a line of text saying
# what the figure is set to, so the PDF still explains itself.
#
# Needs node (for npx) and Google Chrome. Nothing is installed permanently.

set -euo pipefail

cd "$(dirname "$0")"
decks=("$@")
if [ ${#decks[@]} -eq 0 ]; then
  decks=()
  for d in session*/; do [ -f "$d/index.html" ] && decks+=("${d%/}"); done
fi

chrome="/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
[ -x "$chrome" ] || chrome="$(command -v google-chrome || command -v chromium || true)"
if [ -z "$chrome" ]; then
  echo "Could not find Google Chrome. Set \$chrome in this script to its path." >&2
  exit 1
fi

port=8731
python3 -m http.server "$port" >/dev/null 2>&1 &
server=$!
trap 'kill $server 2>/dev/null || true' EXIT
sleep 2

mkdir -p pdf
for deck in "${decks[@]}"; do
  echo "==> $deck"
  PUPPETEER_SKIP_DOWNLOAD=1 npx -y decktape@3.14.0 \
    --chrome-path "$chrome" \
    -s 1280x720 -p 1200 \
    reveal "http://localhost:$port/$deck/?export" "pdf/$deck.pdf"
done

echo
echo "Written to docs/pdf/:"
ls -la pdf/*.pdf | awk '{printf "  %7.1f KB  %s\n", $5/1024, $9}'
