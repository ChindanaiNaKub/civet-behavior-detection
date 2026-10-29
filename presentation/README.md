# CMZ Phase 2 deck

A 25-slide, 16:9 reveal.js deck. It opens offline, with reveal.js and the fonts
vendored under `vendor/` and `assets/fonts/`. Nothing loads from a CDN, so a room
without wifi is fine.

## Present

Open `index.html` in a browser. A local server is optional:

```
python3 -m http.server 8000
```

then `http://127.0.0.1:8000/index.html`.

| Key | Action |
| --- | --- |
| `→` `space` `PgDn` | next slide or fragment |
| `←` `PgUp` | previous |
| `Esc` `O` | slide overview |
| `S` | speaker view, with the notes under each slide |
| `F` | fullscreen |
| `Alt` + click | zoom into a slide |
| `Ctrl` + `Shift` + `F` | search slide text |

## Export

`CMZ-Phase-2-Deck.pdf` is the current export, one page per slide.

To regenerate it, open `index.html?print-pdf` in Chrome or Chromium and print the
page to PDF: margins **none**, background graphics **on**, headers and footers
**off**. Reveal lays the slides out one per page.

```
http://127.0.0.1:8000/index.html?print-pdf
```

The page size is fixed to 1280 × 720 px in `deck.css`, so every slide is one page.
Headless `--print-to-pdf` alone prints a single page, because reveal's print layout
settles after the load event; use the browser dialog or drive it with Puppeteer.

## Edit

- `index.html` holds all 25 slides in order. Each `<section>` is one slide.
- `deck.css` holds the palette, type scale, and the layout classes
  (`cover`, `divider`, `split`, `stats`, `rows`, `matrix`, `stepper`, `term`, `lanes`).
- Speaker notes live in `<aside class="notes">` inside their slide.

## Content sources

Every number comes from the project reports, not from this folder:
`docs/EVALUATION-REPORT.md`, `docs/TASKS-2-AND-3.md`, `docs/FEATURES.md`,
`data/source-clips.csv` and `docs/adr/`.

Images are frames from the committed footage (`data/raw/`) and figures from
`docs/figures/`. The loop on slide 5 is clip `547292`, re-encoded to 1280 px wide
with the burned-in demo bands removed.
