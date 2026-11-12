# Carlos Cifuentes — personal website

Buildless HTML site. Open locally with `python3 -m http.server 8765` and visit http://localhost:8765.

## Pages
- `index.html`: professional introduction and selected projects.
- `projects.html`: project descriptions, tools/skills and interactive research tools.
- `teaching.html`: teaching portfolio and course descriptions; materials available on request.
- `cv.html`: curriculum, with print styles.
- `publications.html`: selected publications and thesis.
- `contact.html`: compatibility redirect to the collapsible Formspree contact form after Languages in `cv.html`.
- `musik.html`, `seen.html`: personal collections, linked under Beyond work.
- `portfolio.html`: compatibility redirect only; no duplicated portfolio content.

## Assets
- `assets/css/site.css`: the only stylesheet, including responsive and print rules.
- `assets/js/site.js`: print action, music sections and tabs.
- `assets/js/projects.js`: thesis and VizieR tools, loaded only by Projects.
- `assets/js/contact.js`: contact submission and contact anchor expansion, loaded only by CV.
- `assets/js/musik.js`: original music player and lyrics, loaded only by Musik.
- `assets/js/redirect.js`: old portfolio hash routes.
- `media/`: original audio and photography.
- `backend/`: existing thesis service and index; not a deprecated frontend asset.

No build step or client-side routing framework is required. Navigation remains normal HTML links. Keep navigation/footer links consistent when adding a page. Project tool labels are based on the previous portfolio; update them only when supported by the project itself. CV facts originate from cifuentes-cv.tex.

The `.env` file is local service configuration and is not a frontend asset. It is not read or copied by this redesign.

## Photo gallery
`media/photos/` preserves the originals. `media/gallery/` contains 800 px WebP previews and 2200 px viewing versions (never upscaled), named with a source-content hash. `assets/js/gallery.js` loads previews near the viewport and opens large photos on demand. `gallery-sw.js` caches only these generated images, up to 240 entries, over HTTPS or localhost. Browser storage can be cleared or evicted. When replacing an image, generate new hashed derivatives and update its links; do not overwrite a hashed image with different content.

## ADS citation counts

`assets/data/ads-citations.json` contains public counts and their retrieval date for all selected publications, including the thesis. It also stores the ADS-derived publication count, citation total, h-index and first-to-third-author count used in the CV. `assets/js/publications.js` displays article counts below each publication description (green for first-author work, grey for co-authored articles), with links to each ADS citations page. The browser never calls the authenticated ADS API.

To refresh locally, set `ADS_API_TOKEN` in the process environment and run `python3 scripts/update_ads_citations.py`. Credentials must not be committed. Partial or invalid responses leave the previous cache untouched; missing counts are not shown as zero.

The `ads-citations.yml` workflow updates the cache in its build workspace every Monday at 06:23 UTC, on pushes to main, and on manual dispatch, then deploys the static site directly. Activation requires an Actions repository secret named `ADS_API_TOKEN`, the workflow pushed to main, and Settings → Pages → Source set to **GitHub Actions**. It intentionally deploys within the same workflow: pushes made with `GITHUB_TOKEN` do not trigger branch-based Pages builds. Cached counts updated during a build are included in that deployment without committing generated changes. The latest successful counts are also saved in the Actions cache and restored before the next update. If ADS fails, that previous snapshot is used with its original date; the committed snapshot is the fallback if no Actions cache is available. `scripts/prepare_pages.py` includes only public static pages, root images/JavaScript, assets and media; backend code and local configuration are excluded.
