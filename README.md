# Carlos Cifuentes — personal website

Buildless HTML site. Open locally with `python3 -m http.server 8765` and visit http://localhost:8765.

## Pages
- `index.html`: professional introduction and selected projects.
- `projects.html`: project descriptions, tools/skills and interactive research tools.
- `cv.html`: curriculum, with print styles.
- `publications.html`: selected publications and thesis.
- `contact.html`: existing Formspree form.
- `musik.html`, `seen.html`: personal collections, linked under Beyond work.
- `portfolio.html`: compatibility redirect only; no duplicated portfolio content.

## Assets
- `assets/css/site.css`: the only stylesheet, including responsive and print rules.
- `assets/js/site.js`: print action, music sections and tabs.
- `assets/js/projects.js`: thesis and VizieR tools, loaded only by Projects.
- `assets/js/contact.js`: contact submission, loaded only by Contact.
- `assets/js/musik.js`: original music player and lyrics, loaded only by Musik.
- `assets/js/redirect.js`: old portfolio hash routes.
- `media/`: original audio and photography.
- `backend/`: existing thesis service and index; not a deprecated frontend asset.

No build step or client-side routing framework is required. Navigation remains normal HTML links. Keep navigation/footer links consistent when adding a page. Project tool labels are based on the previous portfolio; update them only when supported by the project itself. CV facts originate from cifuentes-cv.tex.

The `.env` file is local service configuration and is not a frontend asset. It is not read or copied by this redesign.

## Photo gallery
`media/photos/` preserves the originals. `media/gallery/` contains 800 px WebP previews and 2200 px viewing versions (never upscaled), named with a source-content hash. `assets/js/gallery.js` loads previews near the viewport and opens large photos on demand. `gallery-sw.js` caches only these generated images, up to 240 entries, over HTTPS or localhost. Browser storage can be cleared or evicted. When replacing an image, generate new hashed derivatives and update its links; do not overwrite a hashed image with different content.
