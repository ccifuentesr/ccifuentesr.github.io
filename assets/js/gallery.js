// Load only thumbnails near the viewport; hidden albums wait until opened.
const thumbnails = document.querySelectorAll('img[data-src]');
const observer = 'IntersectionObserver' in window ? new IntersectionObserver(entries => {
  entries.forEach(({target, isIntersecting}) => {
    if (isIntersecting) { target.src = target.dataset.src; observer.unobserve(target); }
  });
}, {rootMargin: '250px'}) : null;
function observePhotos() {
  thumbnails.forEach(img => {
    if (img.closest('[hidden]') || img.hasAttribute('src')) return;
    if (observer) observer.observe(img); else img.src = img.dataset.src;
  });
}
observePhotos();
window.addEventListener('hashchange', observePhotos);
const viewer = document.getElementById('photo-viewer');
const full = document.getElementById('photo-full');
const status = document.getElementById('photo-status');
let album = [], index = 0;
function displayPhoto() {
  const link = album[index];
  full.hidden = true;
  status.textContent = 'Loading photo…';
  full.alt = link.querySelector('img').alt;
  full.src = link.href;
  document.getElementById('photo-counter').textContent = `${index + 1} / ${album.length}`;
  const caption = link.closest('.seen-photo, .haus-photo')?.querySelector('.seen-caption');
  document.getElementById('photo-caption').textContent = caption ? caption.textContent.replace(/\s+/g, ' ').trim() : full.alt;
  document.getElementById('photo-prev').disabled = index === 0;
  document.getElementById('photo-next').disabled = index === album.length - 1;
}
full.addEventListener('load', () => { full.hidden = false; status.textContent = ''; });
full.addEventListener('error', () => { status.textContent = 'This photo could not be loaded. Please try again.'; });
document.querySelectorAll('.photo-open').forEach(link => link.addEventListener('click', event => {
  if (event.ctrlKey || event.metaKey || event.shiftKey || event.altKey) return;
  event.preventDefault();
  album = [...link.closest('[data-photo-panel]').querySelectorAll('.photo-open')];
  index = album.indexOf(link);
  viewer.showModal(); displayPhoto();
}));
function movePhoto(step) { if (index + step >= 0 && index + step < album.length) { index += step; displayPhoto(); } }
document.getElementById('photo-prev').addEventListener('click', () => movePhoto(-1));
document.getElementById('photo-next').addEventListener('click', () => movePhoto(1));
document.getElementById('photo-close').addEventListener('click', () => viewer.close());
viewer.addEventListener('click', event => { if (event.target === viewer) viewer.close(); });
viewer.addEventListener('keydown', event => {
  if (event.key === 'ArrowRight') { event.preventDefault(); movePhoto(1); }
  if (event.key === 'ArrowLeft') { event.preventDefault(); movePhoto(-1); }
});
viewer.addEventListener('close', () => { full.removeAttribute('src'); full.hidden = true; });
window.addEventListener('hashchange', () => { if (viewer.open) viewer.close(); });
// The worker caches only generated gallery images; never pages, forms or API calls.
if ('serviceWorker' in navigator && window.isSecureContext) {
  navigator.serviceWorker.register('gallery-sw.js').catch(() => { /* Gallery still works without persistent caching. */ });
}
