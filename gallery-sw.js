const CACHE = 'cifuentes-gallery-v1';
const LIMIT = 240;
self.addEventListener('install', () => self.skipWaiting());
self.addEventListener('activate', event => event.waitUntil(self.clients.claim()));
self.addEventListener('fetch', event => {
  const url = new URL(event.request.url);
  const directory = new URL('media/gallery/', self.registration.scope);
  if (event.request.method !== 'GET' || url.origin !== directory.origin || !url.pathname.startsWith(directory.pathname) || !url.pathname.endsWith('.webp')) return;
  event.respondWith((async () => {
    const cache = await caches.open(CACHE);
    const cached = await cache.match(event.request);
    if (cached) return cached;
    const response = await fetch(event.request);
    if (response.ok) {
      try {
        await cache.put(event.request, response.clone());
        const keys = await cache.keys();
        await Promise.all(keys.slice(0, Math.max(0, keys.length - LIMIT)).map(key => cache.delete(key)));
      } catch (_) { /* Storage quotas must not prevent the image being displayed. */ }
    }
    return response;
  })());
});
