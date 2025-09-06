// ReplyFlow Service Worker
//
// This service worker implements a simple caching strategy to support
// offline usage and faster repeat visits.  It caches a set of core
// application shell files during the install phase and then serves
// requests from the cache if available.  For same‑origin GET requests
// that aren't yet cached, it fetches from the network and stores
// successful responses back into the cache for future use.

const CACHE_NAME = 'replyflow-cache-v1';
const urlsToCache = [
  '/',
  '/index.html',
  '/login.html',
  '/signup.html',
  '/dashboard.html',
  '/favicon.png',
  '/static/manifest.json'
];

// Install event: cache core assets
self.addEventListener('install', event => {
  event.waitUntil(
    caches.open(CACHE_NAME).then(cache => {
      return cache.addAll(urlsToCache);
    })
  );
});

// Fetch event: respond from cache if possible, otherwise fetch from network and cache the result.
self.addEventListener('fetch', event => {
  // Only handle GET requests
  if (event.request.method !== 'GET') {
    return;
  }
  const url = new URL(event.request.url);
  // Only handle same‑origin requests
  if (url.origin === location.origin) {
    event.respondWith(
      caches.match(event.request).then(cachedResponse => {
        if (cachedResponse) {
          return cachedResponse;
        }
        return fetch(event.request)
          .then(networkResponse => {
            // Cache the response for future visits
            return caches.open(CACHE_NAME).then(cache => {
              cache.put(event.request, networkResponse.clone());
              return networkResponse;
            });
          })
          .catch(() => {
            // Provide a fallback response when offline and the resource isn't cached
            return new Response('You are offline and the requested resource is not cached.', {
              status: 503,
              headers: { 'Content-Type': 'text/plain' }
            });
          });
      })
    );
  }
});