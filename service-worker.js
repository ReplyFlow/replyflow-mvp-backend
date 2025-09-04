/* ReplyFlow PWA Service Worker
 *
 * This service worker provides basic caching to enable offline access to
 * the core pages of the ReplyFlow web application. During the install
 * phase it pre‑caches a set of essential assets. For each fetch request
 * it will respond with a cached version if available, falling back to
 * the network if the resource hasn't been cached yet. This is a
 * straightforward "cache first" strategy suitable for a simple PWA.
 */

// Bump the cache version to ensure updated assets (like dashboard.html) are
// fetched after deployment. When this version changes, the old cache will be
// purged and a new cache will be created during installation.
const CACHE_NAME = 'replyflow-cache-v3';

// List of URLs to pre‑cache. We include the root page and key static assets.
const PRECACHE_URLS = [
  '/',
  '/index.html',
  '/contact.html',
  '/manage_subscription.html',
  '/terms.html',
  '/privacy.html',
  // Pre‑cache the dashboard to ensure that updates to it are served and that
  // it is available offline. If the file changes, bump CACHE_NAME above.
  '/dashboard.html',
  // Beta signup page for early access users
  '/replyflow_signup_page.html',
  // FAQ page for frequently asked questions
  '/faq.html',
  '/static/manifest.json',
  '/static/pwa-icon-192.png',
  '/static/pwa-icon-512.png'
];

// Install event: cache static assets
self.addEventListener('install', (event) => {
  event.waitUntil(
    caches.open(CACHE_NAME).then((cache) => {
      return cache.addAll(PRECACHE_URLS);
    })
  );
});

// Activate event: clean up old caches if necessary
self.addEventListener('activate', (event) => {
  event.waitUntil(
    caches.keys().then((cacheNames) => {
      return Promise.all(
        cacheNames.map((cacheName) => {
          if (cacheName !== CACHE_NAME) {
            return caches.delete(cacheName);
          }
        })
      );
    })
  );
});

// Fetch event: respond with cache or network
self.addEventListener('fetch', (event) => {
  // Only handle GET requests
  if (event.request.method !== 'GET') {
    return;
  }
  event.respondWith(
    caches.match(event.request).then((cachedResponse) => {
      // Serve cached response if found
      if (cachedResponse) {
        return cachedResponse;
      }
      // Else fetch from network and cache the result
      return fetch(event.request).then((networkResponse) => {
        // Skip caching opaque responses (e.g. cross‑origin resources)
        if (!networkResponse || networkResponse.status !== 200 || networkResponse.type === 'opaque') {
          return networkResponse;
        }
        const responseClone = networkResponse.clone();
        caches.open(CACHE_NAME).then((cache) => {
          cache.put(event.request, responseClone);
        });
        return networkResponse;
      });
    }).catch(() => {
      // Fallback: if both cache and network fail, serve a generic offline page
      return new Response('<h1>Offline</h1><p>You are offline and the requested resource is not cached.</p>', {
        headers: { 'Content-Type': 'text/html' }
      });
    })
  );
});