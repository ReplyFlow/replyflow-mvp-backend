# ReplyFlow MVP API

This directory contains a minimal FastAPI-based backend for the ReplyFlow micro‑SaaS. The purpose of this MVP is to provide a simple HTTP endpoint that accepts a social media comment and returns a draft reply in a desired tone. The project is structured to be easy to extend with additional routes and business logic as the product matures.

## Features

- **Single endpoint**: `POST /generate_reply` accepts a JSON payload with a `comment` and optional `tone` and returns a draft reply.
- **OpenAI integration**: If an OpenAI API key is available in your environment (`OPENAI_API_KEY`), the service will call the OpenAI Chat Completion API to produce the reply using the GPT‑3.5 Turbo model.
- **Fallback mechanism**: When the OpenAI API is not configured or available (e.g., during local development without internet), the service will fall back to a deterministic placeholder reply. This ensures the endpoint remains functional in constrained environments.

## Getting Started

1. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

2. **Set your OpenAI API key** (optional but recommended for meaningful replies):

   ```bash
   export OPENAI_API_KEY=sk-your-key-here
   ```

3. **Run the server**

   ```bash
   uvicorn main:app --reload
   ```

   The server will start on `http://127.0.0.1:8000` by default. The `--reload` flag enables auto‑reloading during development.

5. **Create a user and log in** (optional):

   To use endpoints that require authentication (such as the Facebook reply helpers), you must first create a user and log in. Use `curl` or a similar tool to register and obtain a session token:

   ```bash
   # Register a new user
   curl -X POST http://127.0.0.1:8000/signup \
     -H "Content-Type: application/json" \
     -d '{"email": "user@example.com", "password": "securepass"}'

   # Log in with the same credentials to receive an access token
   curl -X POST http://127.0.0.1:8000/login \
     -H "Content-Type: application/json" \
     -d '{"email": "user@example.com", "password": "securepass"}'
   # The response will include an `access_token` which you must include in the
   # Authorization header (e.g. `Authorization: Bearer <token>`) when calling
   # protected endpoints.
   ```

4. **Test the endpoint**

   Use `curl`, Postman or any HTTP client to test:

   ```bash
   curl -X POST http://127.0.0.1:8000/generate_reply \
     -H "Content-Type: application/json" \
     -d '{"comment": "Great product!", "tone": "enthusiastic"}'
   ```

   The response will be a JSON object containing a `reply` string. If the OpenAI API is configured, it will reflect the requested tone; otherwise it will be a placeholder.

## Facebook OAuth Flow

This project includes a basic example of how to integrate Facebook OAuth so that
users can connect their Facebook Pages and let ReplyFlow post replies on their
behalf. To enable this feature:

1. **Create a Facebook app** at <https://developers.facebook.com>. Add the
   “Facebook Login” and “Pages API” products. In your app settings, add the
   redirect URI you intend to use (for example `http://localhost:8000/facebook/callback`).
2. **Set environment variables** for your app credentials and redirect URI:

   ```bash
   export FACEBOOK_APP_ID=your-app-id
   export FACEBOOK_APP_SECRET=your-app-secret
   export FACEBOOK_REDIRECT_URI=http://localhost:8000/facebook/callback
   ```

3. **Start the server** and navigate your browser to `/facebook/login`. You will
   be redirected to Facebook to authorize the requested scopes
   (`pages_manage_engagement,pages_read_engagement,pages_read_user_engagement`).
4. **Complete the authorization**. Facebook will redirect back to your
   configured callback with a `code`. The `/facebook/callback` route exchanges
   this code for a user access token and retrieves the Page tokens using
   `GET https://graph.facebook.com/v18.0/me/accounts`. In this demo implementation
   the handler stores each Page token to a local JSON file and renders a simple
   success page. In a production environment you should securely store these
   tokens (e.g. in a database) and associate them with your application user.

Once you have at least one Page token stored on the server, you can generate
and post replies on behalf of those Pages. The API exposes two helper
endpoints for this purpose:

- `POST /facebook/prepare_replies` – Fetches recent posts and comments for
  each stored Page and uses the same logic as `/generate_reply` to craft
  draft responses. The endpoint accepts optional query parameters `tone`,
  `max_posts` and `max_comments`. It returns a list of objects containing
  the post ID, comment ID, original comment text and the suggested reply.
  No replies are posted to Facebook when calling this endpoint; it is
  intended for review or moderation.

- `POST /facebook/post_replies` – Performs the same preparation as above
  and then iterates through the prepared replies to publish them via
  `POST /{comment_id}/comments` on Facebook. Because posting replies
  creates external side effects, you should prompt the user for explicit
  confirmation before invoking this endpoint. The response is a
  dictionary summarizing how many replies were successfully posted per
  Page.

You can experiment with these endpoints using the included `example_client.py`
script. This script demonstrates how to call the various API routes from
Python:

```bash
# Generate a single reply for an arbitrary comment
python3 example_client.py --comment "Great product!" --tone friendly

# Prepare draft replies for recent comments on connected Pages
python3 example_client.py --prepare --tone friendly --max-posts 3 --max-comments 5

# Post the prepared replies (this will publish replies on Facebook)
python3 example_client.py --post --tone friendly

## Stripe Webhook Integration

To automatically provision or update user subscriptions when customers complete
a checkout on Stripe, this API exposes a `/stripe/webhook` endpoint. You must
configure this endpoint as a webhook in your Stripe dashboard and provide the
`STRIPE_WEBHOOK_SECRET` so the server can verify incoming events. When a
`checkout.session.completed` event is received, the webhook handler will look
for `metadata.plan` and `metadata.user_id` on the session to assign a
subscription plan (e.g. "solo" or "pro") to the corresponding user. You can
also configure price IDs via the `STRIPE_PRICE_SOLO` and `STRIPE_PRICE_PRO`
environment variables to map price IDs to plan names.

Example environment setup:

```bash
export STRIPE_WEBHOOK_SECRET=whsec_your_webhook_signing_secret
export STRIPE_PRICE_SOLO=price_1234abcd
export STRIPE_PRICE_PRO=price_5678wxyz
```

After setting up your webhook in Stripe (with the correct endpoint URL and
secret), Stripe will send event notifications to your server. The handler in
`main.py` performs a minimal signature verification (in production you should
use the official Stripe library) and updates the user plan accordingly.
```

By default the script assumes the API is running on `http://127.0.0.1:8000`.
Use the `--base-url` option to target a different host or port.

This flow follows Facebook’s requirements for commenting on posts: you must
request the appropriate permissions and use Page access tokens to post
comments or replies【786512023695782†L41-L49】【786512023695782†L91-L104】. See
the `/facebook/callback` implementation for details.

### Persisting Tokens

The application uses a simple file-based token store (`facebook_tokens.json`) to
persist tokens across server restarts. Each time a user completes the OAuth
flow, the callback handler stores the user access token and each Page token
returned by Facebook in this JSON file. You can override the storage path by
setting the `FACEBOOK_TOKEN_STORE` environment variable.

To view the stored tokens (for debugging only), call the `/admin/facebook_tokens`
endpoint. **Important:** this route is unsecured and should not be exposed in
production. You should replace this mechanism with a proper database and
authentication before going live.

## Extending the MVP

This MVP is intentionally minimal. To evolve it into a full product, consider adding:

- **OAuth and API integrations**: Build flows to connect users’ Instagram, X, Reddit or Facebook accounts for comment retrieval and posting. Each platform has its own authentication scheme and permission scopes. An example Facebook OAuth implementation is included under `facebook_login` and `facebook_callback` routes; it exchanges authorization codes for user and Page access tokens.
- **Persistent storage**: Add a database layer to store user profiles, connected accounts, and usage metrics.
- **Rate limiting and quotas**: Enforce usage limits per subscription tier to keep API costs predictable.
- **Management endpoints**: Expose routes for managing tone presets, subscription billing details (e.g. via Stripe), and integrations with Slack or Notion for notifications.

Feel free to adapt the architecture to your preferred backend framework (e.g. Node.js/Express) as you scale.