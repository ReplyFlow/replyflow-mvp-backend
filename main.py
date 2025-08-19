"""
Minimal FastAPI server for ReplyFlow MVP.

This server exposes a simple API endpoint that accepts a social media
comment along with a desired tone, and returns a draft reply. It is
designed to power the ReplyFlow micro‑SaaS, which helps creators reply
quickly and on‑brand across platforms like Instagram, X, Reddit and
Facebook. The logic here is intentionally lightweight so it can serve
as a drop‑in microservice within a larger system.

Key characteristics:

* **OpenAI integration** – if an API key is provided via the `OPENAI_API_KEY`
  environment variable, the service uses OpenAI's Chat Completion API to
  craft context‑aware replies. This enables high‑quality responses that
  respect the requested tone.
* **Fallback mechanism** – in development or offline scenarios where
  OpenAI isn't available, the endpoint falls back to a deterministic
  placeholder reply. This ensures that the API remains functional even
  without external dependencies.

To run the server locally:

    export OPENAI_API_KEY=your-key-here
    python3 -m uvicorn main:app --reload --port 8000

After starting, you can test the API with curl:

    curl -X POST http://localhost:8000/generate_reply \
      -H "Content-Type: application/json" \
      -d '{"comment": "Love this product!", "tone": "friendly"}'

The response will contain a JSON object with a draft reply. In a full
deployment, this endpoint would be invoked by the backend after pulling
comments from supported platforms via their respective APIs.
"""

import os
import json
import urllib.parse
import uuid
import datetime
import hmac
import hashlib
import secrets
from typing import Optional, Dict, Any

from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.responses import RedirectResponse, HTMLResponse
from pydantic import BaseModel

try:
    import openai  # type: ignore
    _openai_available = True
except Exception:
    # If openai can't be imported (no network or not installed), we'll fallback
    openai = None  # type: ignore
    _openai_available = False

# Import httpx if available for making HTTP requests to external APIs (e.g. Facebook).
try:
    import httpx  # type: ignore
    _httpx_available = True
except Exception:
    httpx = None  # type: ignore
    _httpx_available = False

# Import TokenStore for storing platform tokens per user. The token_store.py file
# defines a SQLite-backed TokenStore class which persists tokens keyed by
# user and provider. We import it here so it can be used to replace the
# legacy FacebookTokenStore defined below.
from token_store import TokenStore  # type: ignore


class GenerateReplyRequest(BaseModel):
    comment: str
    tone: Optional[str] = "friendly"


class GenerateReplyResponse(BaseModel):
    reply: str


app = FastAPI(title="ReplyFlow MVP API", version="0.1.0")

# Enable CORS so that the frontend hosted on a different domain can
# communicate with this API. Without CORS headers, browsers will block
# requests made from Netlify or other origins during the preflight check.
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, restrict this to your frontend domain(s)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----------------------------------------------------------------------------
# Facebook OAuth configuration
#
# To enable Facebook integration, you must create an app on
# https://developers.facebook.com, add the "Facebook Login" product and
# request the appropriate Page permissions. Define your app credentials and
# redirect URI via environment variables as shown below. The redirect URI
# must be added to your Facebook app's OAuth settings.
#
# Environment variables used:
#   FACEBOOK_APP_ID:     Your Facebook App ID
#   FACEBOOK_APP_SECRET: Your Facebook App Secret
#   FACEBOOK_REDIRECT_URI: The URI on your server where Facebook should
#                          redirect users after they grant permissions.
#
# When running locally you might set FACEBOOK_REDIRECT_URI to something like
# "http://localhost:8000/facebook/callback". In production it should be
# whichever domain points at this service.

FACEBOOK_SCOPES = [
    # Request permissions needed to read page posts and comments, manage posts,
    # view the list of Pages, and read user-generated content. The
    # pages_manage_metadata permission is required in some Graph API versions
    # to retrieve certain metadata fields (like comment IDs) and perform the
    # MODERATE task on a Page. Without this permission, the comments endpoint
    # may return empty results for some Pages.
    "pages_manage_engagement",
    "pages_manage_posts",
    "pages_read_engagement",
    "pages_read_user_content",
    "pages_manage_metadata",
    "pages_show_list",
]

class FacebookTokenStore:
    """A simple file-based store for Facebook tokens.

    Tokens are persisted to a JSON file so that they survive server restarts.
    Each entry in the store is a dictionary with keys:
      - page_id: The ID of the Facebook Page
      - page_access_token: The token that allows posting on behalf of the Page
      - user_access_token: The user-level access token returned by OAuth
      - created_at: UTC timestamp when the token was stored

    In a production system you should encrypt these values and associate
    them with your application user. This store is only for demonstration.
    """

    def __init__(self, path: str | None = None) -> None:
        self.path = path or os.getenv("FACEBOOK_TOKEN_STORE", "facebook_tokens.json")
        # Ensure file exists
        if not os.path.exists(self.path):
            with open(self.path, "w", encoding="utf-8") as fh:
                json.dump([], fh)

    def _load(self) -> list[dict]:
        try:
            with open(self.path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
                return data if isinstance(data, list) else []
        except Exception:
            return []

    def save_token(self, page_id: str, page_access_token: str, user_access_token: str) -> None:
        tokens = self._load()
        tokens.append({
            "page_id": page_id,
            "page_access_token": page_access_token,
            "user_access_token": user_access_token,
            "created_at": datetime.datetime.utcnow().isoformat() + "Z",
        })
        with open(self.path, "w", encoding="utf-8") as fh:
            json.dump(tokens, fh)

    def list_tokens(self) -> list[dict]:
        return self._load()


# Initialize a global token store. We use the SQLite-backed TokenStore to persist
# page tokens per-user and provider. This avoids overwriting tokens when
# multiple users connect different Facebook Pages. The REPLYFLOW_TOKEN_STORE
# environment variable can override the database location.
token_store = TokenStore()

# Maintain an in-memory mapping from OAuth "state" values to the user ID of
# the person initiating the Facebook login. When Facebook redirects back to
# our callback endpoint we use this mapping to associate the returned code
# with the correct user. Because this mapping lives only in memory, the
# Facebook login flow will not work across server restarts without additional
# persistence. See facebook_login() and facebook_callback() for usage.
pending_facebook_states: dict[str, str] = {}

# -----------------------------------------------------------------------------
# Simple user and session management
#
# This implementation provides a minimal authentication system suitable for the
# ReplyFlow MVP. Users are persisted in a JSON file with hashed passwords and
# associated plans. A session token is issued upon successful login and must
# be included in the "Authorization" header as a Bearer token for protected
# endpoints. In production you should migrate to a proper database and use
# standard JWT or OAuth flows instead of this ad‑hoc mechanism.

USERS_FILE = os.getenv("REPLYFLOW_USERS_FILE", "users.json")

class UserStore:
    """A basic file‑based user store.

    Users are stored in a JSON list with fields:
      - id: unique identifier
      - email: email address
      - password_hash: SHA256 hash of password and salt
      - salt: per‑user salt used for hashing
      - plan: the subscribed plan (e.g. "solo", "pro" or None)

    The user store persists across server restarts. In a multi‑instance
    deployment you should use a real database instead.
    """

    def __init__(self, path: str = USERS_FILE) -> None:
        self.path = path
        # ensure the file exists
        if not os.path.exists(self.path):
            with open(self.path, "w", encoding="utf-8") as fh:
                json.dump([], fh)

    def _load(self) -> list[dict]:
        try:
            with open(self.path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
                return data if isinstance(data, list) else []
        except Exception:
            return []

    def _save(self, users: list[dict]) -> None:
        with open(self.path, "w", encoding="utf-8") as fh:
            json.dump(users, fh)

    def find_by_email(self, email: str) -> Optional[dict]:
        email = email.strip()
        for user in self._load():
            if user.get("email") == email:
                return user
        return None

    def add_user(self, email: str, password: str) -> dict:
        users = self._load()
        if any(u.get("email") == email for u in users):
            raise ValueError("User already exists")
        salt = secrets.token_hex(16)
        password_hash = self._hash_password(password, salt)
        user = {
            "id": uuid.uuid4().hex,
            "email": email,
            "salt": salt,
            "password_hash": password_hash,
            "plan": None,
        }
        users.append(user)
        self._save(users)
        return user

    def verify_user(self, email: str, password: str) -> Optional[dict]:
        user = self.find_by_email(email)
        if not user:
            return None
        expected = user.get("password_hash")
        salt = user.get("salt")
        if not expected or not salt:
            return None
        provided = self._hash_password(password, salt)
        if hmac.compare_digest(provided, expected):
            return user
        return None

    def update_plan(self, user_id: str, plan: Optional[str]) -> None:
        users = self._load()
        for user in users:
            if user.get("id") == user_id:
                user["plan"] = plan
                break
        self._save(users)

    @staticmethod
    def _hash_password(password: str, salt: str) -> str:
        return hashlib.sha256((salt + password).encode("utf-8")).hexdigest()


class SessionStore:
    """In‑memory session store mapping session tokens to user IDs.

    A new session token is generated upon successful login. Sessions persist
    only in memory, so restarting the server will require users to log in
    again. In production you should persist sessions or use JWTs.
    """
    def __init__(self) -> None:
        self.sessions: Dict[str, str] = {}

    def create_session(self, user_id: str) -> str:
        token = secrets.token_urlsafe(32)
        self.sessions[token] = user_id
        return token

    def get_user_id(self, token: str) -> Optional[str]:
        return self.sessions.get(token)

    def delete_session(self, token: str) -> None:
        self.sessions.pop(token, None)


# Instantiate global stores
user_store = UserStore()
session_store = SessionStore()


def get_current_user(request: Request) -> dict:
    """Dependency to retrieve the currently authenticated user.

    Reads the Authorization header and validates the session token. If the
    token is valid, returns the corresponding user record. Otherwise raises
    a 401 error.
    """
    auth_header = request.headers.get("Authorization")
    if not auth_header or not auth_header.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or invalid Authorization header")
    token = auth_header.split(" ", 1)[1].strip()
    user_id = session_store.get_user_id(token)
    if not user_id:
        raise HTTPException(status_code=401, detail="Invalid session token")
    # Find user by ID
    users = user_store._load()
    user = next((u for u in users if u.get("id") == user_id), None)
    if not user:
        raise HTTPException(status_code=401, detail="User not found")
    return user

async def _prepare_facebook_replies(user_id: str, tone: str = "friendly", max_posts: int = 3, max_comments: int = 5) -> list[dict]:
    """Fetch recent comments from stored Facebook pages and generate draft replies for a user.

    This helper function retrieves a limited number of recent posts for each page
    stored in the token store, then fetches a limited number of comments on
    each post. For each comment, it uses the same logic as the /generate_reply
    endpoint to produce a reply in the requested tone. The returned list
    contains dicts with the original comment ID, post ID, comment text and
    the generated reply. No replies are posted in this function.

    Args:
        tone: The tone to use for generated replies (default "friendly").
        max_posts: The maximum number of recent posts to fetch per page.
        max_comments: The maximum number of comments to fetch per post.

    Returns:
        A list of dictionaries with keys: comment_id, post_id, comment,
        reply, page_id.
    """
    if not _httpx_available:
        raise RuntimeError("httpx is required for Facebook API interactions")
    prepared = []
    tone = tone.strip().lower() if tone else "friendly"
    # Fetch all stored page tokens for this user. The TokenStore API expects
    # the user_id and provider name to list relevant tokens. If no tokens
    # exist for this user the result will be an empty list, and an empty
    # replies list will be returned.
    tokens = token_store.list_tokens(user_id, "facebook")
    async with httpx.AsyncClient() as client:
        for token_info in tokens:
            page_id = token_info.get("page_id")
            page_token = token_info.get("page_access_token")
            user_token = token_info.get("user_access_token")
            if not page_id or not page_token:
                continue

            # Use the user access token, if available, to fetch posts and comments.
            # Some Graph API endpoints return richer data when using a user token (with the
            # necessary permissions) compared to the page token alone. Fallback to the
            # page token if the user token is missing.
            request_token = user_token or page_token
            # Fetch recent posts published by the page. Using the /posts edge
            # returns only posts created by the Page, whereas the /feed edge can
            # include visitor posts as well. If this request fails, skip this page.
            try:
                feed_resp = await client.get(
                    f"https://graph.facebook.com/v18.0/{page_id}/posts",
                    params={"access_token": request_token, "limit": max_posts},
                    timeout=10,
                )
                feed_data = feed_resp.json()
                # If the API responded with an error, surface it so the caller knows why
                if isinstance(feed_data, dict) and feed_data.get("error"):
                    raise HTTPException(status_code=400, detail=f"Facebook feed error: {feed_data.get('error')}")
            except Exception:
                continue
            for post in feed_data.get("data", [])[:max_posts]:
                post_id = post.get("id")
                if not post_id:
                    continue
                # Fetch comments on the post
                try:
                    # Fetch comments for the post. We specify the 'filter' parameter as 'stream'
                    # to ensure that we receive both top-level comments and replies to
                    # comments, and set an explicit order to get the newest comments first.
                    comments_resp = await client.get(
                        f"https://graph.facebook.com/v18.0/{post_id}/comments",
                        params={
                            "access_token": request_token,
                            "limit": max_comments,
                            "filter": "stream",
                            "order": "chronological",
                            "fields": "id,message"
                        },
                        timeout=10,
                    )
                    comments_data = comments_resp.json()
                    # If the API responded with an error, raise an exception to surface it to the caller
                    if isinstance(comments_data, dict) and comments_data.get("error"):
                        raise HTTPException(status_code=400, detail=f"Facebook comments error: {comments_data.get('error')}")
                except Exception:
                    continue
                for comment in comments_data.get("data", [])[:max_comments]:
                    comment_id = comment.get("id")
                    comment_message = comment.get("message", "")
                    # Skip if either ID or message is missing
                    if not comment_id or not comment_message:
                        continue
                    # Generate reply using the internal logic (OpenAI or fallback)
                    try:
                        if _openai_available and os.getenv("OPENAI_API_KEY"):
                            draft_reply = await generate_with_openai(comment_message, tone)
                        else:
                            draft_reply = generate_fallback_reply(comment_message, tone)
                    except Exception:
                        draft_reply = generate_fallback_reply(comment_message, tone)
                    prepared.append({
                        "page_id": page_id,
                        "post_id": post_id,
                        "comment_id": comment_id,
                        "comment": comment_message,
                        "reply": draft_reply,
                    })
    return prepared


@app.api_route("/facebook/prepare_replies", methods=["GET", "POST"])
async def facebook_prepare_replies(
    tone: str = "friendly",
    max_posts: int = 3,
    max_comments: int = 5,
    current_user: dict = Depends(get_current_user),
) -> list[dict]:
    """Prepare draft replies for recent comments on connected Facebook Pages.

    This endpoint uses the stored Page tokens to fetch a limited number of
    recent posts and comments, generates replies using the same logic as
    /generate_reply, and returns the prepared replies without posting them.

    It accepts both GET and POST requests: query parameters or JSON body
    values are supported for tone, max_posts and max_comments. Only the
    Authorization header is mandatory to identify the current user.

    Parameters:
        tone: The desired tone for the replies (default friendly).
        max_posts: Maximum number of recent posts to consider per Page (default 3).
        max_comments: Maximum number of comments to consider per post (default 5).

    Returns:
        A list of draft replies with comment and post context.
    """
    user_id = current_user.get("id")
    prepared = await _prepare_facebook_replies(user_id, tone, max_posts, max_comments)
    return prepared


@app.post("/facebook/post_replies")
async def facebook_post_replies(
    tone: str = "friendly",
    max_posts: int = 3,
    max_comments: int = 5,
    current_user: dict = Depends(get_current_user),
) -> dict:
    """Fetch comments and post generated replies (requires confirmation).

    This endpoint fetches recent comments and posts replies using the stored Page
    tokens. Because posting replies to Facebook is an external side effect, the
    caller must confirm before invoking this endpoint. It is separated from
    prepare_replies so that you can review the proposed replies first.

    Parameters:
        tone: The desired tone for the replies (default friendly).
        max_posts: Maximum number of recent posts to consider per Page (default 3).
        max_comments: Maximum number of comments to consider per post (default 5).

    Returns:
        A dict summarizing the number of replies posted per page.
    """
    # Prepare replies for the current user but do not post yet
    user_id = current_user.get("id")
    prepared = await _prepare_facebook_replies(user_id, tone, max_posts, max_comments)
    results: dict[str, int] = {}
    if not _httpx_available:
        raise HTTPException(status_code=500, detail="httpx is required for Facebook API interactions")
    async with httpx.AsyncClient() as client:
        for item in prepared:
            page_id = item["page_id"]
            page_token = None
            # Find the page token belonging to the current user and page
            for t in token_store.list_tokens(user_id, "facebook"):
                if t.get("page_id") == page_id:
                    page_token = t.get("page_access_token")
                    break
            if not page_token:
                continue
            comment_id = item["comment_id"]
            reply_message = item["reply"]
            try:
                await client.post(
                    f"https://graph.facebook.com/v18.0/{comment_id}/comments",
                    params={"access_token": page_token},
                    json={"message": reply_message},
                    timeout=10,
                )
                results[page_id] = results.get(page_id, 0) + 1
            except Exception:
                # Skip posting this reply if there's an error
                continue
    return {"replies_posted": results}


@app.get("/admin/facebook_tokens")
def list_facebook_tokens(current_user: dict = Depends(get_current_user)) -> list[dict]:
    """Return all stored Facebook page and user access tokens for the current user.

    This endpoint is for demonstration and debugging purposes only. In a
    production environment you should secure this route (e.g. with
    authentication) and avoid returning sensitive tokens directly.
    """
    user_id = current_user.get("id")
    return token_store.list_tokens(user_id, "facebook")


@app.get("/facebook/login")
async def facebook_login(token: Optional[str] = None) -> RedirectResponse:
    """Begin the Facebook OAuth flow for the logged in user.

    The frontend should provide the user's session token via the `token`
    query parameter. We validate the session token and generate a unique
    state value to associate the OAuth callback with the correct user.
    If the user is not authenticated or the token is missing, we
    redirect them to the frontend login page.
    """
    client_id = os.getenv("FACEBOOK_APP_ID")
    redirect_uri = os.getenv("FACEBOOK_REDIRECT_URI")
    if not client_id or not redirect_uri:
        raise HTTPException(
            status_code=500,
            detail="FACEBOOK_APP_ID and FACEBOOK_REDIRECT_URI must be set in the environment",
        )
    # Determine the session token from the query parameter
    session_token = token
    if not session_token:
        # No token provided; redirect to the frontend login page
        frontend_login = os.getenv("FRONTEND_LOGIN_URL") or "/login.html"
        return RedirectResponse(frontend_login)
    # Validate the session token and retrieve the user ID
    user_id = session_store.get_user_id(session_token)
    if not user_id:
        frontend_login = os.getenv("FRONTEND_LOGIN_URL") or "/login.html"
        return RedirectResponse(frontend_login)
    # Create a unique state value and record the mapping to the user ID. Embed the
    # session token into the state string so that if the server restarts or
    # loses the in-memory mapping, we can still recover the user ID from the
    # session token included in the state parameter during the callback.
    # We concatenate the session token and a random UUID using a colon. During
    # the callback, we will parse the session token from the state.
    random_state = uuid.uuid4().hex
    state = f"{session_token}:{random_state}"
    # Preserve the mapping as a fallback for in-memory workflows
    pending_facebook_states[random_state] = user_id
    params = {
        "client_id": client_id,
        "redirect_uri": redirect_uri,
        "scope": ",".join(FACEBOOK_SCOPES),
        "response_type": "code",
        "state": state,
    }
    auth_url = "https://www.facebook.com/v18.0/dialog/oauth"
    url = f"{auth_url}?{urllib.parse.urlencode(params)}"
    return RedirectResponse(url)


@app.get("/facebook/callback")
async def facebook_callback(code: str, state: Optional[str] = None) -> HTMLResponse:
    """Handle the OAuth callback from Facebook, exchange the code, and show a success page.

    Facebook redirects the user to this endpoint with a short-lived
    authorization code and the state value we generated in facebook_login().
    We exchange the code for a user access token, request the list of Pages
    the user manages and their page tokens, and persist those tokens. The
    response is a small HTML snippet that notifies the opener (dashboard)
    via postMessage and then closes the popup or redirects to the dashboard.
    """
    app_id = os.getenv("FACEBOOK_APP_ID")
    app_secret = os.getenv("FACEBOOK_APP_SECRET")
    redirect_uri = os.getenv("FACEBOOK_REDIRECT_URI")
    if not (app_id and app_secret and redirect_uri):
        raise HTTPException(
            status_code=500,
            detail="FACEBOOK_APP_ID, FACEBOOK_APP_SECRET and FACEBOOK_REDIRECT_URI must be set",
        )
    if not _httpx_available:
        raise HTTPException(status_code=500, detail="httpx is not available")

    # Validate and consume the state parameter to determine which user initiated the OAuth flow.
    user_id: Optional[str] = None
    session_token_in_state: Optional[str] = None
    if state:
        # If the state contains a colon, it was generated by facebook_login() and includes
        # the session token. The format is "<session_token>:<random_state>". We split it
        # to extract the session token and the random state token. We then look up the
        # user ID from the session store.
        if ":" in state:
            session_token_in_state, random_state = state.split(":", 1)
            # Try to retrieve the user ID using the session token
            if session_token_in_state:
                user_id = session_store.get_user_id(session_token_in_state)
            # Remove the random_state entry from the pending_facebook_states mapping if present
            if random_state:
                pending_facebook_states.pop(random_state, None)
        else:
            # Fallback: original behaviour where state is just a random key mapped to user_id
            user_id = pending_facebook_states.pop(state, None)

    # Exchange code for a short-lived user access token
    token_params = {
        "client_id": app_id,
        "client_secret": app_secret,
        "redirect_uri": redirect_uri,
        "code": code,
    }
    async with httpx.AsyncClient() as client:
        token_resp = await client.get(
            "https://graph.facebook.com/v18.0/oauth/access_token",
            params=token_params,
            timeout=10,
        )
        token_data = token_resp.json()
        access_token = token_data.get("access_token")
        if not access_token:
            raise HTTPException(status_code=400, detail=f"Failed to obtain access token: {token_data}")

        # Request the Pages managed by the user to obtain a page token. The
        # user must have the required permissions (e.g., pages_read_engagement). The
        # first page in the list is returned for demonstration; you may need
        # to let the user pick the relevant page.
        # Request the Pages managed by the user along with their page access tokens.
        # We explicitly request the access_token field because the default response
        # may omit it. Without the page token we cannot read comments or post replies.
        pages_resp = await client.get(
            "https://graph.facebook.com/v18.0/me/accounts",
            params={"access_token": access_token, "fields": "id,name,access_token"},
            timeout=10,
        )
        pages_data = pages_resp.json()

    # If we could not determine the user_id, we cannot save tokens; show an error message
    if not user_id:
        html_content = """
        <html>
        <head><title>Facebook Connection Error</title></head>
        <body>
            <h2>Unable to Connect Facebook</h2>
            <p>We couldn't associate this Facebook login with a signed‑in user. Please try connecting again from your dashboard.</p>
        </body>
        </html>
        """
        return HTMLResponse(content=html_content, status_code=400)

    # Store tokens for each page the user manages. Persist both the page and user tokens.
    if pages_data and "data" in pages_data:
        for page in pages_data.get("data", []):
            page_id = page.get("id")
            page_token = page.get("access_token")
            if page_id and page_token:
                token_store.save_token(
                    user_id=user_id,
                    provider="facebook",
                    page_id=page_id,
                    page_access_token=page_token,
                    user_access_token=access_token,
                )
    # Prepare a small HTML page that notifies the opener (the dashboard) that Facebook
    # has been connected. If there's no window.opener, redirect back to the dashboard.
    dashboard_url = os.getenv("FRONTEND_DASHBOARD_URL") or "/dashboard.html"
    html_content = f"""
    <html>
    <head><title>Facebook Connected</title></head>
    <body>
        <script>
        (function() {{
            if (window.opener) {{
                try {{
                    window.opener.postMessage('facebook_connected', '*');
                }} catch (e) {{
                    /* ignore cross-origin errors */
                }}
                window.close();
            }} else {{
                window.location = '{dashboard_url}';
            }}
        }})();
        </script>
        <p>Facebook Connected. You can close this window.</p>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content, status_code=200)


async def generate_with_openai(comment: str, tone: str) -> str:
    """Generate a reply using the OpenAI API.

    Args:
        comment: The original comment text.
        tone: Desired reply tone.

    Returns:
        The generated reply from OpenAI.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY environment variable is not set")
    if not _openai_available:
        raise RuntimeError("openai package is not available")

    openai.api_key = api_key
    # Compose a prompt instructing the model how to respond. Adjust the
    # guidelines here to refine the reply style.
    system_prompt = (
        "You are a helpful assistant that drafts concise and engaging "
        f"public replies in a {tone} tone. Keep the reply under 130 characters "
        "and encourage further interaction when appropriate."
    )
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": comment},
    ]
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=messages,
            max_tokens=60,
            temperature=0.7,
        )
    except Exception as exc:
        raise RuntimeError(f"OpenAI API error: {exc}")
    # Extract the assistant message content
    reply_content = response["choices"][0]["message"]["content"].strip()
    return reply_content


def generate_fallback_reply(comment: str, tone: str) -> str:
    """Generate a placeholder reply when OpenAI isn't available.

    The fallback simply reverses the comment and appends a note indicating the
    desired tone. This ensures that the API remains functional even if the
    LLM cannot be reached. In production, this fallback should be replaced or
    removed to avoid confusing end users.
    """
    reversed_comment = comment[::-1]
    return f"[fallback {tone} reply] {reversed_comment}"


@app.post("/generate_reply", response_model=GenerateReplyResponse)
async def generate_reply(req: GenerateReplyRequest) -> GenerateReplyResponse:
    """Generate a draft reply for a given comment.

    This single endpoint is platform‑agnostic: you can send in comments
    originating from Instagram, X, Reddit or Facebook and the
    service will produce a response in the requested tone. Under the hood
    it uses OpenAI when available, otherwise it returns a deterministic
    placeholder. In a production implementation, you would call this
    endpoint after fetching comments from the respective platform APIs.
    """
    comment = req.comment.strip()
    tone = (req.tone or "friendly").strip().lower()
    if not comment:
        raise HTTPException(status_code=400, detail="Comment cannot be empty")

    try:
        if _openai_available and os.getenv("OPENAI_API_KEY"):
            reply = await generate_with_openai(comment, tone)
        else:
            reply = generate_fallback_reply(comment, tone)
    except Exception as exc:
        # Fallback on any exception during generation
        reply = generate_fallback_reply(comment, tone)

    return GenerateReplyResponse(reply=reply)


# -----------------------------------------------------------------------------
# User authentication endpoints
#
# These endpoints provide minimal user registration and login functionality.
# They are not suitable for production use without additional security
# considerations (e.g. email verification, password complexity, rate limiting).

class SignupRequest(BaseModel):
    email: str
    password: str


class LoginRequest(BaseModel):
    email: str
    password: str


@app.post("/signup")
def signup(req: SignupRequest) -> dict:
    """Register a new user.

    Accepts a JSON body with `email` and `password`. If a user with the
    provided email already exists, returns a 400 error. On success,
    returns a message and the new user id.
    """
    email = req.email.strip()
    password = req.password
    if not email or not password:
        raise HTTPException(status_code=400, detail="Email and password are required")
    try:
        user = user_store.add_user(email, password)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return {"message": "User created", "user_id": user.get("id")}


@app.post("/login")
def login(req: LoginRequest) -> dict:
    """Authenticate an existing user.

    Accepts a JSON body with `email` and `password`. If the credentials are
    valid, returns a session token to include in subsequent requests.
    """
    email = req.email.strip()
    password = req.password
    if not email or not password:
        raise HTTPException(status_code=400, detail="Email and password are required")
    user = user_store.verify_user(email, password)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid credentials")
    token = session_store.create_session(user.get("id"))
    return {"access_token": token, "token_type": "bearer"}


@app.get("/me")
def me(current_user: dict = Depends(get_current_user)) -> dict:
    """Return the current authenticated user's info."""
    return {"id": current_user.get("id"), "email": current_user.get("email"), "plan": current_user.get("plan")}


# -----------------------------------------------------------------------------
# Stripe webhook handler
#
# This endpoint listens for events from Stripe to provision or update user
# subscriptions. Set the STRIPE_WEBHOOK_SECRET environment variable to
# enable signature verification. The handler currently processes
# `checkout.session.completed` events and updates the user's plan based on
# price IDs configured in STRIPE_PRICE_SOLO and STRIPE_PRICE_PRO.

@app.post("/stripe/webhook")
async def stripe_webhook(request: Request) -> dict:
    """Handle incoming Stripe webhook events.

    Stripe signs webhook events using a secret. You must set the
    STRIPE_WEBHOOK_SECRET environment variable to the value configured in
    your Stripe dashboard. The event is parsed and, if valid, processed
    accordingly. Currently only checkout.session.completed is handled.
    """
    # Read payload and signature
    payload = await request.body()
    sig_header = request.headers.get("Stripe-Signature", "")
    webhook_secret = os.getenv("STRIPE_WEBHOOK_SECRET")
    # Stripe library isn't available in this environment, so we perform a
    # simplified verification of the signature. In production you should
    # install stripe and use stripe.WebhookSignature.verify_header.
    def is_valid_signature(payload: bytes, sig_header: str, secret: str) -> bool:
        try:
            # Stripe signatures are of the form t=timestamp,v1=signature
            items = dict(x.split("=", 1) for x in sig_header.split(","))
            signature = items.get("v1")
            timestamp = items.get("t")
            if not signature or not timestamp:
                return False
            signed_payload = f"{timestamp}.".encode() + payload
            expected = hmac.new(secret.encode(), signed_payload, hashlib.sha256).hexdigest()
            return hmac.compare_digest(expected, signature)
        except Exception:
            return False
    # If secret is set and signature invalid, return 400
    if webhook_secret and not is_valid_signature(payload, sig_header, webhook_secret):
        raise HTTPException(status_code=400, detail="Invalid signature")
    try:
        event = json.loads(payload.decode())
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid payload")
    event_type = event.get("type")
    # Handle checkout.session.completed
    if event_type == "checkout.session.completed":
        data_obj = event.get("data", {}).get("object", {})
        customer_email = data_obj.get("customer_details", {}).get("email")
        # Stripe checkout sessions can carry custom metadata; we expect at least
        # a "plan" key indicating the purchased tier (e.g. "solo" or "pro").
        # Optionally a "user_id" may be passed if an existing user is upgrading.
        metadata: Dict[str, Any] = data_obj.get("metadata", {}) or {}
        plan_from_meta: Optional[str] = metadata.get("plan")
        user_id_from_meta: Optional[str] = metadata.get("user_id")
        # If a user ID is provided in metadata, simply update that user's plan.
        if plan_from_meta and user_id_from_meta:
            try:
                user_store.update_plan(user_id_from_meta, plan_from_meta)
            except Exception:
                pass
        else:
            # When no user ID is provided, attempt to create or update a user
            # based on the customer email. If the user already exists, update
            # their plan. If they don't, create them with a random password
            # and assign the purchased plan.
            # Determine the plan to assign; prefer explicit metadata and fall
            # back to environment price matching if provided.
            plan: Optional[str] = None
            if plan_from_meta:
                plan = plan_from_meta
            else:
                # Fallback: determine plan by matching price ID; you must set
                # STRIPE_PRICE_SOLO and STRIPE_PRICE_PRO environment variables
                solo_price = os.getenv("STRIPE_PRICE_SOLO")
                pro_price = os.getenv("STRIPE_PRICE_PRO")
                # The Stripe event payload may include the subscription/line item
                # but it's not reliably available without expanding line items.
                # We therefore rely on metadata for plan selection unless you
                # explicitly set price IDs and ensure they are passed here.
                # This block is left as a placeholder for future enhancement.
                line_price = None
                if line_price and (line_price == solo_price or line_price == pro_price):
                    plan = "solo" if line_price == solo_price else "pro"
            # Only proceed if we have an email and determined a plan
            if customer_email and plan:
                # Normalize email
                email_norm = customer_email.strip().lower()
                existing_user = user_store.find_by_email(email_norm)
                if existing_user:
                    # update plan for existing user
                    try:
                        user_store.update_plan(existing_user.get("id"), plan)
                    except Exception:
                        pass
                else:
                    # create a new user with a random password
                    random_password = secrets.token_urlsafe(12)
                    try:
                        new_user = user_store.add_user(email_norm, random_password)
                        user_store.update_plan(new_user.get("id"), plan)
                    except Exception:
                        # In case of race condition where user was created between
                        # find_by_email and add_user, just update plan
                        existing = user_store.find_by_email(email_norm)
                        if existing:
                            user_store.update_plan(existing.get("id"), plan)
            # If no email or plan, do nothing; we still respond success
        return {"status": "success"}
    # Always return success for unhandled event types
    return {"status": "ignored"}

# ---------------------------------------------------------------------------
# Instagram, X, and Reddit placeholder endpoints

# Instagram endpoints
@app.get("/instagram/login")
async def instagram_login() -> dict:
    """Placeholder login endpoint for Instagram OAuth.

    In a full implementation this would redirect the user to Instagram's OAuth
    authorization page and handle the callback. For now it simply returns
    a message indicating that the feature is not yet implemented.
    """
    return {"message": "Instagram login is not implemented yet. Stay tuned!"}


@app.post("/instagram/prepare_replies")
async def instagram_prepare_replies(
    tone: str = "friendly",
    max_posts: int = 3,
    max_comments: int = 5,
    current_user: dict = Depends(get_current_user),
) -> list[dict]:
    """Prepare draft replies for Instagram comments (placeholder).

    This stub endpoint returns an empty list to indicate that no comments
    were found. When Instagram integration is implemented, it should fetch
    recent comments on the user's connected Instagram accounts and call
    /generate_reply for each comment.
    """
    # Since we don't yet have an Instagram API integration, return an empty list.
    return []


@app.post("/instagram/post_replies")
async def instagram_post_replies(
    tone: str = "friendly",
    max_posts: int = 3,
    max_comments: int = 5,
    current_user: dict = Depends(get_current_user),
) -> dict:
    """Post prepared replies to Instagram (placeholder).

    This stub endpoint does nothing and returns a summary message.
    In the future it should iterate over prepared replies and post them
    to the corresponding Instagram comments via the Instagram Graph API.
    """
    return {"status": "Instagram posting not implemented yet"}

# X (formerly Twitter) endpoints
@app.get("/x/login")
async def x_login() -> dict:
    """Placeholder login endpoint for X OAuth.

    This stub simply returns a message indicating that the X login flow
    hasn't been implemented. A real implementation would redirect the user
    to X's OAuth dialog and handle the callback.
    """
    return {"message": "X (formerly Twitter) login is not implemented yet. Stay tuned!"}


@app.post("/x/prepare_replies")
async def x_prepare_replies(
    tone: str = "friendly",
    max_posts: int = 3,
    max_comments: int = 5,
    current_user: dict = Depends(get_current_user),
) -> list[dict]:
    """Prepare draft replies for X tweets (placeholder).

    This stub endpoint returns an empty list. When X integration is
    supported, it should fetch recent tweets or mentions and draft replies
    using /generate_reply.
    """
    return []


@app.post("/x/post_replies")
async def x_post_replies(
    tone: str = "friendly",
    max_posts: int = 3,
    max_comments: int = 5,
    current_user: dict = Depends(get_current_user),
) -> dict:
    """Post prepared replies to X (placeholder).

    This stub endpoint does nothing and returns a summary message.
    A full implementation would post replies to X using the Twitter API.
    """
    return {"status": "X posting not implemented yet"}

# Reddit endpoints
@app.get("/reddit/login")
async def reddit_login() -> dict:
    """Placeholder login endpoint for Reddit OAuth.

    This stub simply returns a message indicating that the Reddit login flow
    hasn't been implemented. A real implementation would redirect the user
    to Reddit's OAuth dialog and handle the callback.
    """
    return {"message": "Reddit login is not implemented yet. Stay tuned!"}


@app.post("/reddit/prepare_replies")
async def reddit_prepare_replies(
    tone: str = "friendly",
    max_posts: int = 3,
    max_comments: int = 5,
    current_user: dict = Depends(get_current_user),
) -> list[dict]:
    """Prepare draft replies for Reddit comments (placeholder).

    This stub endpoint returns an empty list. When Reddit integration is
    supported, it should fetch recent comments on the user's Reddit posts
    and draft replies using /generate_reply.
    """
    return []


@app.post("/reddit/post_replies")
async def reddit_post_replies(
    tone: str = "friendly",
    max_posts: int = 3,
    max_comments: int = 5,
    current_user: dict = Depends(get_current_user),
) -> dict:
    """Post prepared replies to Reddit (placeholder).

    This stub endpoint does nothing and returns a summary message.
    A full implementation would post replies to Reddit via the Reddit API.
    """
    return {"status": "Reddit posting not implemented yet"}