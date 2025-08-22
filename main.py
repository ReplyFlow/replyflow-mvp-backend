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

# ----------------------------------------------------------------------------
# In-memory log of failed reply postings
#
# Failed replies (e.g. due to network or API errors) are recorded here so that
# users can view them in the dashboard. Each log entry is a dictionary with
# keys: user_id, platform, timestamp, and any additional details such as
# page_id, comment_id, tweet_id, etc. The log is global and does not persist
# across server restarts. In a production system you might store these
# entries in a database.
FAILED_REPLY_LOGS: list[dict] = []

def log_failed_reply(user_id: str, platform: str, **details: Any) -> None:
    """
    Append a failed reply entry to the global FAILED_REPLY_LOGS. Each entry
    includes the user_id, the platform (e.g. "facebook", "x", "reddit"), a
    timestamp in UTC, and any additional details about the failure (such
    as page_id, comment_id, tweet_id, and error message).

    Args:
        user_id: The ID of the current user for whom the reply failed.
        platform: The name of the platform where the failure occurred.
        **details: Additional details describing the failure. Keys may
            include page_id, comment_id, tweet_id, etc.
    """
    entry = {
        "user_id": user_id,
        "platform": platform,
        # Append a 'Z' to indicate UTC time in ISO8601 format
        "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
    }
    entry.update(details)
    FAILED_REPLY_LOGS.append(entry)

# ----------------------------------------------------------------------------
# Blacklist filtering configuration
#
# To comply with content guidelines, we implement a simple blacklist filter
# that screens comments and generated replies for profanity or legal issues.
# You can customize the blacklist via the REPLYFLOW_BLACKLIST environment
# variable. Provide a comma‑separated list of words or phrases to block.
# The default list below covers common profanities and certain legal terms.

def _load_blacklist() -> set[str]:
    """Load a set of blacklisted words from the environment or defaults.

    Reads the REPLYFLOW_BLACKLIST environment variable (comma‑separated) and
    merges it with a default list of terms. All terms are lower‑cased.

    Returns:
        A set of lower‑cased words to treat as blacklisted.
    """
    env_val = os.getenv("REPLYFLOW_BLACKLIST") or ""
    # Default blacklist covers common profanity and basic legal terms.
    default_words = {
        "fuck",
        "shit",
        "asshole",
        "bitch",
        "cunt",
        "bastard",
        "damn",
        "lawsuit",
        "sue",
        "lawyer",
        "illegal",
        "crime",
        "scam",
    }
    words: set[str] = set()
    for token in env_val.split(','):
        token = token.strip().lower()
        if token:
            words.add(token)
    words.update(default_words)
    return words


# Initialize blacklist at module import time
BLACKLIST: set[str] = _load_blacklist()

def contains_blacklisted(text: str) -> bool:
    """Check if the provided text contains any blacklisted word.

    Args:
        text: The text to inspect.

    Returns:
        True if any blacklisted term is present (case‑insensitive), False otherwise.
    """
    if not text:
        return False
    lower = text.lower()
    for word in BLACKLIST:
        if word and word in lower:
            return True
    return False

def get_safe_reply() -> str:
    """Return a generic safe reply when the content violates guidelines."""
    return "I'm sorry, but I'm unable to respond to that comment due to content guidelines."

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
    # Optional maximum number of tokens for the generated reply. If provided,
    # this value will be clamped between 1 and 120 to control costs. If not
    # provided, the default of 60 tokens is used. See generate_with_openai.
    max_tokens: Optional[int] = None


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
    "business_management",
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

# -----------------------------------------------------------------------------
# Facebook token refresh utilities
#
# Facebook long‑lived user access tokens generally last about 60 days, but they
# can be refreshed as long as they are at least 24 hours old and have not
# expired【528623933385330†L59-L63】. To reduce the chance that a user will need
# to explicitly re‑authenticate, we implement a helper that checks how long
# ago a user access token was stored. If it exceeds a configurable threshold
# (approximately 50 days), we call the Facebook OAuth endpoint again to
# exchange the current long‑lived token for a new one. This refresh happens
# quietly on the server side and updates all stored tokens for the user.

# Number of days after which a stored long‑lived user token should be
# refreshed automatically. By default, refresh tokens that are older than 50
# days. The official documentation notes long‑lived tokens can be refreshed
# as long as they are at least 24 hours old and not expired【528623933385330†L59-L63】.
FB_TOKEN_REFRESH_THRESHOLD_DAYS = int(os.getenv("FB_TOKEN_REFRESH_THRESHOLD_DAYS", "50"))

async def _refresh_facebook_user_token_if_needed(user_id: str) -> None:
    """Refresh a stored Facebook user access token if it is nearing expiry.

    Look up the user's stored Facebook tokens. If the `created_at` timestamp
    indicates the token was stored more than `FB_TOKEN_REFRESH_THRESHOLD_DAYS`
    ago, attempt to exchange the existing long‑lived user token for a new
    long‑lived token. Update the user token for all pages in the token store.

    Args:
        user_id: The application user ID whose tokens should be refreshed.
    """
    # Ensure httpx and required environment variables are available
    if not _httpx_available:
        return
    app_id = os.getenv("FACEBOOK_APP_ID")
    app_secret = os.getenv("FACEBOOK_APP_SECRET")
    if not (app_id and app_secret):
        # Cannot refresh without app credentials
        return
    # Fetch all tokens for this user and provider
    tokens = token_store.list_tokens(user_id, "facebook")
    if not tokens:
        return
    # Determine when the first token was created (all tokens share the same
    # user token, so using the earliest created_at is sufficient). If the
    # timestamp is not ISO8601 parseable, skip refresh silently.
    try:
        # tokens are stored with created_at in ISO format; parse to datetime
        created_at_str = tokens[0].get("created_at")
        if created_at_str:
            created_at_dt = datetime.datetime.fromisoformat(created_at_str.rstrip("Z"))
        else:
            created_at_dt = None
    except Exception:
        created_at_dt = None
    if created_at_dt:
        age = datetime.datetime.utcnow() - created_at_dt
        if age < datetime.timedelta(days=FB_TOKEN_REFRESH_THRESHOLD_DAYS):
            # Token is still fresh enough; no refresh needed
            return
    # Determine current user access token from the first token entry
    current_user_token = tokens[0].get("user_access_token")
    if not current_user_token:
        return
    # Perform the refresh: call the Graph API to exchange the long‑lived token
    # for a new one. According to Meta's documentation, long‑lived tokens can be
    # refreshed by calling the OAuth endpoint with the same grant_type and
    # access_token parameters【528623933385330†L59-L63】.
    refresh_params = {
        "grant_type": "fb_exchange_token",
        "client_id": app_id,
        "client_secret": app_secret,
        # Note: the fb_exchange_token parameter accepts both short‑lived and
        # long‑lived user access tokens. If the token is still valid and more
        # than 24 hours old, Facebook will return a new long‑lived token.
        "fb_exchange_token": current_user_token,
    }
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.get(
                "https://graph.facebook.com/v23.0/oauth/access_token",
                params=refresh_params,
                timeout=10,
            )
            data = resp.json()
            new_token = data.get("access_token")
            if not new_token:
                # If no access_token returned, don't update tokens
                return
            # If the new token is the same as the existing one, skip update
            if new_token == current_user_token:
                return
    except Exception:
        # On network or parsing errors, silently skip refresh
        return
    # Update all stored tokens for this user: keep existing page and IG fields
    # but replace the user_access_token and update created_at timestamp. Use
    # REPLACE semantics via save_token; this will overwrite existing entries.
    for t in tokens:
        page_id = t.get("page_id")
        page_token = t.get("page_access_token")
        ig_account_id = t.get("ig_account_id")
        if page_id and page_token:
            token_store.save_token(
                user_id=user_id,
                provider="facebook",
                page_id=str(page_id),
                page_access_token=page_token,
                ig_account_id=ig_account_id,
                user_access_token=new_token,
            )

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

async def _prepare_facebook_replies(
    user_id: str,
    tone: str = "friendly",
    max_posts: int = 3,
    max_comments: int = 5,
    page_ids: Optional[list[str]] = None,
    max_tokens: int = 60,
) -> list[dict]:
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
    # If a list of page IDs was provided, filter the tokens to include only those pages.
    if page_ids:
        # Normalize page_ids to strings for comparison
        desired_ids = set(str(pid) for pid in page_ids)
        tokens = [t for t in tokens if t.get("page_id") and str(t.get("page_id")) in desired_ids]
    async with httpx.AsyncClient() as client:
        for token_info in tokens:
            page_id = token_info.get("page_id")
            page_token = token_info.get("page_access_token")
            user_token = token_info.get("user_access_token")
            if not page_id or not page_token:
                continue

            # Always prefer the page token when querying page edges. While user
            # tokens can sometimes access page data, they often lack the
            # granular page permissions needed for posts/comments. Using the
            # page access token ensures the API has the correct scopes.
            request_token = page_token

            # Helper to fetch posts with pagination. Try /posts first, then /feed as fallback.
            posts: list[dict] = []
            # A list of endpoints to try. Each entry is a tuple of the URL and params dict.
            endpoints_to_try = [
                (f"https://graph.facebook.com/v23.0/{page_id}/posts", {"access_token": request_token, "limit": max_posts}),
                (f"https://graph.facebook.com/v23.0/{page_id}/feed", {"access_token": request_token, "limit": max_posts}),
            ]
            for url, params in endpoints_to_try:
                if posts:
                    break  # Already fetched posts from previous endpoint
                next_url: Optional[str] = None
                current_params = params.copy()
                while len(posts) < max_posts:
                    try:
                        if next_url:
                            resp = await client.get(next_url, timeout=10)
                        else:
                            resp = await client.get(url, params=current_params, timeout=10)
                        data = resp.json()
                        # If the API responded with an error, break and try next endpoint
                        if isinstance(data, dict) and data.get("error"):
                            break
                        posts.extend(data.get("data", []))
                        # Stop if enough posts collected
                        if len(posts) >= max_posts:
                            break
                        # Determine next page URL if available
                        next_url = data.get("paging", {}).get("next")
                        if not next_url:
                            break
                        # After first request, clear params to avoid duplication when using next_url
                        current_params = {}
                    except Exception:
                        # On error, break and try next endpoint
                        break
            # Limit posts to max_posts
            posts = posts[:max_posts]
            # For each post, fetch comments with pagination
            for post in posts:
                post_id = post.get("id")
                if not post_id:
                    continue
                comments: list[dict] = []
                next_comment_url: Optional[str] = None
                # Build initial comments URL and params
                comments_url = f"https://graph.facebook.com/v23.0/{post_id}/comments"
                comments_params = {
                    "access_token": request_token,
                    "limit": max_comments,
                    "filter": "stream",
                    "order": "chronological",
                    "fields": "id,message",
                }
                while len(comments) < max_comments:
                    try:
                        if next_comment_url:
                            c_resp = await client.get(next_comment_url, timeout=10)
                        else:
                            c_resp = await client.get(comments_url, params=comments_params, timeout=10)
                        c_data = c_resp.json()
                        if isinstance(c_data, dict) and c_data.get("error"):
                            # If the API responded with an error, break
                            break
                        comments.extend(c_data.get("data", []))
                        if len(comments) >= max_comments:
                            break
                        next_comment_url = c_data.get("paging", {}).get("next")
                        if not next_comment_url:
                            break
                        # Clear params after first request
                        comments_params = {}
                    except Exception:
                        break
                # Process comments up to the maximum
                for comment in comments[:max_comments]:
                    comment_id = comment.get("id")
                    comment_message = comment.get("message", "")
                    if not comment_id or not comment_message:
                        continue
                    # Skip or flag comments containing blacklisted terms
                    if contains_blacklisted(comment_message):
                        draft_reply = get_safe_reply()
                        prepared.append({
                            "page_id": page_id,
                            "post_id": post_id,
                            "comment_id": comment_id,
                            "comment": comment_message,
                            "reply": draft_reply,
                            "blacklisted": True,
                        })
                        continue
                    # Generate reply normally and apply blacklist filter to the output
                    try:
                        if _openai_available and os.getenv("OPENAI_API_KEY"):
                            draft_reply = await generate_with_openai(comment_message, tone, max_tokens=max_tokens)
                        else:
                            draft_reply = generate_fallback_reply(comment_message, tone)
                    except Exception:
                        draft_reply = generate_fallback_reply(comment_message, tone)
                    # If generated reply contains blacklisted words, replace with safe reply
                    if contains_blacklisted(draft_reply):
                        draft_reply = get_safe_reply()
                        prepared.append({
                            "page_id": page_id,
                            "post_id": post_id,
                            "comment_id": comment_id,
                            "comment": comment_message,
                            "reply": draft_reply,
                            "blacklisted": True,
                        })
                    else:
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
    max_tokens: int = 60,
    current_user: dict = Depends(get_current_user),
    request: Request = None,
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
    # Attempt to refresh the user's long‑lived token if it's old. This call
    # silently returns if a refresh isn't needed or fails.
    try:
        await _refresh_facebook_user_token_if_needed(user_id)
    except Exception:
        # Ignore any errors during token refresh to avoid blocking the main flow
        pass
    # Determine selected pages based on query parameters or JSON body. Parse body if POST.
    selected_pages: Optional[list[str]] = None
    body_data: Dict[str, Any] = {}
    if request and request.method == "POST":
        try:
            body_data = await request.json()
        except Exception:
            body_data = {}
        if isinstance(body_data, dict):
            # Override tone and limits if provided in body
            if body_data.get("tone"):
                tone = str(body_data.get("tone"))
            if body_data.get("max_posts"):
                try:
                    max_posts = int(body_data.get("max_posts"))
                except Exception:
                    pass
            if body_data.get("max_comments"):
                try:
                    max_comments = int(body_data.get("max_comments"))
                except Exception:
                    pass
            # Override max_tokens if provided and clamp within range
            if body_data.get("max_tokens"):
                try:
                    mt_val = int(body_data.get("max_tokens"))
                    if mt_val < 1:
                        mt_val = 1
                    elif mt_val > 120:
                        mt_val = 120
                    max_tokens = mt_val
                except Exception:
                    pass
            # Read page_ids and page_id from body
            body_page_ids = body_data.get("page_ids")
            body_page_id = body_data.get("page_id")
            if body_page_ids and isinstance(body_page_ids, list):
                selected_pages = [str(pid) for pid in body_page_ids]
            elif body_page_id:
                selected_pages = [str(body_page_id)]
    # If not found in body or not POST, check query parameters
    if selected_pages is None and request:
        # parse multiple page_ids from query parameters (could be repeated)
        query_ids = request.query_params.getlist("page_ids") if hasattr(request.query_params, 'getlist') else []
        if query_ids:
            selected_pages = [str(pid) for pid in query_ids]
        else:
            query_page_id = request.query_params.get("page_id") if request.query_params else None
            if query_page_id:
                selected_pages = [str(query_page_id)]
        # Parse max_tokens from query parameters if provided
        query_mt = None
        try:
            query_mt = request.query_params.get("max_tokens") if request.query_params else None
        except Exception:
            query_mt = None
        if query_mt:
            try:
                mt_val_q = int(query_mt)
                if mt_val_q < 1:
                    mt_val_q = 1
                elif mt_val_q > 120:
                    mt_val_q = 120
                max_tokens = mt_val_q
            except Exception:
                pass
    prepared = await _prepare_facebook_replies(
        user_id,
        tone,
        max_posts,
        max_comments,
        page_ids=selected_pages,
        max_tokens=max_tokens,
    )
    return prepared


@app.post("/facebook/post_replies")
async def facebook_post_replies(
    tone: str = "friendly",
    max_posts: int = 3,
    max_comments: int = 5,
    max_tokens: int = 60,
    current_user: dict = Depends(get_current_user),
    request: Request = None,
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
    # Attempt to refresh the user's long‑lived token before performing API calls
    try:
        await _refresh_facebook_user_token_if_needed(user_id)
    except Exception:
        pass
    # Determine selected pages based on body or query parameters
    selected_pages: Optional[list[str]] = None
    body_data: Dict[str, Any] = {}
    if request and request.method == "POST":
        try:
            body_data = await request.json()
        except Exception:
            body_data = {}
        if isinstance(body_data, dict):
            # Override tone and limits if provided in body
            if body_data.get("tone"):
                tone = str(body_data.get("tone"))
            if body_data.get("max_posts"):
                try:
                    max_posts = int(body_data.get("max_posts"))
                except Exception:
                    pass
            if body_data.get("max_comments"):
                try:
                    max_comments = int(body_data.get("max_comments"))
                except Exception:
                    pass
            body_page_ids = body_data.get("page_ids")
            body_page_id = body_data.get("page_id")
            if body_page_ids and isinstance(body_page_ids, list):
                selected_pages = [str(pid) for pid in body_page_ids]
            elif body_page_id:
                selected_pages = [str(body_page_id)]

            # max_tokens override from body
            if body_data.get("max_tokens"):
                try:
                    mt_val = int(body_data.get("max_tokens"))
                    if mt_val < 1:
                        mt_val = 1
                    elif mt_val > 120:
                        mt_val = 120
                    max_tokens = mt_val
                except Exception:
                    pass
    if selected_pages is None and request:
        # parse multiple page_ids from query parameters
        query_ids = request.query_params.getlist("page_ids") if hasattr(request.query_params, 'getlist') else []
        if query_ids:
            selected_pages = [str(pid) for pid in query_ids]
        else:
            query_page_id = request.query_params.get("page_id") if request.query_params else None
            if query_page_id:
                selected_pages = [str(query_page_id)]
    prepared = await _prepare_facebook_replies(
        user_id,
        tone,
        max_posts,
        max_comments,
        page_ids=selected_pages,
        max_tokens=max_tokens,
    )
    results: dict[str, int] = {}
    if not _httpx_available:
        raise HTTPException(status_code=500, detail="httpx is required for Facebook API interactions")
    async with httpx.AsyncClient() as client:
        for item in prepared:
            # Skip items flagged as blacklisted to avoid posting profane or
            # legally sensitive replies. These items contain a safe reply and
            # a 'blacklisted' flag, indicating the original comment or reply
            # violated our guidelines.
            if item.get("blacklisted"):
                continue
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
                    f"https://graph.facebook.com/v23.0/{comment_id}/comments",
                    params={"access_token": page_token},
                    json={"message": reply_message},
                    timeout=10,
                )
                results[page_id] = results.get(page_id, 0) + 1
            except Exception as e:
                # Log the failure so it can be viewed in the log viewer
                log_failed_reply(
                    user_id=user_id,
                    platform="facebook",
                    page_id=page_id,
                    comment_id=comment_id,
                    error=str(e),
                )
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


# New endpoint: list Facebook pages for the current user.
# Returns a list of page objects with id and name. Uses stored page tokens to
# query the Graph API for page names. This is useful for displaying a
# page selection dropdown in the frontend.
@app.get("/admin/facebook_pages")
async def list_facebook_pages(current_user: dict = Depends(get_current_user)) -> list[dict]:
    """
    Retrieve the list of connected Facebook pages for the current user, including names.

    The page tokens stored in the token store do not include page names, so this
    endpoint calls the Graph API for each page to fetch its name using either
    the user access token or page access token. If the API call fails, the
    page ID is returned with the ID as its name.

    Returns:
        A list of dicts with keys: id, name.
    """
    user_id = current_user.get("id")
    # Refresh the user's token if it's nearing expiry. Any errors are ignored.
    try:
        await _refresh_facebook_user_token_if_needed(user_id)
    except Exception:
        pass
    tokens = token_store.list_tokens(user_id, "facebook")
    pages: list[dict] = []
    # If httpx is not available, return basic id list
    if not _httpx_available:
        for t in tokens:
            page_id = t.get("page_id")
            if page_id:
                pages.append({"id": page_id, "name": str(page_id)})
        return pages
    # Use async httpx client to fetch page names
    async with httpx.AsyncClient() as client:
        for t in tokens:
            page_id = t.get("page_id")
            # Use the user token first; fallback to page token if user token missing
            user_token = t.get("user_access_token") or t.get("page_access_token")
            if not page_id or not user_token:
                continue
            try:
                resp = await client.get(
                    f"https://graph.facebook.com/v23.0/{page_id}",
                    params={"access_token": user_token, "fields": "id,name"},
                    timeout=10,
                )
                data = resp.json()
                page_name = data.get("name") or str(page_id)
                pages.append({"id": page_id, "name": page_name})
            except Exception:
                pages.append({"id": page_id, "name": str(page_id)})
    return pages

@app.get("/admin/failed_replies")
def list_failed_replies(current_user: dict = Depends(get_current_user)) -> list[dict]:
    """
    Retrieve a list of failed reply postings for the current user.

    When posting replies to Facebook, X, or Reddit fails due to an exception
    (e.g. network error, invalid token), the details are recorded in an
    in-memory log. This endpoint returns those entries for the current
    authenticated user. Each log entry includes the platform, timestamp,
    and identifying IDs (such as page_id, comment_id, tweet_id) along with
    the error message.

    Returns:
        A list of dictionaries containing failed reply metadata for the user.
    """
    user_id = current_user.get("id")
    return [entry for entry in FAILED_REPLY_LOGS if entry.get("user_id") == user_id]


@app.delete("/admin/failed_replies")
def clear_failed_replies(current_user: dict = Depends(get_current_user)) -> dict:
    """
    Delete all failed reply logs for the current user.

    This endpoint removes any entries from the global FAILED_REPLY_LOGS that
    belong to the requesting user. It can be used to clear the log after
    reviewing failures.

    Returns:
        A dictionary indicating success.
    """
    user_id = current_user.get("id")
    # Filter out entries for this user and assign the new list
    global FAILED_REPLY_LOGS
    FAILED_REPLY_LOGS = [entry for entry in FAILED_REPLY_LOGS if entry.get("user_id") != user_id]
    return {"cleared": True}

# ------------------------------------------------------------------------------
# Serve the beta sign‑up page
#
# To collect pre‑beta users, we serve a static HTML form at /beta‑signup. You
# should replace YOUR_FORM_ID in replyflow_signup_page.html with your actual
# Formspree form ID (see instructions below). This route simply reads the
# file from disk and returns it as HTML. If you are hosting the frontend
# separately (e.g. via a CDN or static hosting service), you may not need this
# endpoint.

@app.get("/beta-signup", response_class=HTMLResponse)
async def beta_signup_page() -> HTMLResponse:
    """
    Serve the beta sign‑up page HTML.

    Returns:
        HTMLResponse containing the contents of replyflow_signup_page.html.
    """
    try:
        with open("replyflow_signup_page.html", "r", encoding="utf-8") as fh:
            content = fh.read()
        return HTMLResponse(content=content)
    except Exception:
        return HTMLResponse(
            content="<h1>Beta sign-up page not found</h1>", status_code=404
        )


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
    auth_url = "https://www.facebook.com/v23.0/dialog/oauth"
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
            "https://graph.facebook.com/v23.0/oauth/access_token",
            params=token_params,
            timeout=10,
        )

        token_data = token_resp.json()
        access_token = token_data.get("access_token")
        if not access_token:
            raise HTTPException(status_code=400, detail=f"Failed to obtain access token: {token_data}")

        # Exchange the short‑lived user token for a long‑lived token. According
        # to Meta's documentation, you can call the OAuth endpoint with the
        # grant_type `fb_exchange_token` to obtain a long‑lived user access
        # token that is generally valid for about 60 days【248791135810213†L90-L106】. This call must be
        # performed server‑side because it requires the app secret【248791135810213†L115-L123】.
        try:
            ll_params = {
                "grant_type": "fb_exchange_token",
                "client_id": app_id,
                "client_secret": app_secret,
                "fb_exchange_token": access_token,
            }
            ll_resp = await client.get(
                "https://graph.facebook.com/v23.0/oauth/access_token",
                params=ll_params,
                timeout=10,
            )
            ll_data = ll_resp.json()
            long_access_token = ll_data.get("access_token")
            # Prefer the long‑lived token if returned
            if long_access_token:
                access_token = long_access_token
        except Exception:
            # If the exchange fails, continue using the short‑lived token.
            pass

        # Request the Pages managed by the user to obtain a page token. The
        # user must have the required permissions (e.g., pages_read_engagement).
        # We explicitly request the access_token and tasks fields because the default response
        # may omit them. Without the page token we cannot read comments or post replies.
        pages_resp = await client.get(
            "https://graph.facebook.com/v23.0/me/accounts",
            params={
                "access_token": access_token,
                # Include tasks so we can inspect the capabilities returned by the API
                "fields": "id,name,access_token,tasks",
            },
            timeout=10,
        )
        pages_data = pages_resp.json()

        # Fallback 1: If no pages are returned (common with Business Manager/New Page Experience),
        # use the /me/assigned_pages endpoint which returns pages you have task‑based access to.
        if (not pages_data or not pages_data.get("data")):
            try:
                assigned_resp = await client.get(
                    "https://graph.facebook.com/v23.0/me/assigned_pages",
                    params={
                        "access_token": access_token,
                        "fields": "id,name,access_token,tasks",
                    },
                    timeout=10,
                )
                assigned_data = assigned_resp.json()
                if assigned_data and assigned_data.get("data"):
                    pages_data = assigned_data
            except Exception:
                # ignore errors; leave pages_data as‑is
                pass

        # Fallback 2: If still no pages, allow an explicit page to be connected via env var.
        if (not pages_data or not pages_data.get("data")):
            fallback_page_id = os.getenv("FACEBOOK_PAGE_ID")
            if fallback_page_id:
                try:
                    page_resp = await client.get(
                        f"https://graph.facebook.com/v23.0/{fallback_page_id}",
                        params={
                            "access_token": access_token,
                            "fields": "id,name,access_token,tasks",
                        },
                        timeout=10,
                    )
                    page_data = page_resp.json()
                    # If a page access token is returned, wrap it in the same structure as /me/accounts
                    if page_data and page_data.get("access_token"):
                        pages_data = {"data": [
                            {
                                "id": page_data.get("id"),
                                "name": page_data.get("name"),
                                "access_token": page_data.get("access_token"),
                                "tasks": page_data.get("tasks", []),
                            }
                        ]}
                except Exception:
                    # ignore failures; pages_data will remain empty
                    pages_data = pages_data or {"data": []}

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


async def generate_with_openai(comment: str, tone: str, max_tokens: int = 60) -> str:
    """Generate a reply using the OpenAI API.

    Args:
        comment: The original comment text.
        tone: Desired reply tone.
        max_tokens: Maximum number of tokens to generate. This value is
            clamped between 1 and 120 to control cost. Defaults to 60 if
            unspecified or invalid.

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
    # Define more descriptive tone descriptions for certain tones. If a tone
    # has a specialized description, use it; otherwise fall back to simply
    # mentioning the tone. This allows the model to better understand
    # nuanced tones like apologetic, quirky or diplomatic.
    tone_descriptions = {
        "friendly": "a warm and friendly",
        "focused": "a focused and direct",
        "enthusiastic": "an enthusiastic and energetic",
        "professional": "a professional and polite",
        "casual": "a casual and relaxed",
        "apologetic": "an apologetic and empathetic, acknowledging the concern",
        "quirky": "a quirky and playful with a touch of humor",
        "diplomatic": "a diplomatic and balanced, neutral yet courteous",
    }
    tone_key = tone.strip().lower() if tone else "friendly"
    # Build description; default to using the raw tone if not in dict
    tone_desc = tone_descriptions.get(tone_key, f"a {tone_key}")
    system_prompt = (
        "You are a helpful assistant that drafts concise and engaging "
        f"public replies in {tone_desc} tone. Keep the reply under 130 characters "
        "and encourage further interaction when appropriate."
    )
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": comment},
    ]
    try:
        # Clamp the requested token limit within [1, 120]. This avoids very long
        # replies and keeps usage within the desired cost range. If
        # max_tokens is not an integer, fall back to 60.
        try:
            mt = int(max_tokens)
        except Exception:
            mt = 60
        if mt < 1:
            mt = 1
        elif mt > 120:
            mt = 120
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=messages,
            max_tokens=mt,
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
    # Determine max_tokens from request and clamp it within range.
    # If unspecified, default to 60.
    max_tokens = 60
    if req.max_tokens is not None:
        try:
            mt_val = int(req.max_tokens)
            if mt_val < 1:
                mt_val = 1
            elif mt_val > 120:
                mt_val = 120
            max_tokens = mt_val
        except Exception:
            # ignore errors and keep default
            pass
    # If the comment is empty, reject the request
    if not comment:
        raise HTTPException(status_code=400, detail="Comment cannot be empty")
    # Filter: if the comment contains blacklisted content, skip generation and
    # return a generic safe reply instead. This prevents the model from
    # processing profane or legally sensitive content.
    if contains_blacklisted(comment):
        safe = get_safe_reply()
        return GenerateReplyResponse(reply=safe)
    # Generate the reply using OpenAI or fallback. Catch exceptions and
    # fallback on failure.
    try:
        if _openai_available and os.getenv("OPENAI_API_KEY"):
            reply = await generate_with_openai(comment, tone, max_tokens=max_tokens)
        else:
            reply = generate_fallback_reply(comment, tone)
    except Exception:
        reply = generate_fallback_reply(comment, tone)
    # Second filter: if the generated reply contains blacklisted terms, use
    # the safe reply instead. This ensures the output adheres to guidelines.
    if contains_blacklisted(reply):
        reply = get_safe_reply()
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
# Instagram, X (Twitter) and Reddit integrations

#
# Instagram integration
#

async def _get_ig_account_id(client: httpx.AsyncClient, page_id: str, access_token: str) -> Optional[str]:
    """Retrieve the Instagram business or creator account ID associated with a Facebook Page.

    Given a Facebook Page ID and a Page access token, query the Graph API for the
    `instagram_business_account` field. If present, return the IG account ID. If the
    field is not present or an error occurs, return None.

    Args:
        client: An httpx.AsyncClient used to make the HTTP request.
        page_id: The Facebook Page ID.
        access_token: A valid page access token with `instagram_basic` or
            `instagram_manage_comments` permissions.

    Returns:
        The IG account ID as a string, or None if not found.
    """
    try:
        resp = await client.get(
            f"https://graph.facebook.com/v17.0/{page_id}",
            params={"access_token": access_token, "fields": "instagram_business_account"},
            timeout=10,
        )
        data = resp.json()
        ig_obj = data.get("instagram_business_account")
        if ig_obj and isinstance(ig_obj, dict):
            return str(ig_obj.get("id")) if ig_obj.get("id") else None
    except Exception:
        return None
    return None


async def _prepare_instagram_replies(
    user_id: str,
    tone: str = "friendly",
    max_posts: int = 3,
    max_comments: int = 5,
    page_ids: Optional[list[str]] = None,
    max_tokens: int = 60,
) -> list[dict]:
    """Fetch recent Instagram comments and generate draft replies.

    This helper uses the stored Facebook Page tokens (provider="facebook") to identify
    Instagram Business or Creator accounts connected to those Pages. It then fetches
    recent media and comments via the Instagram Graph API and calls the reply
    generation logic for each comment. The result is a list of dictionaries with
    context and draft reply information.

    Args:
        user_id: The current application user ID.
        tone: Desired tone for replies.
        max_posts: Maximum number of media posts to fetch per IG account.
        max_comments: Maximum number of comments per media.
        page_ids: Optional list of Facebook Page IDs to restrict which IG accounts to use.

    Returns:
        A list of dicts with keys: page_id, ig_account_id, media_id, comment_id,
        comment (original text) and reply (generated text).
    """
    if not _httpx_available:
        raise RuntimeError("httpx is required for Instagram API interactions")
    prepared: list[dict] = []
    # Normalize tone
    tone = tone.strip().lower() if tone else "friendly"
    # Retrieve all Facebook page tokens for this user. We'll use these tokens
    # because Instagram Graph API calls are authorized via the connected Page
    # access token. Filter by selected pages if provided.
    fb_tokens = token_store.list_tokens(user_id, "facebook")
    if page_ids:
        desired = set(str(pid) for pid in page_ids)
        fb_tokens = [t for t in fb_tokens if t.get("page_id") and str(t.get("page_id")) in desired]
    async with httpx.AsyncClient() as client:
        for token_info in fb_tokens:
            page_id = token_info.get("page_id")
            page_token = token_info.get("page_access_token")
            if not page_id or not page_token:
                continue
            # Determine the IG account ID. Use cached value if present; otherwise fetch.
            ig_account_id = token_info.get("ig_account_id")
            if not ig_account_id:
                ig_account_id = await _get_ig_account_id(client, page_id, page_token)
                # Persist the IG account ID if we found it
                if ig_account_id:
                    try:
                        token_store.save_token(
                            user_id=user_id,
                            provider="facebook",
                            page_id=str(page_id),
                            page_access_token=page_token,
                            ig_account_id=str(ig_account_id),
                            user_access_token=token_info.get("user_access_token"),
                        )
                    except Exception:
                        pass
            if not ig_account_id:
                continue  # no IG account attached to this page
            # Fetch recent media posts for this IG account
            media: list[dict] = []
            next_media_url: Optional[str] = None
            media_url = f"https://graph.facebook.com/v17.0/{ig_account_id}/media"
            media_params = {
                "access_token": page_token,
                "fields": "id,caption",
                "limit": max_posts,
            }
            while len(media) < max_posts:
                try:
                    if next_media_url:
                        m_resp = await client.get(next_media_url, timeout=10)
                    else:
                        m_resp = await client.get(media_url, params=media_params, timeout=10)
                    m_data = m_resp.json()
                    # Check for error
                    if isinstance(m_data, dict) and m_data.get("error"):
                        break
                    media.extend(m_data.get("data", []))
                    if len(media) >= max_posts:
                        break
                    next_media_url = m_data.get("paging", {}).get("next")
                    if not next_media_url:
                        break
                    # Clear params after first request
                    media_params = {}
                except Exception:
                    break
            media = media[:max_posts]
            # For each media, fetch comments
            for m in media:
                media_id = m.get("id")
                if not media_id:
                    continue
                comments: list[dict] = []
                next_comment_url: Optional[str] = None
                comments_url = f"https://graph.facebook.com/v17.0/{media_id}/comments"
                comments_params = {
                    "access_token": page_token,
                    "fields": "id,text",
                    "limit": max_comments,
                }
                while len(comments) < max_comments:
                    try:
                        if next_comment_url:
                            c_resp = await client.get(next_comment_url, timeout=10)
                        else:
                            c_resp = await client.get(comments_url, params=comments_params, timeout=10)
                        c_data = c_resp.json()
                        if isinstance(c_data, dict) and c_data.get("error"):
                            break
                        comments.extend(c_data.get("data", []))
                        if len(comments) >= max_comments:
                            break
                        next_comment_url = c_data.get("paging", {}).get("next")
                        if not next_comment_url:
                            break
                        comments_params = {}
                    except Exception:
                        break
                # Process each comment
                for c in comments[:max_comments]:
                    comment_id = c.get("id")
                    comment_text = c.get("text") or c.get("message") or ""
                    if not comment_id or not comment_text:
                        continue
                    # Skip or flag comments containing blacklisted terms
                    if contains_blacklisted(comment_text):
                        reply_text = get_safe_reply()
                        prepared.append({
                            "page_id": page_id,
                            "ig_account_id": ig_account_id,
                            "media_id": media_id,
                            "comment_id": comment_id,
                            "comment": comment_text,
                            "reply": reply_text,
                            "blacklisted": True,
                        })
                        continue
                    # Generate a reply using OpenAI or fallback
                    try:
                        if _openai_available and os.getenv("OPENAI_API_KEY"):
                            reply_text = await generate_with_openai(comment_text, tone, max_tokens=max_tokens)
                        else:
                            reply_text = generate_fallback_reply(comment_text, tone)
                    except Exception:
                        reply_text = generate_fallback_reply(comment_text, tone)
                    # If generated reply contains blacklisted words, replace with safe reply
                    if contains_blacklisted(reply_text):
                        reply_text = get_safe_reply()
                        prepared.append({
                            "page_id": page_id,
                            "ig_account_id": ig_account_id,
                            "media_id": media_id,
                            "comment_id": comment_id,
                            "comment": comment_text,
                            "reply": reply_text,
                            "blacklisted": True,
                        })
                    else:
                        prepared.append({
                            "page_id": page_id,
                            "ig_account_id": ig_account_id,
                            "media_id": media_id,
                            "comment_id": comment_id,
                            "comment": comment_text,
                            "reply": reply_text,
                        })
    return prepared


@app.post("/instagram/prepare_replies")
async def instagram_prepare_replies(
    tone: str = "friendly",
    max_posts: int = 3,
    max_comments: int = 5,
    max_tokens: int = 60,
    current_user: dict = Depends(get_current_user),
    request: Request = None,
) -> list[dict]:
    """Prepare draft replies for Instagram comments.

    This endpoint fetches recent comments from the user's connected Instagram
    Business or Creator accounts (via the Facebook Page tokens), generates
    replies using the same logic as /generate_reply, and returns them. It
    accepts optional JSON body or query parameters for tone, max_posts,
    max_comments, and page_id(s) to filter which IG accounts are used.
    """
    user_id = current_user.get("id")
    # Parse JSON body if present (for POST requests)
    selected_pages: Optional[list[str]] = None
    body_data: Dict[str, Any] = {}
    if request and request.method == "POST":
        try:
            body_data = await request.json()
        except Exception:
            body_data = {}
        if isinstance(body_data, dict):
            if body_data.get("tone"):
                tone = str(body_data.get("tone"))
            if body_data.get("max_posts"):
                try:
                    max_posts = int(body_data.get("max_posts"))
                except Exception:
                    pass
            if body_data.get("max_comments"):
                try:
                    max_comments = int(body_data.get("max_comments"))
                except Exception:
                    pass
            # max_tokens override from body
            if body_data.get("max_tokens"):
                try:
                    mt_val = int(body_data.get("max_tokens"))
                    if mt_val < 1:
                        mt_val = 1
                    elif mt_val > 120:
                        mt_val = 120
                    max_tokens = mt_val
                except Exception:
                    pass
            body_page_ids = body_data.get("page_ids")
            body_page_id = body_data.get("page_id")
            if body_page_ids and isinstance(body_page_ids, list):
                selected_pages = [str(pid) for pid in body_page_ids]
            elif body_page_id:
                selected_pages = [str(body_page_id)]
    # Fallback to query params
    if selected_pages is None and request:
        query_ids = request.query_params.getlist("page_ids") if hasattr(request.query_params, 'getlist') else []
        if query_ids:
            selected_pages = [str(pid) for pid in query_ids]
        else:
            query_page_id = request.query_params.get("page_id") if request.query_params else None
            if query_page_id:
                selected_pages = [str(query_page_id)]
        # Parse max_tokens from query params if provided
        q_mt = None
        try:
            q_mt = request.query_params.get("max_tokens") if request.query_params else None
        except Exception:
            q_mt = None
        if q_mt:
            try:
                mt_val_q = int(q_mt)
                if mt_val_q < 1:
                    mt_val_q = 1
                elif mt_val_q > 120:
                    mt_val_q = 120
                max_tokens = mt_val_q
            except Exception:
                pass
    prepared = await _prepare_instagram_replies(
        user_id=user_id,
        tone=tone,
        max_posts=max_posts,
        max_comments=max_comments,
        page_ids=selected_pages,
        max_tokens=max_tokens,
    )
    return prepared


@app.post("/instagram/post_replies")
async def instagram_post_replies(
    tone: str = "friendly",
    max_posts: int = 3,
    max_comments: int = 5,
    max_tokens: int = 60,
    current_user: dict = Depends(get_current_user),
    request: Request = None,
) -> dict:
    """Post generated Instagram replies to their respective comments.

    This endpoint performs the same comment fetching and reply generation as
    /instagram/prepare_replies, then iterates through the prepared list and
    posts each reply back to Instagram using the Graph API. A summary of
    how many replies were posted per Facebook Page is returned. Any errors
    encountered during posting are ignored so that other replies can still
    be posted.
    """
    user_id = current_user.get("id")
    # Parse request body for tone, limits, max_tokens and page selection
    selected_pages: Optional[list[str]] = None
    body_data: Dict[str, Any] = {}
    if request and request.method == "POST":
        try:
            body_data = await request.json()
        except Exception:
            body_data = {}
        if isinstance(body_data, dict):
            if body_data.get("tone"):
                tone = str(body_data.get("tone"))
            if body_data.get("max_posts"):
                try:
                    max_posts = int(body_data.get("max_posts"))
                except Exception:
                    pass
            if body_data.get("max_comments"):
                try:
                    max_comments = int(body_data.get("max_comments"))
                except Exception:
                    pass
            # max_tokens override from body
            if body_data.get("max_tokens"):
                try:
                    mt_val = int(body_data.get("max_tokens"))
                    if mt_val < 1:
                        mt_val = 1
                    elif mt_val > 120:
                        mt_val = 120
                    max_tokens = mt_val
                except Exception:
                    pass
            body_page_ids = body_data.get("page_ids")
            body_page_id = body_data.get("page_id")
            if body_page_ids and isinstance(body_page_ids, list):
                selected_pages = [str(pid) for pid in body_page_ids]
            elif body_page_id:
                selected_pages = [str(body_page_id)]
    # Fallback to query parameters for page ids and max_tokens
    if selected_pages is None and request:
        query_ids = request.query_params.getlist("page_ids") if hasattr(request.query_params, 'getlist') else []
        if query_ids:
            selected_pages = [str(pid) for pid in query_ids]
        else:
            query_page_id = request.query_params.get("page_id") if request.query_params else None
            if query_page_id:
                selected_pages = [str(query_page_id)]
        # Parse max_tokens from query params if provided
        q_mt = None
        try:
            q_mt = request.query_params.get("max_tokens") if request.query_params else None
        except Exception:
            q_mt = None
        if q_mt:
            try:
                mt_val_q = int(q_mt)
                if mt_val_q < 1:
                    mt_val_q = 1
                elif mt_val_q > 120:
                    mt_val_q = 120
                max_tokens = mt_val_q
            except Exception:
                pass
    # Prepare replies
    prepared = await _prepare_instagram_replies(
        user_id=user_id,
        tone=tone,
        max_posts=max_posts,
        max_comments=max_comments,
        page_ids=selected_pages,
        max_tokens=max_tokens,
    )
    results: dict[str, int] = {}
    if not _httpx_available:
        raise HTTPException(status_code=500, detail="httpx is required for Instagram API interactions")
    async with httpx.AsyncClient() as client:
        for item in prepared:
            # Skip blacklisted items to prevent posting safe replies for
            # comments/replies that contained profanity or legal issues.
            if item.get("blacklisted"):
                continue
            page_id = item["page_id"]
            comment_id = item["comment_id"]
            reply_message = item["reply"]
            # Find the page token for this page
            page_token = None
            for t in token_store.list_tokens(user_id, "facebook"):
                if t.get("page_id") == page_id:
                    page_token = t.get("page_access_token")
                    break
            if not page_token:
                continue
            # Post the reply: POST /{comment_id}/replies
            try:
                await client.post(
                    f"https://graph.facebook.com/v17.0/{comment_id}/replies",
                    params={"access_token": page_token},
                    json={"message": reply_message},
                    timeout=10,
                )
                results[page_id] = results.get(page_id, 0) + 1
            except Exception:
                continue
    return {"replies_posted": results}


#
# X (Twitter) integration
#

async def _prepare_x_replies(
    user_id: str,
    tone: str = "friendly",
    max_posts: int = 3,
    max_comments: int = 5,
    max_tokens: int = 60,
) -> list[dict]:
    """Fetch mentions from X (Twitter) and generate draft replies.

    This helper queries the Twitter API for recent mentions of the authenticated user.
    It requires either stored tokens for provider="x" in the token store or
    environment variables: `X_BEARER_TOKEN` for read-only operations and, if you
    intend to post replies, the OAuth 2.0 user access token with the `tweet.write`
    scope (e.g. via `X_ACCESS_TOKEN`). If neither is available, an empty list
    is returned. Errors during fetch are ignored.

    Args:
        user_id: The current application user ID.
        tone: The tone for generated replies.
        max_posts: Maximum number of mentions to fetch.
        max_comments: Unused for Twitter (replies are one-level).

    Returns:
        A list of dicts with keys: tweet_id, tweet_text, reply.
    """
    if not _httpx_available:
        raise RuntimeError("httpx is required for Twitter API interactions")
    prepared: list[dict] = []
    tone = tone.strip().lower() if tone else "friendly"
    # Determine bearer token: prefer stored user token from token_store for provider="x"
    bearer_token: Optional[str] = None
    # Look for stored tokens
    tokens = token_store.list_tokens(user_id, "x")
    if tokens:
        # Use the first stored user_access_token for API calls
        bearer_token = tokens[0].get("user_access_token") or tokens[0].get("page_access_token")
    # Fallback to environment variable
    if not bearer_token:
        bearer_token = os.getenv("X_BEARER_TOKEN") or os.getenv("TWITTER_BEARER_TOKEN")
    if not bearer_token:
        return []
    headers = {"Authorization": f"Bearer {bearer_token}"}
    try:
        async with httpx.AsyncClient() as client:
            # Retrieve authenticated user information to get their user ID
            me_resp = await client.get("https://api.twitter.com/2/users/me", headers=headers, timeout=10)
            me_data = me_resp.json()
            # Extract user id
            tw_user_id = None
            # new API returns data field
            if isinstance(me_data, dict):
                data_obj = me_data.get("data")
                if isinstance(data_obj, dict):
                    tw_user_id = data_obj.get("id")
            if not tw_user_id:
                return []
            # Fetch mentions: GET /2/users/:id/mentions
            mentions_url = f"https://api.twitter.com/2/users/{tw_user_id}/mentions"
            params = {"max_results": max_posts}
            mentions_resp = await client.get(mentions_url, headers=headers, params=params, timeout=10)
            mentions_data = mentions_resp.json()
            tweets = []
            if isinstance(mentions_data, dict):
                tweets = mentions_data.get("data", []) or []
            for tw in tweets[:max_posts]:
                tweet_id = tw.get("id")
                tweet_text = tw.get("text") or ""
                if not tweet_id or not tweet_text:
                    continue
                # Skip or flag tweets containing blacklisted terms
                if contains_blacklisted(tweet_text):
                    reply_text = get_safe_reply()
                    prepared.append({"tweet_id": tweet_id, "comment": tweet_text, "reply": reply_text, "blacklisted": True})
                    continue
                try:
                    if _openai_available and os.getenv("OPENAI_API_KEY"):
                        reply_text = await generate_with_openai(tweet_text, tone, max_tokens=max_tokens)
                    else:
                        reply_text = generate_fallback_reply(tweet_text, tone)
                except Exception:
                    reply_text = generate_fallback_reply(tweet_text, tone)
                # If generated reply contains blacklisted words, replace with safe reply
                if contains_blacklisted(reply_text):
                    reply_text = get_safe_reply()
                    prepared.append({"tweet_id": tweet_id, "comment": tweet_text, "reply": reply_text, "blacklisted": True})
                else:
                    prepared.append({"tweet_id": tweet_id, "comment": tweet_text, "reply": reply_text})
    except Exception:
        return []
    return prepared


@app.post("/x/prepare_replies")
async def x_prepare_replies(
    tone: str = "friendly",
    max_posts: int = 3,
    max_comments: int = 5,
    max_tokens: int = 60,
    current_user: dict = Depends(get_current_user),
    request: Request = None,
) -> list[dict]:
    """Prepare draft replies for X (Twitter) mentions.

    This endpoint queries the Twitter API for recent mentions of the authenticated
    user and generates replies. It accepts JSON body or query parameters for
    tone and max_posts. max_comments is ignored for Twitter but included
    for signature compatibility.
    """
    user_id = current_user.get("id")
    # Parse JSON body for tone, max_posts and max_tokens if provided
    if request and request.method == "POST":
        try:
            body = await request.json()
        except Exception:
            body = {}
        if isinstance(body, dict):
            if body.get("tone"):
                tone = str(body.get("tone"))
            if body.get("max_posts"):
                try:
                    max_posts = int(body.get("max_posts"))
                except Exception:
                    pass
            if body.get("max_tokens"):
                try:
                    mt = int(body.get("max_tokens"))
                    if mt < 1:
                        mt = 1
                    elif mt > 120:
                        mt = 120
                    max_tokens = mt
                except Exception:
                    pass
    # Fallback to query params for tone, max_posts and max_tokens
    if request:
        qp_tone = request.query_params.get("tone") if request.query_params else None
        if qp_tone:
            tone = str(qp_tone)
        qp_mp = request.query_params.get("max_posts") if request.query_params else None
        if qp_mp:
            try:
                max_posts = int(qp_mp)
            except Exception:
                pass
        qp_mt = None
        try:
            qp_mt = request.query_params.get("max_tokens") if request.query_params else None
        except Exception:
            qp_mt = None
        if qp_mt:
            try:
                mt_q = int(qp_mt)
                if mt_q < 1:
                    mt_q = 1
                elif mt_q > 120:
                    mt_q = 120
                max_tokens = mt_q
            except Exception:
                pass
    prepared = await _prepare_x_replies(
        user_id=user_id,
        tone=tone,
        max_posts=max_posts,
        max_comments=max_comments,
        max_tokens=max_tokens,
    )
    return prepared


@app.post("/x/post_replies")
async def x_post_replies(
    tone: str = "friendly",
    max_posts: int = 3,
    max_comments: int = 5,
    max_tokens: int = 60,
    current_user: dict = Depends(get_current_user),
    request: Request = None,
) -> dict:
    """Post generated replies to mentions on X (Twitter).

    After generating replies via _prepare_x_replies, this endpoint iterates over
    each generated reply and calls the Twitter API to publish it. It requires
    environment variables or stored tokens that allow posting (tweet.write scope).
    A summary of how many replies were posted is returned. If posting fails
    or tokens are unavailable, the corresponding replies are skipped.
    """
    user_id = current_user.get("id")
    # Parse body for tone, max_posts and max_tokens if provided
    if request and request.method == "POST":
        try:
            body = await request.json()
        except Exception:
            body = {}
        if isinstance(body, dict):
            if body.get("tone"):
                tone = str(body.get("tone"))
            if body.get("max_posts"):
                try:
                    max_posts = int(body.get("max_posts"))
                except Exception:
                    pass
            if body.get("max_tokens"):
                try:
                    mt = int(body.get("max_tokens"))
                    if mt < 1:
                        mt = 1
                    elif mt > 120:
                        mt = 120
                    max_tokens = mt
                except Exception:
                    pass
    # Fallback to query params for tone, max_posts and max_tokens
    if request:
        q_tone = request.query_params.get("tone") if request.query_params else None
        if q_tone:
            tone = str(q_tone)
        q_mp = request.query_params.get("max_posts") if request.query_params else None
        if q_mp:
            try:
                max_posts = int(q_mp)
            except Exception:
                pass
        q_mt = None
        try:
            q_mt = request.query_params.get("max_tokens") if request.query_params else None
        except Exception:
            q_mt = None
        if q_mt:
            try:
                mt_q = int(q_mt)
                if mt_q < 1:
                    mt_q = 1
                elif mt_q > 120:
                    mt_q = 120
                max_tokens = mt_q
            except Exception:
                pass
    prepared = await _prepare_x_replies(
        user_id=user_id,
        tone=tone,
        max_posts=max_posts,
        max_comments=max_comments,
        max_tokens=max_tokens,
    )
    # Determine posting credentials: look for stored token or env variables
    # For posting to Twitter, we need an OAuth 2.0 user access token with write
    # permissions. We'll look for X_ACCESS_TOKEN in token store or env.
    access_token: Optional[str] = None
    tokens = token_store.list_tokens(user_id, "x")
    if tokens:
        access_token = tokens[0].get("page_access_token") or tokens[0].get("user_access_token")
    if not access_token:
        access_token = os.getenv("X_ACCESS_TOKEN") or os.getenv("TWITTER_ACCESS_TOKEN")
    if not access_token:
        return {"replies_posted": 0}
    # Determine the user id for posting; we need to know our user id to set reply
    bearer_token = os.getenv("X_BEARER_TOKEN") or os.getenv("TWITTER_BEARER_TOKEN")
    tw_user_id = None
    if _httpx_available and bearer_token:
        try:
            async with httpx.AsyncClient() as client:
                me_resp = await client.get(
                    "https://api.twitter.com/2/users/me",
                    headers={"Authorization": f"Bearer {bearer_token}"},
                    timeout=10,
                )
                me_data = me_resp.json()
                data_obj = me_data.get("data") if isinstance(me_data, dict) else None
                if isinstance(data_obj, dict):
                    tw_user_id = data_obj.get("id")
        except Exception:
            tw_user_id = None
    # Post each reply; use OAuth2 user token (Bearer) for posting
    results = 0
    headers = {"Authorization": f"Bearer {access_token}", "Content-Type": "application/json"}
    async with httpx.AsyncClient() as client:
        for item in prepared:
            # Skip blacklisted items to avoid posting safe replies. These
            # represent mentions that contained prohibited content or generated
            # replies that were filtered by our blacklist.
            if item.get("blacklisted"):
                continue
            tweet_id = item.get("tweet_id")
            reply_text = item.get("reply")
            if not tweet_id or not reply_text:
                continue
            # Compose payload for the API. According to Twitter API v2, to reply
            # to a tweet you send a POST to /2/tweets with a JSON body containing
            # the `text` and a `reply` object with `in_reply_to_tweet_id`.
            payload = {
                "text": reply_text,
                "reply": {
                    "in_reply_to_tweet_id": tweet_id
                }
            }
            try:
                await client.post(
                    "https://api.twitter.com/2/tweets",
                    headers=headers,
                    json=payload,
                    timeout=10,
                )
                results += 1
            except Exception as e:
                # Log the failure to the failed replies log
                log_failed_reply(
                    user_id=user_id,
                    platform="x",
                    tweet_id=tweet_id,
                    error=str(e),
                )
                continue
    return {"replies_posted": results}


#
# Reddit integration
#

async def _get_reddit_access_token(client_id: str, client_secret: str, username: str, password: str, user_agent: str) -> Optional[str]:
    """Perform OAuth2 password grant to obtain a Reddit access token.

    This function performs a POST request to Reddit's /api/v1/access_token endpoint
    using the provided credentials. If successful, it returns the access token;
    otherwise it returns None. On any network or parsing error, None is returned.
    """
    try:
        # Reddit requires Basic Auth with client_id:client_secret and a user agent
        auth = (client_id, client_secret)
        headers = {"User-Agent": user_agent or "ReplyFlow/0.1"}
        data = {
            "grant_type": "password",
            "username": username,
            "password": password,
        }
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                "https://www.reddit.com/api/v1/access_token",
                data=data,
                auth=auth,
                headers=headers,
                timeout=10,
            )
            token_data = resp.json()
            return token_data.get("access_token") if isinstance(token_data, dict) else None
    except Exception:
        return None


async def _prepare_reddit_replies(
    user_id: str,
    tone: str = "friendly",
    max_posts: int = 3,
    max_comments: int = 5,
    max_tokens: int = 60,
) -> list[dict]:
    """Fetch recent Reddit comments on the user's posts and generate replies.

    This helper logs in to Reddit using credentials supplied via environment variables
    or stored tokens (provider="reddit"), retrieves the current user's recent
    comments or inbox (mentions) and generates replies using OpenAI or the fallback.
    It returns a list of dictionaries containing the comment fullname, original
    comment text and the generated reply.

    Args:
        user_id: The current application user ID.
        tone: Tone for generated replies.
        max_posts: Maximum number of posts or comment threads to fetch.
        max_comments: Maximum number of comments per thread (unused for simple inbox).

    Returns:
        A list of dicts with keys: comment_id (fullname), comment, reply.
    """
    if not _httpx_available:
        raise RuntimeError("httpx is required for Reddit API interactions")
    prepared: list[dict] = []
    tone = tone.strip().lower() if tone else "friendly"
    # Determine credentials: prefer stored token for provider="reddit"
    tokens = token_store.list_tokens(user_id, "reddit")
    reddit_token: Optional[str] = None
    reddit_user_agent: str = os.getenv("REDDIT_USER_AGENT", "ReplyFlow/0.1")
    if tokens:
        reddit_token = tokens[0].get("user_access_token") or tokens[0].get("page_access_token")
    # If no token, attempt to authenticate using env credentials
    if not reddit_token:
        client_id = os.getenv("REDDIT_CLIENT_ID")
        client_secret = os.getenv("REDDIT_CLIENT_SECRET")
        username = os.getenv("REDDIT_USERNAME")
        password = os.getenv("REDDIT_PASSWORD")
        if client_id and client_secret and username and password:
            reddit_token = await _get_reddit_access_token(client_id, client_secret, username, password, reddit_user_agent)
    if not reddit_token:
        return []
    headers = {
        "Authorization": f"bearer {reddit_token}",
        "User-Agent": reddit_user_agent,
    }
    try:
        async with httpx.AsyncClient() as client:
            # Fetch the user's recent comments (their own posts). We'll use the user's
            # comment stream from /api/v1/me to get the username, then fetch the
            # latest comments on posts authored by this user. Simplify by using the
            # inbox mentions endpoint (user's unread messages/comments). This call
            # returns recent mentions/comments requiring reply.
            me_resp = await client.get("https://oauth.reddit.com/api/v1/me", headers=headers, timeout=10)
            me_data = me_resp.json()
            username = None
            if isinstance(me_data, dict):
                username = me_data.get("name") or me_data.get("subreddit", {}).get("name")
            if not username:
                return []
            # Retrieve the user's unread mentions/messages which often contain comments
            inbox_resp = await client.get("https://oauth.reddit.com/message/unread", headers=headers, timeout=10)
            inbox_data = inbox_resp.json()
            # The unread listing returns a listing with children, each a message or comment
            children = []
            if isinstance(inbox_data, dict):
                data_obj = inbox_data.get("data")
                if isinstance(data_obj, dict):
                    children = data_obj.get("children", [])
            count = 0
            for child in children:
                if count >= max_posts:
                    break
                data = child.get("data") or {}
                # Only process comments (kind == t1) where the subject is a comment
                kind = child.get("kind") or data.get("type")
                if kind != "t1" and kind != "comment":
                    continue
                comment_id = data.get("name") or data.get("id")
                comment_body = data.get("body") or data.get("body_html") or ""
                if not comment_id or not comment_body:
                    continue
                # Skip or flag comments containing blacklisted terms
                if contains_blacklisted(comment_body):
                    reply_text = get_safe_reply()
                    prepared.append({"comment_id": comment_id, "comment": comment_body, "reply": reply_text, "blacklisted": True})
                    count += 1
                    continue
                try:
                    if _openai_available and os.getenv("OPENAI_API_KEY"):
                        reply_text = await generate_with_openai(comment_body, tone, max_tokens=max_tokens)
                    else:
                        reply_text = generate_fallback_reply(comment_body, tone)
                except Exception:
                    reply_text = generate_fallback_reply(comment_body, tone)
                # If generated reply contains blacklisted words, replace with safe reply
                if contains_blacklisted(reply_text):
                    reply_text = get_safe_reply()
                    prepared.append({"comment_id": comment_id, "comment": comment_body, "reply": reply_text, "blacklisted": True})
                else:
                    prepared.append({"comment_id": comment_id, "comment": comment_body, "reply": reply_text})
                count += 1
    except Exception:
        return []
    return prepared


@app.post("/reddit/prepare_replies")
async def reddit_prepare_replies(
    tone: str = "friendly",
    max_posts: int = 3,
    max_comments: int = 5,
    max_tokens: int = 60,
    current_user: dict = Depends(get_current_user),
    request: Request = None,
) -> list[dict]:
    """Prepare draft replies for Reddit comments.

    This endpoint logs in to Reddit using stored tokens or environment credentials,
    fetches recent mentions/comments and generates replies. It accepts JSON body
    or query parameters for tone and max_posts. The max_comments parameter is
    included for interface consistency but unused.
    """
    user_id = current_user.get("id")
    # Parse body for tone, max_posts and max_tokens
    if request and request.method == "POST":
        try:
            body = await request.json()
        except Exception:
            body = {}
        if isinstance(body, dict):
            if body.get("tone"):
                tone = str(body.get("tone"))
            if body.get("max_posts"):
                try:
                    max_posts = int(body.get("max_posts"))
                except Exception:
                    pass
            if body.get("max_tokens"):
                try:
                    mt = int(body.get("max_tokens"))
                    if mt < 1:
                        mt = 1
                    elif mt > 120:
                        mt = 120
                    max_tokens = mt
                except Exception:
                    pass
    # Query params override (tone, max_posts, max_tokens)
    if request:
        q_tone = request.query_params.get("tone") if request.query_params else None
        if q_tone:
            tone = str(q_tone)
        q_mp = request.query_params.get("max_posts") if request.query_params else None
        if q_mp:
            try:
                max_posts = int(q_mp)
            except Exception:
                pass
        q_mt = None
        try:
            q_mt = request.query_params.get("max_tokens") if request.query_params else None
        except Exception:
            q_mt = None
        if q_mt:
            try:
                mt_q = int(q_mt)
                if mt_q < 1:
                    mt_q = 1
                elif mt_q > 120:
                    mt_q = 120
                max_tokens = mt_q
            except Exception:
                pass
    prepared = await _prepare_reddit_replies(
        user_id=user_id,
        tone=tone,
        max_posts=max_posts,
        max_comments=max_comments,
        max_tokens=max_tokens,
    )
    return prepared


@app.post("/reddit/post_replies")
async def reddit_post_replies(
    tone: str = "friendly",
    max_posts: int = 3,
    max_comments: int = 5,
    max_tokens: int = 60,
    current_user: dict = Depends(get_current_user),
    request: Request = None,
) -> dict:
    """Post generated replies to Reddit comments.

    This endpoint generates draft replies as in /reddit/prepare_replies and then
    posts each reply back to Reddit using the /api/comment endpoint【761405074861166†L1393-L1405】. Only unread
    mentions are processed. A summary count of replies posted is returned. If
    posting fails or credentials are missing, the reply is skipped.
    """
    user_id = current_user.get("id")
    # Parse body for tone, max_posts and max_tokens
    if request and request.method == "POST":
        try:
            body = await request.json()
        except Exception:
            body = {}
        if isinstance(body, dict):
            if body.get("tone"):
                tone = str(body.get("tone"))
            if body.get("max_posts"):
                try:
                    max_posts = int(body.get("max_posts"))
                except Exception:
                    pass
            if body.get("max_tokens"):
                try:
                    mt = int(body.get("max_tokens"))
                    if mt < 1:
                        mt = 1
                    elif mt > 120:
                        mt = 120
                    max_tokens = mt
                except Exception:
                    pass
    # Query params override (tone, max_posts, max_tokens)
    if request:
        q_tone = request.query_params.get("tone") if request.query_params else None
        if q_tone:
            tone = str(q_tone)
        q_mp = request.query_params.get("max_posts") if request.query_params else None
        if q_mp:
            try:
                max_posts = int(q_mp)
            except Exception:
                pass
        q_mt = None
        try:
            q_mt = request.query_params.get("max_tokens") if request.query_params else None
        except Exception:
            q_mt = None
        if q_mt:
            try:
                mt_q = int(q_mt)
                if mt_q < 1:
                    mt_q = 1
                elif mt_q > 120:
                    mt_q = 120
                max_tokens = mt_q
            except Exception:
                pass
    prepared = await _prepare_reddit_replies(
        user_id=user_id,
        tone=tone,
        max_posts=max_posts,
        max_comments=max_comments,
        max_tokens=max_tokens,
    )
    # Determine access token for posting
    # Prefer stored token from token_store; fallback to env credentials via OAuth
    reddit_token: Optional[str] = None
    tokens = token_store.list_tokens(user_id, "reddit")
    if tokens:
        reddit_token = tokens[0].get("user_access_token") or tokens[0].get("page_access_token")
    reddit_user_agent: str = os.getenv("REDDIT_USER_AGENT", "ReplyFlow/0.1")
    # If no token, attempt to get one
    if not reddit_token:
        client_id = os.getenv("REDDIT_CLIENT_ID")
        client_secret = os.getenv("REDDIT_CLIENT_SECRET")
        username = os.getenv("REDDIT_USERNAME")
        password = os.getenv("REDDIT_PASSWORD")
        if client_id and client_secret and username and password:
            reddit_token = await _get_reddit_access_token(client_id, client_secret, username, password, reddit_user_agent)
    if not reddit_token:
        return {"replies_posted": 0}
    headers = {
        "Authorization": f"bearer {reddit_token}",
        "User-Agent": reddit_user_agent,
        "Content-Type": "application/x-www-form-urlencoded",
    }
    results = 0
    async with httpx.AsyncClient() as client:
        for item in prepared:
            # Skip blacklisted items to avoid posting replies that were flagged
            # due to prohibited content. These items contain a safe reply and
            # should not be published.
            if item.get("blacklisted"):
                continue
            comment_id = item.get("comment_id")
            reply_text = item.get("reply")
            if not comment_id or not reply_text:
                continue
            # The Reddit API expects the fullname of the parent comment (e.g. t1_xxx)
            # and the reply text as raw markdown【761405074861166†L1393-L1405】. We also include api_type=json to
            # request JSON response.
            data = {
                "thing_id": comment_id,
                "text": reply_text,
                "api_type": "json",
            }
            try:
                await client.post(
                    "https://oauth.reddit.com/api/comment",
                    data=data,
                    headers=headers,
                    timeout=10,
                )
                results += 1
            except Exception as e:
                # Log the failure for later viewing
                log_failed_reply(
                    user_id=user_id,
                    platform="reddit",
                    comment_id=comment_id,
                    error=str(e),
                )
                continue
    return {"replies_posted": results}