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
import uuid
import datetime
import time
import base64
import hmac
import hashlib
import secrets
import logging
import httpx
import urllib.parse
from typing import Optional, Dict, Any
from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.responses import RedirectResponse, HTMLResponse
from pydantic import BaseModel
from urllib.parse import urlencode, quote_plus
from token_store import TokenStore

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
    "pages_manage_engagement",
    "pages_read_engagement",
    "pages_manage_posts",
    "pages_show_list"
]


# Initialize a global token store. You can override the storage path by setting
token_store = TokenStore()

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

async def _prepare_facebook_replies(tone: str = "friendly", max_posts: int = 3, max_comments: int = 5) -> list[dict]:
    """Fetch recent comments from stored Facebook pages and generate draft replies.

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
    tokens = token_store.list_tokens()
    async with httpx.AsyncClient() as client:
        for token_info in tokens:
            page_id = token_info.get("page_id")
            page_token = token_info.get("page_access_token")
            if not page_id or not page_token:
                continue
            # Fetch recent posts for the page
            try:
                feed_resp = await client.get(
                    f"https://graph.facebook.com/v18.0/{page_id}/feed",
                    params={"access_token": page_token, "limit": max_posts},
                    timeout=10,
                )
                feed_data = feed_resp.json()
            except Exception:
                continue
            for post in feed_data.get("data", [])[:max_posts]:
                post_id = post.get("id")
                if not post_id:
                    continue
                # Fetch comments on the post
                try:
                    comments_resp = await client.get(
                        f"https://graph.facebook.com/v18.0/{post_id}/comments",
                        params={"access_token": page_token, "limit": max_comments},
                        timeout=10,
                    )
                    comments_data = comments_resp.json()
                except Exception:
                    continue
                for comment in comments_data.get("data", [])[:max_comments]:
                    comment_id = comment.get("id")
                    comment_message = comment.get("message", "")
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


@app.get("/admin/facebook_tokens")
def list_facebook_tokens(current_user: dict = Depends(get_current_user)) -> list[dict]:
    """Return all stored Facebook page and user access tokens.

    This endpoint is for demonstration and debugging purposes only. In a
    production environment you should secure this route (e.g. with
    authentication) and avoid returning sensitive tokens directly.
    """
    return token_store.list_tokens()


@app.get("/facebook/login")
async def facebook_login(request: Request):
    client_id = os.getenv("FACEBOOK_APP_ID")
    redirect_uri = os.getenv("FACEBOOK_REDIRECT_URI")
    scope = ",".join(FACEBOOK_SCOPES)

    # Grab session token from cookie or header
    session_token = None
    auth_header = request.headers.get("Authorization")
    if auth_header and auth_header.startswith("Bearer "):
        session_token = auth_header.split(" ", 1)[1].strip()
    if not session_token:
        session_token = request.cookies.get("session_token")

    # If the user is not logged in, redirect them to /login
    if not session_token:
        return RedirectResponse(url="/login")

    state = session_token  # Use session token as state
    auth_url = (
        f"https://www.facebook.com/v18.0/dialog/oauth?"
        f"client_id={client_id}&"
        f"redirect_uri={redirect_uri}&"
        f"scope={scope}&"
        f"response_type=code&"
        f"state={state}"
    )
    return RedirectResponse(url=auth_url)


@app.get("/facebook/callback")
async def facebook_callback(request: Request, code: str, state: Optional[str] = None):
    """Handle the OAuth callback from Facebook, exchange the code, and redirect to dashboard."""

    FRONTEND_BASE_URL = os.getenv("FRONTEND_BASE_URL", "https://replyflowapp.com")
    DASHBOARD_URL = os.getenv("DASHBOARD_URL", f"{FRONTEND_BASE_URL}/dashboard.html")

    def _redir(reason: Optional[str] = None):
        if reason:
            return RedirectResponse(
                url=f"{DASHBOARD_URL}?connected=facebook&error=1&reason={quote_plus(reason)}",
                status_code=302,
            )
        return RedirectResponse(
            url=f"{DASHBOARD_URL}?connected=facebook",
            status_code=302,
        )

    app_id = os.getenv("FACEBOOK_APP_ID")
    app_secret = os.getenv("FACEBOOK_APP_SECRET")
    redirect_uri = os.getenv("FACEBOOK_REDIRECT_URI")

    if not (app_id and app_secret and redirect_uri):
        logging.error("Missing Facebook config environment variables")
        return _redir("missing_config")

    if not _httpx_available:
        logging.error("httpx is not available")
        return _redir("httpx_unavailable")

    try:
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
            logging.error(f"Failed to obtain access token: {token_data}")
            return _redir("no_access_token")

        # Determine current user ID from Authorization header or session cookie
        session_token: Optional[str] = None
        auth_header = request.headers.get("Authorization")
        if auth_header and auth_header.startswith("Bearer "):
            session_token = auth_header.split(" ", 1)[1].strip()
        if not session_token:
            # session_token = request.cookies.get("session_token")
            session_token = state
            current_user_id = session_store.get_user_id(session_token) if session_token else None

        # If user not found, stop
        if not current_user_id:
            return _redir("no_user")

        # Fetch Pages managed by the user
        pages_resp = await client.get(
            "https://graph.facebook.com/v18.0/me/accounts",
            params={"access_token": access_token},
            timeout=10,
        )
        pages_data = pages_resp.json()

            # Save access tokens per page
        if current_user_id:
          for page in pages_data.get("data", []):
              page_id = page.get("id")
              page_access_token = page.get("access_token")
              if page_id and page_access_token:
                  token_store.save_token(
                      user_id=current_user_id,
                      provider="facebook",
                      page_id=page_id,
                      page_access_token=page_access_token,
                      user_access_token=access_token
                  )

        return _redir(None)

  except Exception as e:
      logging.exception("Unexpected error in Facebook callback")
      return _redir("unexpected_error")

@app.post("/facebook/prepare_replies")
async def facebook_prepare_replies(
    tone: str = "friendly",
    max_posts: int = 3,
    max_comments: int = 5,
    current_user: dict = Depends(get_current_user),
):
    page_token = token_store.get_page_token(current_user["id"], "facebook")
    if not page_token:
        return {
            "status": "no_page_token",
            "message": "Please connect a Facebook Page first.",
            "replies": []
        }

    try:
        prepared = await _prepare_facebook_replies(tone, max_posts, max_comments)
    except Exception as e:
        logging.exception("Failed to prepare replies")
        return {
            "status": "error",
            "message": "Something went wrong while preparing replies.",
            "replies": []
        }

    if not prepared:
        return {
            "status": "no_comments",
            "message": "No comments found.",
            "replies": []
        }

    return {
        "status": "ok",
        "message": "Replies prepared successfully.",
        "replies": prepared
    }


@app.post("/facebook/post_replies")
async def facebook_post_replies(
    tone: str = "friendly",
    max_posts: int = 3,
    max_comments: int = 5,
    current_user: dict = Depends(get_current_user),
) -> dict:
    if not _httpx_available:
        raise HTTPException(status_code=500, detail="httpx is required for Facebook API interactions")

    try:
        prepared = await _prepare_facebook_replies(tone, max_posts, max_comments)
    except Exception as e:
        logging.exception("Failed to prepare replies for posting")
        return {
            "status": "error",
            "message": "Something went wrong while preparing replies.",
            "replies_posted": {}
        }

    if not prepared:
        return {
            "status": "no_comments",
            "message": "No comments found to reply to.",
            "replies_posted": {}
        }

    results: dict[str, int] = {}

    async with httpx.AsyncClient() as client:
        for item in prepared:
            page_id = item["page_id"]
            comment_id = item["comment_id"]
            reply_message = item["reply"]

            page_token = token_store.get_page_token(current_user["id"], "facebook")
            if not page_token:
                continue

            try:
                await client.post(
                    f"https://graph.facebook.com/v18.0/{comment_id}/comments",
                    params={"access_token": page_token},
                    json={"message": reply_message},
                    timeout=10,
                )
                results[page_id] = results.get(page_id, 0) + 1
            except Exception as e:
                logging.warning(f"Failed to post reply to comment {comment_id} on page {page_id}: {e}")
                continue

    return {
        "status": "ok",
        "message": "Replies posted successfully.",
        "replies_posted": results
    }


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

# --- config (keep your existing FRONTEND_BASE_URL / DASHBOARD_URL) ---
FRONTEND_BASE_URL = os.getenv("FRONTEND_BASE_URL", "https://replyflowapp.com")
DASHBOARD_URL = os.getenv("DASHBOARD_URL", f"{FRONTEND_BASE_URL}/dashboard.html")

INSTAGRAM_APP_ID = os.getenv("INSTAGRAM_APP_ID")        # same Meta app id
INSTAGRAM_APP_SECRET = os.getenv("INSTAGRAM_APP_SECRET")
INSTAGRAM_REDIRECT_URI = os.getenv("INSTAGRAM_REDIRECT_URI")

IG_SCOPES = [
    "instagram_basic",
    "instagram_manage_comments",
    "pages_show_list",
    "pages_read_engagement",
    "pages_manage_engagement",
]

logger = logging.getLogger("replyflow.instagram")

def _redir_instagram(reason: Optional[str] = None) -> RedirectResponse:
    if reason:
        return RedirectResponse(
            url=f"{DASHBOARD_URL}?connected=instagram&error=1&reason={quote_plus(reason)}",
            status_code=302,
        )
    return RedirectResponse(
        url=f"{DASHBOARD_URL}?connected=instagram",
        status_code=302,
    )

# --- lightweight state generator/validator (no server session needed) ---
def _make_state(secret: str, ttl_sec: int = 600) -> str:
    ts = str(int(time.time()))
    sig = hmac.new(secret.encode(), ts.encode(), hashlib.sha256).digest()
    return f"{ts}.{base64.urlsafe_b64encode(sig).decode().rstrip('=')}"

def _check_state(state: str, secret: str, ttl_sec: int = 600) -> bool:
    try:
        ts_str, sig_b64 = state.split(".", 1)
        ts = int(ts_str)
        if abs(time.time() - ts) > ttl_sec:
            return False
        expected = hmac.new(secret.encode(), ts_str.encode(), hashlib.sha256).digest()
        ok = hmac.compare_digest(
            base64.urlsafe_b64encode(expected).decode().rstrip("="),
            sig_b64
        )
        return ok
    except Exception:
        return False

# --- /instagram/login: build Meta OAuth URL with Instagram scopes ---
@app.get("/instagram/login")
async def instagram_login():
    if not (INSTAGRAM_APP_ID and INSTAGRAM_APP_SECRET and INSTAGRAM_REDIRECT_URI):
        logger.error("Missing Instagram config envs (APP_ID/SECRET/REDIRECT_URI)")
        return _redir_instagram("missing_config")

    state = _make_state(INSTAGRAM_APP_SECRET)
    params = {
        "client_id": INSTAGRAM_APP_ID,          # your Meta App ID
        "redirect_uri": INSTAGRAM_REDIRECT_URI, # this endpoint's callback
        "state": state,
        "response_type": "code",
        "scope": " ".join(IG_SCOPES),
    }
    auth_url = f"https://www.facebook.com/v18.0/dialog/oauth?{urlencode(params)}"
    # 302 to the Meta Login dialog
    return RedirectResponse(url=auth_url, status_code=302)

# --- /instagram/callback: exchange code, find IG Business Account, redirect ---
@app.get("/instagram/callback")
async def instagram_callback(request: Request, code: Optional[str] = None, state: Optional[str] = None):
    if not (INSTAGRAM_APP_ID and INSTAGRAM_APP_SECRET and INSTAGRAM_REDIRECT_URI):
        logger.error("Missing Instagram config envs during callback")
        return _redir_instagram("missing_config")

    if not code:
        logger.warning("Missing OAuth code in Instagram callback")
        return _redir_instagram("missing_code")

    if not state or not _check_state(state, INSTAGRAM_APP_SECRET):
        logger.warning("Invalid state in Instagram callback")
        return _redir_instagram("invalid_state")

    try:
        async with httpx.AsyncClient(timeout=15) as client:
            # 1) Exchange code -> short-lived user access token
            token_params = {
                "client_id": INSTAGRAM_APP_ID,
                "client_secret": INSTAGRAM_APP_SECRET,
                "redirect_uri": INSTAGRAM_REDIRECT_URI,
                "code": code,
            }
            token_resp = await client.get(
                "https://graph.facebook.com/v18.0/oauth/access_token",
                params=token_params,
            )
            token_data = token_resp.json()
            user_access_token = token_data.get("access_token")
            if not user_access_token:
                logger.error(f"IG token exchange failed: {token_data}")
                return _redir_instagram("no_access_token")

            # 2) (Optional but recommended) Exchange to a long-lived token
            #    Note: for FB user tokens: /oauth/access_token with fb_exchange_token
            ll_params = {
                "grant_type": "fb_exchange_token",
                "client_id": INSTAGRAM_APP_ID,
                "client_secret": INSTAGRAM_APP_SECRET,
                "fb_exchange_token": user_access_token,
            }
            ll_resp = await client.get(
                "https://graph.facebook.com/v18.0/oauth/access_token",
                params=ll_params,
            )
            ll_data = ll_resp.json()
            long_lived_user_token = ll_data.get("access_token", user_access_token)

            # 3) Get Pages the user manages (need a Page that connects to IG)
            pages_resp = await client.get(
                "https://graph.facebook.com/v18.0/me/accounts",
                params={"access_token": long_lived_user_token, "limit": 50},
            )
            pages_data = pages_resp.json()
            pages = pages_data.get("data", []) if isinstance(pages_data, dict) else []

            # 4) For each Page, check if it has an Instagram Business Account
            ig_account_id = None
            page_access_token = None
            page_id = None

            for p in pages:
                pid = p.get("id")
                ptoken = p.get("access_token")
                if not (pid and ptoken):
                    continue
                page_fields_resp = await client.get(
                    f"https://graph.facebook.com/v18.0/{pid}",
                    params={"fields": "instagram_business_account", "access_token": ptoken},
                )
                page_fields = page_fields_resp.json()
                iba = page_fields.get("instagram_business_account", {}) if isinstance(page_fields, dict) else {}
                if "id" in iba:
                    ig_account_id = iba["id"]
                    page_access_token = ptoken
                    page_id = pid
                    break

            # 5) Persist what you need so your app can act (example)
                                                                                                                                                                                      
            # After exchanging code for a token and finding IG business account and page token
            token_store.save_token(
                user_id=current_user_id,
                provider="instagram",
                page_id=page["id"],
                page_access_token=page["access_token"],
                ig_account_id=ig_user_id,
                user_access_token=long_lived_user_token
            )


            # 6) Success redirect (include which IG account you connected if available)
            success_qs = "connected=instagram"
            if ig_account_id:
                success_qs += f"&ig_id={quote_plus(ig_account_id)}"
            if page_id:
                success_qs += f"&page_id={quote_plus(page_id)}"
            return RedirectResponse(url=f"{DASHBOARD_URL}?{success_qs}", status_code=302)

    except Exception as e:
        logger.exception("Unexpected error during Instagram callback: %s", e)
        return _redir_instagram("unexpected_error")


# ---- Instagram helpers ------------------------------------------------------
async def _prepare_instagram_replies(
    tone: str,
    max_posts: int,
    max_comments: int,
    current_user: dict,
) -> list[dict]:
    """
    Fetch recent Instagram media for the connected Business/Creator account
    and build draft replies for recent comments.
    Returns a list of dicts: [{media_id, comment_id, original_comment, reply, ig_user_id}]
    """
    # You should already save these in your OAuth callback:
    # token_store.save_token(user_id, provider="instagram",
    # ig_account_id=<IG user id>, page_access_token=<page token>)
    ig_token = token_store.get_token_for_user(current_user.get("id"), provider="instagram")
    if not ig_token:
        return []

    ig_user_id = ig_token.get("ig_account_id")
    page_access_token = ig_token.get("page_access_token")
    if not (ig_user_id and page_access_token):
        return []

    prepared: list[dict] = []
    if not _httpx_available:
        return prepared

    async with httpx.AsyncClient(timeout=15) as client:
        # 1) Recent media
        media_resp = await client.get(
            f"https://graph.facebook.com/v18.0/{ig_user_id}/media",
            params={
                "fields": "id,caption",
                "limit": max_posts,
                "access_token": page_access_token,
            },
        )
        media_items = (media_resp.json() or {}).get("data", [])

        # 2) For each media, fetch comments and draft replies
        for m in media_items:
            media_id = m.get("id")
            if not media_id:
                continue

            comments_resp = await client.get(
                f"https://graph.facebook.com/v18.0/{media_id}/comments",
                params={
                    "fields": "id,text,username,hidden",
                    "limit": max_comments,
                    "access_token": page_access_token,
                },
            )
            comments = (comments_resp.json() or {}).get("data", []) or []

            for c in comments[:max_comments]:
                cid = c.get("id")
                text = c.get("text")
                if not (cid and text):
                    continue

                prompt = f"Reply on behalf of the Instagram account. Tone: {tone}. Comment: {text}"
                draft_reply = await generate_reply(prompt)  # your existing generator

                prepared.append({
                    "media_id": media_id,
                    "comment_id": cid,
                    "original_comment": text,
                    "reply": draft_reply,
                    "ig_user_id": ig_user_id,
                })

    return prepared


@app.post("/instagram/prepare_replies")
async def instagram_prepare_replies(
    tone: str = "friendly",
    max_posts: int = 3,
    max_comments: int = 5,
    current_user: dict = Depends(get_current_user),
):
    ig_token = token_store.get_token_for_user(current_user["id"], provider="instagram")
    if not ig_token:
        return {
            "status": "no_account",
            "message": "Please connect an Instagram Business or Creator account first.",
            "replies": []
        }

    ig_user_id = ig_token.get("ig_account_id")
    page_access_token = ig_token.get("page_access_token")
    if not ig_user_id or not page_access_token:
        return {
            "status": "no_account",
            "message": "Missing Instagram access token. Please reconnect Instagram.",
            "replies": []
        }

    try:
        prepared = await _prepare_instagram_replies(
            tone=tone,
            max_posts=max_posts,
            max_comments=max_comments,
            current_user=current_user
        )
    except Exception as e:
        logging.exception("Failed to prepare Instagram replies")
        return {
            "status": "error",
            "message": "Something went wrong while preparing replies.",
            "replies": []
        }

    if not prepared:
        return {
            "status": "no_comments",
            "message": "No comments found.",
            "replies": []
        }

    return {
        "status": "ok",
        "message": "Replies prepared successfully.",
        "replies": prepared
    }


@app.post("/instagram/post_replies")
async def instagram_post_replies(
    tone: str = "friendly",
    max_posts: int = 3,
    max_comments: int = 5,
    current_user: dict = Depends(get_current_user),
) -> dict:
    if not _httpx_available:
        raise HTTPException(status_code=500, detail="httpx is required for Instagram API interactions")

    ig_token = token_store.get_token_for_user(current_user["id"], provider="instagram")
    if not ig_token:
        return {
            "status": "no_account",
            "message": "Please connect an Instagram Business or Creator account first.",
            "replies_posted": {}
        }

    page_access_token = ig_token.get("page_access_token")
    if not page_access_token:
        return {
            "status": "no_account",
            "message": "Instagram token not found. Please reconnect Instagram.",
            "replies_posted": {}
        }

    try:
        prepared = await _prepare_instagram_replies(
            tone=tone,
            max_posts=max_posts,
            max_comments=max_comments,
            current_user=current_user
        )
    except Exception as e:
        logging.exception("Failed to prepare Instagram replies for posting")
        return {
            "status": "error",
            "message": "Something went wrong while preparing replies.",
            "replies_posted": {}
        }

    if not prepared:
        return {
            "status": "no_comments",
            "message": "No comments found to reply to.",
            "replies_posted": {}
        }

    results: dict[str, int] = {}

    async with httpx.AsyncClient(timeout=15) as client:
        for item in prepared:
            comment_id = item["comment_id"]
            reply_message = item["reply"]
            ig_user_id = item["ig_user_id"]

            try:
                await client.post(
                    f"https://graph.facebook.com/v18.0/{comment_id}/replies",
                    params={"access_token": page_access_token},
                    json={"message": reply_message},
                )
                results[ig_user_id] = results.get(ig_user_id, 0) + 1
            except Exception as e:
                logging.warning(f"Failed to post IG reply to comment {comment_id}: {e}")
                continue

    return {
        "status": "ok",
        "message": "Replies posted successfully.",
        "replies_posted": results
    }


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
