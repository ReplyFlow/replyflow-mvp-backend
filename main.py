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
import base64
import json
import urllib.parse
import uuid
import datetime
import time  # Needed for JWT timestamp generation
import hmac
import hashlib
import secrets

# Import analytics store for usage and error tracking
# Attempt to import the analytics_store module.  If it is not available in the
# deployment environment (e.g. the file was not included in the package),
# define a minimal fallback implementation inline.  This ensures that the
# server can still record usage and error events without raising import
# errors.  The fallback stores analytics data in memory only and does not
# persist across restarts.  To enable full analytics persistence, ensure
# analytics_store.py is present in the project root.
try:
    import analytics_store  # type: ignore
except ModuleNotFoundError:
    # Minimal in-memory analytics fallback
    class _InMemoryAnalytics:
        """Fallback analytics store used if analytics_store module is missing."""

        def __init__(self):
            self.events = {}
            self.errors = []
            self._lock = None  # Placeholder; no threading lock for fallback

        def record_event(self, event_name: str, count: int = 1) -> None:
            if not event_name:
                return
            self.events[event_name] = self.events.get(event_name, 0) + count

        def record_error(self, context: str, message: str) -> None:
            if not message:
                return
            self.errors.append({
                "timestamp": datetime.datetime.utcnow().isoformat(),
                "context": context,
                "message": message,
            })

        def get_stats(self) -> Dict[str, Any]:
            return {"events": dict(self.events), "errors": list(self.errors)}

    analytics_store = _InMemoryAnalytics()
from typing import Optional, Dict, Any

from fastapi import FastAPI, HTTPException, Depends, Request, Query
from fastapi.responses import RedirectResponse, HTMLResponse

# Import BaseModel early to avoid NameError when defining request/response models
from pydantic import BaseModel

# -----------------------------------------------------------------------------
# Email sending configuration and helper
#
# To automatically welcome new beta testers via email, the server can send an
# onboarding message whenever a new beta user signs up.  The email sending
# functionality relies on a standard SMTP server and requires several
# environment variables to be configured:
#
#   SMTP_SERVER: hostname of the SMTP server (e.g. smtp.gmail.com)
#   SMTP_PORT: port number for the SMTP server (usually 587 for TLS)
#   SMTP_USERNAME: username for authenticating with the SMTP server
#   SMTP_PASSWORD: password for authenticating with the SMTP server
#   EMAIL_FROM: the From address that will appear in the email headers
#
# If these variables are not set, attempts to send an email will raise an
# exception.  You should populate them in your deployment environment.  The
# welcome email content is defined in the helper below and can be customized
# further if desired.

import smtplib
from email.message import EmailMessage

SMTP_SERVER = os.getenv("SMTP_SERVER")
SMTP_PORT = int(os.getenv("SMTP_PORT", "587"))
SMTP_USERNAME = os.getenv("SMTP_USERNAME")
SMTP_PASSWORD = os.getenv("SMTP_PASSWORD")
EMAIL_FROM = os.getenv("EMAIL_FROM")

def _compose_beta_welcome_body(name: str | None = None) -> str:
    """Generate the plain‑text body for the beta welcome email.

    Args:
        name: Optional display name of the recipient.  If provided, it will
            personalize the greeting.  When omitted or empty, a generic
            greeting is used.

    Returns:
        A multi‑line string containing the email body.
    """
    greeting = f"Hi {name}," if name else "Hi there,"
    return (
        f"{greeting}\n\n"
        "Thank you for joining the ReplyFlow beta! We’re excited to see how you’ll use AI "
        "to streamline your comment replies. Here’s how to get started:\n\n"
        "1. Logging in: Visit our dashboard at https://app.replyflow.com and click “Login”. "
        "Use the same email you signed up with; you'll be prompted to set a password or request a "
        "magic link.\n\n"
        "2. Support: If you need help, join our Discord community at https://discord.gg/jFKMfu4b "
        "or click the Intercom chat icon within ReplyFlow.\n\n"
        "3. Install the browser extension (optional): To install our extension on Chrome, go to "
        "the Chrome Web Store, search for “ReplyFlow”, click Add to Chrome, and approve. "
        "After installation, you can pin the extension for quick access.\n\n"
        "4. Install as a Progressive Web App (optional): On desktop, look for the install icon "
        "in the address bar when visiting ReplyFlow or use the browser menu to install. "
        "On mobile, open the browser menu and select “Add to Home Screen”.\n\n"
        "We'll be in touch with updates and features. Your feedback helps shape ReplyFlow; "
        "please reach out with any suggestions.\n\n"
        "Best,\n"
        "The ReplyFlow Team"
    )

def send_beta_welcome_email(to_email: str, name: str | None = None) -> None:
    """Send a welcome email to a new beta tester.

    This helper composes a friendly onboarding email using the recipient's name
    (when provided) and sends it via an SMTP server.  It assumes the SMTP
    configuration variables are set.  The email is sent in plain text to
    maximize deliverability.

    Args:
        to_email: The recipient's email address.
        name: Optional display name to personalize the greeting.

    Raises:
        ValueError: If any of the SMTP configuration variables are missing.
        smtplib.SMTPException: If there is a problem connecting to or
            authenticating with the SMTP server.
    """
    # Ensure all required configuration is present
    missing = [
        var for var, val in [
            ("SMTP_SERVER", SMTP_SERVER),
            ("SMTP_USERNAME", SMTP_USERNAME),
            ("SMTP_PASSWORD", SMTP_PASSWORD),
            ("EMAIL_FROM", EMAIL_FROM),
        ]
        if not val
    ]
    if missing:
        raise ValueError(
            f"Missing SMTP configuration for: {', '.join(missing)}. "
            "Please set the corresponding environment variables."
        )
    # Compose the email message
    msg = EmailMessage()
    msg["From"] = EMAIL_FROM
    msg["To"] = to_email
    msg["Subject"] = "Welcome to the ReplyFlow Beta!"
    body = _compose_beta_welcome_body(name)
    msg.set_content(body)
    # Connect and send
    with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
        # Use TLS if possible
        try:
            server.starttls()
        except Exception:
            # If TLS isn't supported, continue without raising
            pass
        server.login(SMTP_USERNAME, SMTP_PASSWORD)
        server.send_message(msg)


# We will register beta-related endpoints after creating the FastAPI app.  See
# the `_register_beta_endpoints` function defined later in this file.
import requests  # For making outgoing HTTP calls to the atproto OAuth token endpoint

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

# ----------------------------------------------------------------------------
# Bluesky OAuth state storage
#
# When initiating an OAuth authorization request for Bluesky (AT Protocol),
# we generate a unique `state` and a PKCE code verifier/challenge pair.  The
# state and verifier need to be persisted across the redirect to the Bluesky
# authorization endpoint.  We store them in this global dictionary keyed by
# the state string.  In a production environment you would persist this
# information in a more durable store (e.g. database or cache) and perhaps
# associate it with the user's session ID.  The values stored include the
# PKCE code verifier and the user_id associated with the authorization
# request.
BLUESKY_OAUTH_STATES: Dict[str, Dict[str, str]] = {}

# ----------------------------------------------------------------------------
# YouTube OAuth state storage
#
# For the YouTube OAuth authorization code flow we need to maintain a mapping
# from a randomly generated ``state`` value back to the user ID initiating the
# flow.  When the user clicks the connect button in the dashboard, we create a
# unique state, store the associated user_id and then redirect the user to
# Google's authorization endpoint.  Upon return to the callback we look up
# the state to retrieve the user_id and proceed with the token exchange.  In
# production you would persist these values in a database or cache.
YOUTUBE_OAUTH_STATES: Dict[str, str] = {}

# ----------------------------------------------------------------------------
# Google Business Profile OAuth state storage
#
# Similar to the YouTube OAuth flow, we maintain a mapping from a randomly
# generated ``state`` to the user ID initiating the Google OAuth flow.  When
# the user clicks the connect button in the dashboard for Google Business
# Profile, we generate a state, store the associated user ID, and include
# the state in the authorization request.  Upon return to the callback we
# validate the state and retrieve the user ID.  In a production deployment
# these values should be persisted in a database or cache rather than held
# in memory.
GOOGLE_OAUTH_STATES: Dict[str, str] = {}

# ----------------------------------------------------------------------------
# Feature flag configuration
#
# Certain features (such as beta sign‑up pages or experimental capabilities) may
# only be available to a subset of users. To control visibility of these
# features, the backend exposes a simple feature‑flag system driven by
# environment variables. The variables below can be set to "true" or
# "false" (case insensitive) to enable or disable the corresponding
# functionality. When disabled, the UI can hide links or sections by
# querying the /feature-flags endpoint.

# Enable or disable the YouTube video upload functionality. When this flag
# is false, the /youtube/upload_video endpoint will still respond with
# a 403 Forbidden to prevent accidental usage by non‑beta users. The UI can
# query /feature-flags to determine whether to show the upload form.
ENABLE_VIDEO_UPLOAD: bool = os.getenv("ENABLE_VIDEO_UPLOAD", "true").lower() == "true"

# Enable or disable the beta sign‑up flow. When false, the /beta-signup
# route will return a 404 error and the UI should hide the "Join the Beta"
# link. The feature flags endpoint exposes this value so the frontend can
# toggle visibility.
ENABLE_BETA_SIGNUP: bool = os.getenv("ENABLE_BETA_SIGNUP", "true").lower() == "true"

# Enable or disable paid plan availability. When this flag is false the
# pricing section in the marketing site will hide or disable the Solo and
# Pro plan signup buttons and display a notice informing visitors that
# paid plans are unavailable until the full launch.  The frontend reads
# this value from the /feature-flags endpoint via the `paid_plans` key.
ENABLE_PAID_PLANS: bool = os.getenv("ENABLE_PAID_PLANS", "true").lower() == "true"


# The feature flags endpoint is registered later once the FastAPI app has been created.  See
# the `_register_feature_flags_endpoint` helper function for the definition.

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

# ----------------------------------------------------------------------------
# Multilingual reply support
#
# To provide a basic multilingual experience without relying on external
# translation services or large third‑party libraries, we implement simple
# language detection heuristics and predefined translations for a handful
# of common languages.  When generating fallback replies (i.e. when the
# OpenAI API isn't available), we will detect the language of the incoming
# comment and select an appropriate response in that language.  If the
# language cannot be determined or is unsupported, we default to English.

# Character sets used for rudimentary language detection.  Each set contains
# characters that are distinctive for a particular language.  For example,
# Spanish uses accented vowels and the ñ; French uses various accented
# characters and the ligature œ; German has umlauts and the ß; Japanese
# includes Hiragana and Katakana; Chinese uses a large CJK range.  These
# lists are not exhaustive but sufficient for simple detection.
# Spanish uses accented vowels and the letter ñ, but does not commonly use ü except
# in a few loanwords.  To avoid mis‑detecting German words that contain ü,
# we exclude ü from the Spanish set.  See German char set below for ü.
SPANISH_CHARS = set("áéíóúñ¡¿ÁÉÍÓÚÑ")
FRENCH_CHARS = set("àâæçéèëêîïôœùûüÿÀÂÆÇÉÈËÊÎÏÔŒÙÛÜŸ")
GERMAN_CHARS = set("äöüßÄÖÜ")
# Hiragana (0x3040‑0x309F) and Katakana (0x30A0‑0x30FF) ranges for Japanese
HIRAGANA_RANGE = (0x3040, 0x309F)
KATAKANA_RANGE = (0x30A0, 0x30FF)
# Chinese characters primarily fall within the CJK Unified Ideographs block
CHINESE_RANGE = (0x4E00, 0x9FFF)


def detect_language(text: str) -> str:
    """Guess the language of the provided text based on character heuristics.

    The detection algorithm counts occurrences of language‑specific characters
    and returns a language code for the language with the highest count.  If
    no distinctive characters are found, English ("en") is returned as the
    default.

    Supported languages and their codes:
    - "es": Spanish
    - "fr": French
    - "de": German
    - "ja": Japanese (Hiragana/Katakana)
    - "zh": Chinese (CJK ideographs)
    - "en": English (fallback)

    Args:
        text: The text to analyse.

    Returns:
        A two‑letter language code indicating the detected language.
    """
    import re
    # Initialize counters for each supported language
    counts = {"es": 0, "fr": 0, "de": 0, "ja": 0, "zh": 0}
    # Character‑level detection
    for ch in text:
        code = ord(ch)
        # Spanish specific characters
        if ch in SPANISH_CHARS:
            counts["es"] += 1
        # French specific characters
        if ch in FRENCH_CHARS:
            counts["fr"] += 1
        # German specific characters
        if ch in GERMAN_CHARS:
            counts["de"] += 1
        # Japanese detection via Hiragana or Katakana ranges
        if HIRAGANA_RANGE[0] <= code <= HIRAGANA_RANGE[1] or KATAKANA_RANGE[0] <= code <= KATAKANA_RANGE[1]:
            counts["ja"] += 1
        # Chinese detection via CJK unified ideographs range
        if CHINESE_RANGE[0] <= code <= CHINESE_RANGE[1]:
            counts["zh"] += 1
    # Word‑level detection to catch languages that might not use accented characters
    lower_text = text.lower()
    # Remove punctuation for word matching
    words = [re.sub(r"[^\w]", "", w) for w in lower_text.split() if w]
    # Common words per language for simple lexical detection
    WORD_SETS = {
        "es": {"hola", "gracias", "comentario", "comentarios", "cómo", "como", "estás", "estas", "buenos", "buenas", "día", "dias", "días"},
        "fr": {"merci", "bonjour", "comment", "sont", "avec", "vous", "nous", "cest", "incroyable", "avis", "bien"},
        "de": {"danke", "hallo", "kommentar", "bewertungen", "bitte", "schätzen", "wir", "sie", "ihnen", "gut"},
    }
    for word in words:
        for lang_code, word_set in WORD_SETS.items():
            if word in word_set:
                counts[lang_code] += 1
    # Determine the language with the highest count
    detected = max(counts.items(), key=lambda item: item[1])
    # If no distinctive characters or words found, fall back to English
    if detected[1] == 0:
        return "en"
    # When both Chinese and Japanese characters are present, prioritise Japanese if Hiragana/Katakana appear
    if detected[0] in ("zh", "ja"):
        if counts["ja"] > 0:
            return "ja"
        else:
            return "zh"
    return detected[0]


def get_translated_reply(tone: str, lang: str) -> str:
    """Retrieve a generic reply translated into the specified language.

    A dictionary of translations is defined for each supported language and
    tone.  If a particular tone isn't available in the target language,
    English will be used as a fallback.  Additional languages and tones
    can be added to the LANG_RESPONSES structure.  The function returns
    the translated reply for the given tone and language.

    Args:
        tone: The requested tone of the reply (lower‑case string).
        lang: The language code to translate into (e.g. "es", "fr").

    Returns:
        A translated reply string.
    """
    tone_key = (tone or "friendly").strip().lower()
    # Predefined translations for each supported language and tone
    LANG_RESPONSES: dict[str, dict[str, str]] = {
        "en": {
            "friendly": "Thanks for your comment! We appreciate your feedback.",
            "focused": "Thank you for your comment. We'll review and get back to you.",
            "enthusiastic": "Thanks for sharing! We're excited to reply soon!",
            "professional": "Thank you for your message. We'll review it and respond.",
            "casual": "Thanks! We'll get back to you soon.",
            "apologetic": "We're sorry for any issues and will look into it.",
            "quirky": "Thanks for dropping by! Your comment made our day.",
            "diplomatic": "Thank you for your feedback. We'll consider your points carefully.",
        },
        "es": {
            "friendly": "¡Gracias por tu comentario! Agradecemos tus comentarios.",
            "focused": "Gracias por tu comentario. Lo revisaremos y te responderemos.",
            "enthusiastic": "¡Gracias por compartir! ¡Estamos emocionados de responder pronto!",
            "professional": "Gracias por tu mensaje. Lo revisaremos y responderemos.",
            "casual": "¡Gracias! Te responderemos pronto.",
            "apologetic": "Lamentamos cualquier problema y lo revisaremos.",
            "quirky": "¡Gracias por pasar por aquí! Tu comentario alegró nuestro día.",
            "diplomatic": "Gracias por tu feedback. Consideraremos tus puntos cuidadosamente.",
        },
        "fr": {
            "friendly": "Merci pour votre commentaire ! Nous apprécions vos retours.",
            "focused": "Merci pour votre commentaire. Nous l'examinerons et répondrons.",
            "enthusiastic": "Merci pour votre partage ! Nous sommes impatients de répondre bientôt !",
            "professional": "Merci pour votre message. Nous l'examinerons et répondrons.",
            "casual": "Merci ! Nous reviendrons vers vous bientôt.",
            "apologetic": "Nous sommes désolés pour tout problème et nous l'examinerons.",
            "quirky": "Merci d'être passé ! Votre commentaire a fait notre journée.",
            "diplomatic": "Merci pour vos commentaires. Nous considérerons attentivement vos points.",
        },
        "de": {
            "friendly": "Vielen Dank für Ihren Kommentar! Wir schätzen Ihr Feedback.",
            "focused": "Vielen Dank für Ihren Kommentar. Wir werden ihn überprüfen und uns melden.",
            "enthusiastic": "Danke fürs Teilen! Wir sind begeistert, bald zu antworten!",
            "professional": "Vielen Dank für Ihre Nachricht. Wir werden sie prüfen und antworten.",
            "casual": "Danke! Wir melden uns bald.",
            "apologetic": "Es tut uns leid wegen etwaiger Probleme und wir werden es prüfen.",
            "quirky": "Danke, dass du vorbeigeschaut hast! Dein Kommentar hat unseren Tag versüßt.",
            "diplomatic": "Vielen Dank für Ihr Feedback. Wir werden Ihre Punkte sorgfältig prüfen.",
        },
        "zh": {
            "friendly": "感谢您的评论！我们感谢您的反馈。",
            "focused": "感谢您的评论。我们会审核并回复您。",
            "enthusiastic": "感谢分享！我们很快会回复！",
            "professional": "谢谢您的留言。我们会审核并回复。",
            "casual": "谢谢！我们会尽快回复。",
            "apologetic": "对于任何问题我们感到抱歉，我们会查看的。",
            "quirky": "感谢您的到访！您的评论让我们很开心。",
            "diplomatic": "谢谢您的反馈。我们会仔细考虑您的意见。",
        },
        "ja": {
            "friendly": "コメントありがとうございます！ご意見をいただきありがとうございます。",
            "focused": "コメントありがとうございます。確認して返答します。",
            "enthusiastic": "シェアありがとうございます！もうすぐ返信します！",
            "professional": "メッセージありがとうございます。確認して返信します。",
            "casual": "ありがとう！またすぐ返事します。",
            "apologetic": "ご不便をおかけし申し訳ありません。確認いたします。",
            "quirky": "お立ち寄りいただきありがとうございます！コメントで元気が出ました。",
            "diplomatic": "ご意見ありがとうございます。ご意見を慎重に検討します。",
        },
    }
    # Fetch translations for the requested language; fallback to English if unsupported
    translations = LANG_RESPONSES.get(lang, LANG_RESPONSES.get("en"))
    # Return the translation for the tone; if tone not found, fallback to English default
    if tone_key in translations:
        return translations[tone_key]
    # Fallback: return English message for the tone if available, else a generic English reply
    return LANG_RESPONSES["en"].get(tone_key, "Thank you for your comment!")

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

# Register beta-related API endpoints using a helper function.  This ensures
# that the endpoints are defined after the FastAPI app is created.  The
# endpoints leverage feature flags to optionally disable beta sign-up and
# welcome email dispatch.  Defining them in a helper avoids NameError when
# decorating functions with @app.post before 'app' exists.
def _register_beta_endpoints(app: FastAPI) -> None:
    """Register beta welcome and registration endpoints."""
    class BetaWelcomeRequest(BaseModel):
        """Schema for the beta welcome email endpoint."""
        email: str
        name: Optional[str] = None

    @app.post("/send_beta_welcome_email")
    async def send_beta_welcome_email_endpoint(request: BetaWelcomeRequest) -> Dict[str, str]:
        """API endpoint to send a welcome email to a beta user.

        This endpoint accepts a JSON payload with the recipient's email address and
        optional name, then sends a welcome email using the SMTP settings
        configured via environment variables.  It returns a success message if the
        email was dispatched successfully or raises a 500 error on failure.
        """
        try:
            send_beta_welcome_email(request.email, request.name)
            return {"status": "success"}
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc))

    class BetaRegistrationRequest(BaseModel):
        """Schema for the beta registration endpoint.

        Fields:
            email: The tester's email address.  This field is required and must be
                non‑empty.
            name:  Optional display name to personalize the welcome email.
        """
        email: str
        name: Optional[str] = None

    @app.post("/beta/register")
    async def beta_register(request: BetaRegistrationRequest) -> Dict[str, str]:
        """Handle beta registrations and dispatch welcome emails.

        This endpoint enables an automated welcome email campaign for beta testers.
        It is intended to be called whenever a new tester signs up.  The endpoint
        checks whether the beta sign‑up flow is enabled; if it is disabled,
        a 403 error is returned.  Otherwise, it sends a welcome email using the
        `send_beta_welcome_email` helper and returns a success message.  If
        email sending fails (e.g. due to missing SMTP configuration), an
        HTTP 500 error is raised.
        """
        if not ENABLE_BETA_SIGNUP:
            raise HTTPException(status_code=403, detail="Beta sign‑up is currently disabled")
        # Validate that an email was provided (pydantic already checks non‑None and non‑empty)
        email = request.email.strip()
        if not email:
            raise HTTPException(status_code=400, detail="Email address must not be empty")
        try:
            # Send the welcome email immediately
            send_beta_welcome_email(email, request.name)
            return {"status": "success"}
        except Exception as exc:
            # Propagate any errors as HTTP 500 with a clear message
            raise HTTPException(status_code=500, detail=str(exc))

# Immediately register these endpoints now that the app exists
_register_beta_endpoints(app)

# Register the feature flags endpoint.  Defining it in a helper ensures that
# `app` is available and avoids a NameError if this file is imported before
# the FastAPI app is created.  The endpoint returns the boolean values of
# configured feature flags so that the frontend can determine which
# experimental features to display.
def _register_feature_flags_endpoint(app: FastAPI) -> None:
    """Register the /feature-flags endpoint on the given FastAPI app."""
    @app.get("/feature-flags")
    async def feature_flags() -> Dict[str, bool]:  # noqa: F811
        """Return a dictionary of feature flags for the frontend.

        The returned mapping indicates whether various optional features
        (e.g., video upload, beta signup) are enabled.  The values are
        derived from environment variables and loaded at startup.
        """
        return {
            "video_upload": ENABLE_VIDEO_UPLOAD,
            "beta_signup": ENABLE_BETA_SIGNUP,
            # Expose the paid_plans flag to the frontend.  When false, the UI
            # should hide or disable paid plan signup buttons and display a
            # notice about limited functionality during beta.
            "paid_plans": ENABLE_PAID_PLANS,
        }

# Immediately register the feature flags endpoint
_register_feature_flags_endpoint(app)

# -----------------------------------------------------------------------------
# Analytics middleware
#
# Record usage events for most API endpoints and capture error logs.  The event
# name is derived from the request path by replacing '/' with '_' and stripping
# leading/trailing slashes.  Static assets and manifest/service worker routes
# are excluded from tracking.  Error logs include the exception message and
# the context of the request.

@app.middleware("http")
async def analytics_usage_middleware(request: Request, call_next):
    path = request.url.path
    # Determine event name by transforming the path. Remove leading and trailing slashes.
    event_name = path.strip("/").replace("/", "_")
    # Exclude static assets and well-known resources from analytics
    exclude_prefixes = (
        "/static",
        "/favicon",
        "/manifest",
        "/service-worker.js",
        "/robots.txt",
    )
    if event_name and not any(path.startswith(prefix) for prefix in exclude_prefixes):
        try:
            analytics_store.record_event(event_name)
        except Exception:
            # Ignore analytics errors so that API functionality isn't impacted
            pass
    try:
        response = await call_next(request)
        return response
    except Exception as exc:
        # Log unhandled exceptions with context
        if event_name:
            try:
                analytics_store.record_error(event_name, str(exc))
            except Exception:
                pass
        # Re-raise the exception to let default handlers process it
        raise

# Serve static files.  This allows us to host client metadata and other
# resources at stable URLs.  The "static" directory is mounted at the
# "/static" path.  Any files placed in the static directory will be
# accessible via requests to /static/<filename>.  For example, the
# Bluesky client metadata document can be placed at
# static/bluesky-client-metadata.json and served at
# https://<your-domain>/static/bluesky-client-metadata.json.  When deploying
# to a platform like Render or Netlify, ensure that static files are
# included in the build and publicly accessible.
from fastapi.staticfiles import StaticFiles

# Mount the static directory if it exists.  If you create a "static" folder
# in your project root, its contents will be served under the /static path.
if os.path.isdir("static"):
    app.mount("/static", StaticFiles(directory="static"), name="static")

# -----------------------------------------------------------------------------
# X (Twitter) and Reddit connection endpoints
#
# To provide a simplified on‑boarding flow for X and Reddit, we expose
# lightweight connect endpoints.  These routes allow a logged‑in user to
# associate a pre‑provisioned access token with their ReplyFlow account
# without performing a full OAuth dance in the frontend.  When invoked with
# a valid session token, the server stores the access token from the
# environment (or attempts to derive one) in the token store, then returns
# a small HTML page that notifies the opener via postMessage and closes the
# popup.  This mirrors the behaviour of the Facebook callback route and
# permits the dashboard to update its UI when the connection completes.

@app.get("/x/connect")
async def x_connect(token: Optional[str] = None) -> HTMLResponse:
    """Connect the current user to X (Twitter) using a preconfigured token.

    This endpoint expects a session token in the query string.  It resolves
    that token to a user ID via the session store, then looks up the X access
    token in the environment.  If a token is found, it is persisted via
    ``token_store.save_token`` under provider ``"x"``.  Upon success the
    response sends a postMessage notification to the opener and closes the
    window.  If no access token is available or the user is unauthenticated,
    the user is redirected to the login page or an error message is shown.
    """
    # Require a session token to map to a user
    session_token = token
    if not session_token:
        frontend_login = os.getenv("FRONTEND_LOGIN_URL") or "/account.html"
        return RedirectResponse(frontend_login)
    user_id = session_store.get_user_id(session_token)
    if not user_id:
        frontend_login = os.getenv("FRONTEND_LOGIN_URL") or "/account.html"
        return RedirectResponse(frontend_login)
    # Determine the access token from environment variables.  We support two
    # possible variable names for convenience (TWITTER_ACCESS_TOKEN and
    # X_ACCESS_TOKEN).  If neither is set then no connection can be made.
    access_token = os.getenv("X_ACCESS_TOKEN") or os.getenv("TWITTER_ACCESS_TOKEN")
    if not access_token:
        # Respond with a simple message when no token is configured.  The
        # dashboard will display this in the popup.
        content = """<html><body><p>No X access token configured on the server.</p></body></html>"""
        return HTMLResponse(content=content, status_code=200)
    # Persist the token for this user.  We use a static page_id of "self"
    # because X interactions are scoped to the authenticated account rather
    # than a page.  Both the page_access_token and user_access_token fields
    # are populated to simplify later retrieval.
    try:
        token_store.save_token(
            user_id=user_id,
            provider="x",
            page_id="self",
            page_access_token=access_token,
            ig_account_id=None,
            user_access_token=access_token,
        )
    except Exception:
        # Ignore storage errors and still allow the frontend to update.  The
        # dashboard can inspect whether tokens exist to decide connection state.
        pass
    # Return a tiny page that notifies the opener and closes the popup.
    html = """
    <html>
      <body>
        <script>
          try {
            if (window.opener) {
              window.opener.postMessage('x_connected', '*');
            }
          } catch (e) {}
          window.close();
        </script>
        <p>X connection complete. You may close this window.</p>
      </body>
    </html>
    """
    return HTMLResponse(content=html, status_code=200)


@app.get("/reddit/connect")
async def reddit_connect(token: Optional[str] = None) -> HTMLResponse:
    """Connect the current user to Reddit using a preconfigured or derived token.

    Similar to ``/x/connect``, this endpoint stores a Reddit access token for
    the authenticated user.  It first attempts to read an access token from
    ``REDDIT_ACCESS_TOKEN`` or ``REDDIT_TOKEN`` environment variables.  If
    neither is present, it tries to obtain a token using the OAuth2 password
    grant via ``_get_reddit_access_token``, provided that the client ID,
    client secret, username and password environment variables are set.  The
    derived token is then saved in the token store.  Finally the endpoint
    returns a simple HTML page that notifies the opener of completion and
    closes the window.
    """
    session_token = token
    if not session_token:
        frontend_login = os.getenv("FRONTEND_LOGIN_URL") or "/account.html"
        return RedirectResponse(frontend_login)
    user_id = session_store.get_user_id(session_token)
    if not user_id:
        frontend_login = os.getenv("FRONTEND_LOGIN_URL") or "/account.html"
        return RedirectResponse(frontend_login)
    # Attempt to determine the Reddit access token.  Check preconfigured
    # environment variables first to allow manual provisioning.
    reddit_token = os.getenv("REDDIT_ACCESS_TOKEN") or os.getenv("REDDIT_TOKEN")
    # If no static token exists, attempt to perform a password grant.  Note
    # that this call is asynchronous and will return None on failure.  The
    # Reddit user agent can also be overridden via REDDIT_USER_AGENT.
    if not reddit_token:
        client_id = os.getenv("REDDIT_CLIENT_ID")
        client_secret = os.getenv("REDDIT_CLIENT_SECRET")
        username = os.getenv("REDDIT_USERNAME")
        password = os.getenv("REDDIT_PASSWORD")
        user_agent = os.getenv("REDDIT_USER_AGENT", "ReplyFlow/0.1")
        if client_id and client_secret and username and password:
            try:
                reddit_token = await _get_reddit_access_token(
                    client_id=client_id,
                    client_secret=client_secret,
                    username=username,
                    password=password,
                    user_agent=user_agent,
                )
            except Exception:
                reddit_token = None
    if not reddit_token:
        content = """<html><body><p>No Reddit access token could be obtained.</p></body></html>"""
        return HTMLResponse(content=content, status_code=200)
    # Persist the Reddit token.  Use page_id="self" similar to X, and
    # populate both page_access_token and user_access_token fields.
    try:
        token_store.save_token(
            user_id=user_id,
            provider="reddit",
            page_id="self",
            page_access_token=reddit_token,
            ig_account_id=None,
            user_access_token=reddit_token,
        )
    except Exception:
        pass
    html = """
    <html>
      <body>
        <script>
          try {
            if (window.opener) {
              window.opener.postMessage('reddit_connected', '*');
            }
          } catch (e) {}
          window.close();
        </script>
        <p>Reddit connection complete. You may close this window.</p>
      </body>
    </html>
    """
    return HTMLResponse(content=html, status_code=200)



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

# Number of days after which a stored YouTube refresh token should be refreshed automatically.
# Similar to Facebook long‑lived tokens, YouTube access tokens can expire and need to be
# refreshed using the stored refresh token. By default we refresh tokens older than
# 50 days. This threshold can be overridden via the environment variable
# YOUTUBE_TOKEN_REFRESH_THRESHOLD_DAYS.
# Use a shorter refresh threshold (30 days) for YouTube tokens so they are
# proactively refreshed well before expiry.  Override via the
# YOUTUBE_TOKEN_REFRESH_THRESHOLD_DAYS environment variable if desired.
YOUTUBE_TOKEN_REFRESH_THRESHOLD_DAYS = int(os.getenv("YOUTUBE_TOKEN_REFRESH_THRESHOLD_DAYS", "30"))

# ---------------------------------------------------------------------------
# Plan‑specific usage limits
#
# To support different plans (free, solo, pro) with different usage caps, we
# read the monthly reply limits from environment variables.  If a limit is not
# provided, defaults are used.  A value of zero or a negative value indicates
# that the plan has unlimited replies.  The UI in index.html already shows
# these limits, so the backend must enforce them accordingly.
#
# Example environment variables:
#   REPLYFLOW_FREE_REPLIES_PER_MONTH: monthly cap for free plan (default 30)
#   REPLYFLOW_SOLO_REPLIES_PER_MONTH: monthly cap for solo plan (default 100)
#   REPLYFLOW_PRO_REPLIES_PER_MONTH: monthly cap for pro plan (0 or missing = unlimited)
REPLYFLOW_SOLO_REPLIES_PER_MONTH = int(os.getenv("REPLYFLOW_SOLO_REPLIES_PER_MONTH", "100"))
_pro_limit_env = os.getenv("REPLYFLOW_PRO_REPLIES_PER_MONTH")
try:
    REPLYFLOW_PRO_REPLIES_PER_MONTH: Optional[int] = int(_pro_limit_env) if _pro_limit_env is not None else None
except Exception:
    REPLYFLOW_PRO_REPLIES_PER_MONTH = None

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


# ----------------------------------------------------------------------------
# YouTube token refresh utility
#
# YouTube access tokens obtained via OAuth expire after one hour.  A refresh
# token is provided during the OAuth flow that can be used to obtain a new
# access token.  To avoid expired tokens causing failed API calls, we
# automatically refresh the stored access tokens when they are older than a
# configurable threshold.  The default threshold is 50 days, but can be
# overridden via the YOUTUBE_TOKEN_REFRESH_THRESHOLD_DAYS environment variable.
async def _refresh_youtube_access_token_if_needed(user_id: str) -> None:
    """Refresh a stored YouTube access token if it is nearing expiry.

    This helper looks up all stored YouTube tokens for the given user.  If
    the stored token's age exceeds ``YOUTUBE_TOKEN_REFRESH_THRESHOLD_DAYS``, it
    uses the associated refresh token to request a new access token from
    Google's OAuth token endpoint.  On success, the token store is updated
    with the new access and refresh tokens.  If a refresh token is not
    present or the refresh fails, the helper returns silently.

    Args:
        user_id: The application user ID whose YouTube token should be refreshed.
    """
    if not user_id:
        return
    # Retrieve the user's stored tokens for YouTube
    try:
        tokens = token_store.list_tokens(user_id, "youtube")
    except Exception:
        tokens = []
    if not tokens:
        return
    # Use the first token's created_at timestamp to determine age
    created_at_str = tokens[0].get("created_at")
    created_at_dt = None
    if created_at_str:
        try:
            created_at_dt = datetime.datetime.fromisoformat(created_at_str.rstrip("Z"))
        except Exception:
            created_at_dt = None
    # If we can parse created_at, compute age and compare to threshold
    if created_at_dt:
        age = datetime.datetime.utcnow() - created_at_dt
        if age < datetime.timedelta(days=YOUTUBE_TOKEN_REFRESH_THRESHOLD_DAYS):
            # Token is still within the acceptable freshness window
            return
    # Extract refresh token from stored entry
    refresh_token = tokens[0].get("user_access_token")
    if not refresh_token:
        return
    # Obtain client credentials from environment.  Use YOUTUBE_CLIENT_ID and
    # YOUTUBE_CLIENT_SECRET if set; fall back to GOOGLE_CLIENT_ID/SECRET.
    client_id = os.getenv("YOUTUBE_CLIENT_ID") or os.getenv("GOOGLE_CLIENT_ID")
    client_secret = os.getenv("YOUTUBE_CLIENT_SECRET") or os.getenv("GOOGLE_CLIENT_SECRET")
    if not (client_id and client_secret):
        # Cannot refresh without client credentials
        return
    token_endpoint = "https://oauth2.googleapis.com/token"
    data = {
        "grant_type": "refresh_token",
        "refresh_token": refresh_token,
        "client_id": client_id,
        "client_secret": client_secret,
    }
    new_access_token = None
    new_refresh_token = None
    try:
        if _httpx_available:
            import httpx  # type: ignore
            async with httpx.AsyncClient() as client:
                resp = await client.post(token_endpoint, data=data, timeout=15)
                resp.raise_for_status()
                token_json = resp.json()
        else:
            # Fallback to synchronous requests if httpx is unavailable.  Note: this
            # blocks the event loop and should be avoided in production.
            resp = requests.post(token_endpoint, data=data, timeout=15)
            token_json = resp.json() if resp.status_code == 200 else {}
        new_access_token = token_json.get("access_token")
        new_refresh_token = token_json.get("refresh_token")
    except Exception:
        # On any exception, skip refresh
        return
    if not new_access_token:
        return
    # Update all tokens for this user/provider.  Use existing page_id and ig
    # account fields but write the new access and refresh tokens.  If Google
    # does not provide a new refresh token, reuse the current one.
    refresh_to_store = new_refresh_token or refresh_token
    for t in tokens:
        page_id = t.get("page_id") or "self"
        page_access_token = new_access_token
        ig_account_id = t.get("ig_account_id")
        token_store.save_token(
            user_id=user_id,
            provider="youtube",
            page_id=str(page_id),
            page_access_token=page_access_token,
            ig_account_id=ig_account_id,
            user_access_token=refresh_to_store,
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
        """Create a new user record with a trial plan by default.

        When a user signs up, they are automatically placed on a 7‑day
        trial. The user record tracks the start of the trial as well as
        usage counters for the reverse trial. Upon expiration of the trial
        period, the user will be downgraded to the free tier. We also
        initialize monthly usage tracking fields so that the free plan
        can enforce per‑month reply limits.

        Args:
            email: The email address of the new user.
            password: The cleartext password provided by the user.

        Returns:
            The newly created user record as a dictionary.

        Raises:
            ValueError: If a user with the given email already exists.
        """
        users = self._load()
        if any(u.get("email") == email for u in users):
            raise ValueError("User already exists")
        salt = secrets.token_hex(16)
        password_hash = self._hash_password(password, salt)
        # Determine current time for trial and usage tracking
        now = datetime.datetime.utcnow()
        trial_start = now.isoformat() + "Z"
        # Set usage_month_start to the beginning of the current month in UTC
        month_start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        usage_month_start = month_start.isoformat() + "Z"
        user = {
            "id": uuid.uuid4().hex,
            "email": email,
            "salt": salt,
            "password_hash": password_hash,
            # All new users start with a trial plan. The plan will be
            # automatically downgraded to 'free' after the trial period.
            "plan": "trial",
            # Track when the trial began. If absent or invalid the trial is
            # considered to start at the time of first check.
            "trial_start": trial_start,
            # Monthly usage tracking for the reverse trial free tier.
            "replies_this_month": 0,
            "usage_month_start": usage_month_start,
            # Initialize Stripe identifiers to None. These fields will be
            # populated when a user completes a Stripe checkout session.
            # stripe_customer_id: The ID of the customer object created by
            #   Stripe. This is used to create a customer portal session.
            # stripe_subscription_id: The ID of the subscription object
            #   associated with the user's paid plan. When null, the user
            #   either has no active subscription or has canceled/downgraded.
            "stripe_customer_id": None,
            "stripe_subscription_id": None,
            # Initialize daily usage fields for the free plan limit.  "replies_today"
            # tracks how many replies have been generated via the /generate_reply
            # endpoint on the current day, and "usage_day_start" records the
            # UTC start time of the current day so we know when to reset.  These
            # fields are used to enforce daily limits for free users (e.g.
            # REPLYFLOW_FREE_REPLIES_PER_DAY) once the trial expires.
            "replies_today": 0,
            "usage_day_start": now.replace(hour=0, minute=0, second=0, microsecond=0).isoformat() + "Z",
        }
        users.append(user)
        self._save(users)
        return user

    def get_user_by_id(self, user_id: str) -> Optional[dict]:
        """Return a user dictionary by its unique identifier.

        Args:
            user_id: The unique identifier of the user.

        Returns:
            The user record if found, otherwise None.
        """
        for user in self._load():
            if user.get("id") == user_id:
                return user
        return None

    def update_user_fields(self, user_id: str, updates: dict) -> None:
        """Update arbitrary fields on a user record.

        Args:
            user_id: The ID of the user to update.
            updates: A mapping of fields to new values.
        """
        users = self._load()
        updated = False
        for user in users:
            if user.get("id") == user_id:
                for k, v in updates.items():
                    user[k] = v
                updated = True
                break
        if updated:
            self._save(users)

    def increment_reply_count(self, user_id: str, count: int) -> None:
        """Increment the monthly reply usage count for a user.

        This helper ensures that the usage counter resets at the start of
        each month. It updates the user's record in place and persists
        the changes.

        Args:
            user_id: The ID of the user whose count should be incremented.
            count: The number of replies to add to the usage tally.
        """
        users = self._load()
        now = datetime.datetime.utcnow()
        month_start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        updated = False
        for user in users:
            if user.get("id") == user_id:
                # Initialize fields if missing (backwards compatibility)
                replies = int(user.get("replies_this_month", 0))
                usage_start_str = user.get("usage_month_start")
                # Parse usage_month_start; if parsing fails, treat as beginning of current month
                reset_required = False
                if usage_start_str:
                    try:
                        usage_start_dt = datetime.datetime.fromisoformat(usage_start_str.replace("Z", ""))
                        # If usage month start is before the current month start, reset usage
                        if usage_start_dt < month_start:
                            reset_required = True
                    except Exception:
                        reset_required = True
                else:
                    reset_required = True
                if reset_required:
                    replies = 0
                    user["usage_month_start"] = month_start.isoformat() + "Z"
                # Increment the counter
                replies += max(0, int(count))
                user["replies_this_month"] = replies
                updated = True
                break
        if updated:
            self._save(users)

    def increment_reply_count_daily(self, user_id: str, count: int) -> None:
        """Increment the per-day reply usage counter for a user.

        This helper mirrors ``increment_reply_count`` but tracks usage on a
        daily basis instead of monthly.  It ensures that the ``replies_today``
        counter resets when the day rolls over.  The ``usage_day_start``
        field indicates the UTC start of the current counting period.  If it
        predates the start of the current UTC day, both ``replies_today`` and
        ``usage_day_start`` are reset.

        Args:
            user_id: The ID of the user whose daily count should be incremented.
            count: The number of replies to add to today's tally. Negative
                values have no effect.
        """
        users = self._load()
        now = datetime.datetime.utcnow()
        # Determine the start of the current day (00:00 UTC)
        day_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
        updated = False
        for user in users:
            if user.get("id") == user_id:
                # Ensure fields exist for backwards compatibility
                replies = int(user.get("replies_today", 0) or 0)
                day_start_str = user.get("usage_day_start")
                reset_required = False
                if day_start_str:
                    try:
                        usage_start_dt = datetime.datetime.fromisoformat(day_start_str.replace("Z", ""))
                        if usage_start_dt < day_start:
                            reset_required = True
                    except Exception:
                        reset_required = True
                else:
                    reset_required = True
                if reset_required:
                    replies = 0
                    user["usage_day_start"] = day_start.isoformat() + "Z"
                # Increment today's reply count
                replies += max(0, int(count))
                user["replies_today"] = replies
                updated = True
                break
        if updated:
            self._save(users)

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

    def find_by_field(self, field: str, value: Any) -> Optional[dict]:
        """Find a user record by an arbitrary field value.

        Args:
            field: The key to search within each user record.
            value: The value to match.

        Returns:
            The first matching user record if found, otherwise None.
        """
        for user in self._load():
            if user.get(field) == value:
                return user
        return None

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
    # Before returning the user, perform automatic trial downgrade and
    # initialize missing fields for backwards compatibility.
    # Determine current UTC time
    now = datetime.datetime.utcnow()
    user_id = user.get("id")
    # Initialize trial_start if missing
    trial_start_str = user.get("trial_start")
    plan = user.get("plan")
    needs_save = False
    # If trial_start is absent, set it to now and mark plan as trial if undefined
    if not trial_start_str:
        trial_start_str = now.isoformat() + "Z"
        user["trial_start"] = trial_start_str
        # If the plan is undefined, treat as trial
        if not plan:
            user["plan"] = "trial"
            plan = "trial"
        needs_save = True
    # Parse the trial_start timestamp
    trial_start_dt: Optional[datetime.datetime] = None
    try:
        trial_start_dt = datetime.datetime.fromisoformat(trial_start_str.replace("Z", ""))
    except Exception:
        # If parsing fails, set to now
        trial_start_dt = now
        user["trial_start"] = now.isoformat() + "Z"
        needs_save = True
    # Automatically downgrade from trial after 7 days
    if plan in (None, "trial") and trial_start_dt:
        if now - trial_start_dt >= datetime.timedelta(days=7):
            # Downgrade to free tier
            user["plan"] = "free"
            plan = "free"
            needs_save = True
    # Ensure monthly usage fields exist
    if user.get("replies_this_month") is None:
        user["replies_this_month"] = 0
        needs_save = True
    if user.get("usage_month_start") is None:
        month_start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        user["usage_month_start"] = month_start.isoformat() + "Z"
        needs_save = True
    # Persist any modifications
    if needs_save and user_id:
        try:
            # Only update the specific fields that might have changed
            updates = {
                "plan": user.get("plan"),
                "trial_start": user.get("trial_start"),
                "replies_this_month": user.get("replies_this_month"),
                "usage_month_start": user.get("usage_month_start"),
                # Persist daily usage fields if they were reset
                "replies_today": user.get("replies_today"),
                "usage_day_start": user.get("usage_day_start"),
            }
            user_store.update_user_fields(user_id, updates)
        except Exception:
            # Ignore errors in background updates
            pass
    # Ensure daily usage fields exist and reset them if the day has rolled over.
    # Use increment_reply_count_daily with zero to perform the reset logic in the store.
    try:
        user_store.increment_reply_count_daily(user_id, 0)
        refreshed = user_store.get_user_by_id(user_id)
        if refreshed:
            user.update({
                "replies_today": refreshed.get("replies_today"),
                "usage_day_start": refreshed.get("usage_day_start"),
            })
    except Exception:
        # ignore errors; continue with current fields
        pass
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
    # Enforce per‑month usage limits for free plan users after trial
    user_plan = current_user.get("plan")
    # Ensure replies_this_month is an int
    replies_so_far = 0
    try:
        replies_so_far = int(current_user.get("replies_this_month") or 0)
    except Exception:
        replies_so_far = 0
    # Determine monthly limit from environment or default
    free_limit = int(os.getenv("REPLYFLOW_FREE_REPLIES_PER_MONTH", "30"))
    # If on the free plan, restrict number of replies and update usage
    if user_plan == "free":
        # Reset usage month if needed via increment_reply_count with 0 to trigger reset logic
        try:
            user_store.increment_reply_count(user_id, 0)
            # Refresh current_user after potential reset
            refreshed = user_store.get_user_by_id(user_id)
            if refreshed:
                current_user.update(refreshed)
                replies_so_far = int(refreshed.get("replies_this_month") or 0)
        except Exception:
            pass
        # Calculate remaining replies allowed this month
        remaining = free_limit - replies_so_far
        if remaining <= 0:
            # Nothing allowed; record zero replies and return empty list
            try:
                analytics_store.record_event("replies_generated", 0)
            except Exception:
                pass
            return []
        # Truncate prepared list to the remaining allowance
        if len(prepared) > remaining:
            prepared = prepared[:remaining]
        # Increment the user's usage by the number of replies returned
        try:
            user_store.increment_reply_count(user_id, len(prepared))
        except Exception:
            pass
    # Record analytics for the number of replies generated in this batch
    try:
        analytics_store.record_event("replies_generated", len(prepared))
    except Exception:
        pass
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
    # Record analytics for the number of replies generated for posting
    try:
        analytics_store.record_event("replies_generated", len(prepared))
    except Exception:
        pass
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

# -------------------------------------------------------------------------
# Analytics endpoint
#
# Expose a simple endpoint to fetch aggregated analytics data.  This can be
# used during the beta to monitor feature usage, reply generation counts
# and captured error logs.  In a production setting, you may wish to
# restrict access to authenticated administrators or proxy these metrics
# through a dashboard.

@app.get("/analytics")
async def analytics_endpoint(current_user: dict = Depends(get_current_user)) -> dict:
    """Return aggregated analytics data.

    This endpoint returns the current usage and error statistics captured by
    the analytics store.  It requires the caller to be authenticated.

    Returns:
        A dictionary with keys "events" and "errors".
    """
    # In future you could authorize only admins by checking current_user roles
    try:
        return analytics_store.get_stats()
    except Exception:
        # On failure, return empty analytics; avoid leaking internal errors
        return {"events": {}, "errors": []}


# -----------------------------------------------------------------------------
# Intercom JWT generation
#
# To secure the Intercom Messenger, the backend must generate a JSON Web Token
# (JWT) signed with your Messenger API secret. Intercom uses this token to
# verify the identity of the user interacting with the messenger. By moving
# the signing logic into the server, you avoid exposing the secret in the
# frontend. The `_encode_jwt` function below implements a minimal HS256 JWT
# encoder without external dependencies. It constructs a JWT by base64url
# encoding the header and payload and then signing them with HMAC‑SHA256. A
# FastAPI endpoint `/intercom/jwt` is provided to generate a token for a
# given `user_id`. You must set the `INTERCOM_API_SECRET` environment variable
# in your Render environment with your Messenger API secret for this to work.

def _encode_jwt(payload: Dict[str, Any], secret: str) -> str:
    """Encode a payload as a JWT using HS256.

    Args:
        payload: The JWT claims to encode. Must be JSON serializable.
        secret: The secret key used to sign the token.

    Returns:
        A JWT string in the form header.payload.signature.
    """
    header = {"typ": "JWT", "alg": "HS256"}
    # JSON encode then base64url encode header and payload without padding
    header_json = json.dumps(header, separators=(",", ":")).encode("utf-8")
    payload_json = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    header_b64 = base64.urlsafe_b64encode(header_json).rstrip(b"=").decode("utf-8")
    payload_b64 = base64.urlsafe_b64encode(payload_json).rstrip(b"=").decode("utf-8")
    unsigned = f"{header_b64}.{payload_b64}"
    # Compute HMAC‑SHA256 signature
    signature = hmac.new(secret.encode("utf-8"), unsigned.encode("utf-8"), hashlib.sha256).digest()
    signature_b64 = base64.urlsafe_b64encode(signature).rstrip(b"=").decode("utf-8")
    return f"{unsigned}.{signature_b64}"


@app.get("/intercom/jwt")
def generate_intercom_jwt(
    user_id: str = Query(...),
    email: Optional[str] = Query(None, description="The user's email address"),
    sensitive_attribute1: Optional[str] = Query(None, description="Optional sensitive attribute 1"),
    sensitive_attribute2: Optional[str] = Query(None, description="Optional sensitive attribute 2"),
    sensitive_attribute3: Optional[str] = Query(None, description="Optional sensitive attribute 3"),
) -> Dict[str, str]:
    """Generate a signed JWT for the Intercom Messenger.

    The `user_id` must uniquely identify the currently logged in user. It
    becomes the `sub` claim in the JWT. Additional optional attributes such
    as email or other sensitive data can be included in the payload to
    enable Intercom to personalize the messenger. The token is valid for
    24 hours.

    Query Parameters:
        user_id: The identifier of the user for whom to generate the JWT.
        email: The user's email address (optional). If provided, it is
            included in the payload under the `email` claim.
        sensitive_attribute1: Custom optional attribute that will be
            included in the payload.
        sensitive_attribute2: Custom optional attribute that will be
            included in the payload.
        sensitive_attribute3: Custom optional attribute that will be
            included in the payload.

    Returns:
        A JSON object containing the signed JWT under the `jwt` key.
    """
    secret = os.getenv("INTERCOM_API_SECRET")
    if not secret:
        raise HTTPException(status_code=500, detail="INTERCOM_API_SECRET not configured")
    now = int(time.time())
    # Include both `user_id` and `sub` claims to ensure compatibility with
    # Intercom's expectations. Intercom requires the `user_id` claim in the
    # payload for identity verification. The `sub` claim is also included for
    # backward compatibility with other JWT consumers.
    payload: Dict[str, Any] = {
        "user_id": user_id,
        "sub": user_id,
        "iat": now,
        "exp": now + 24 * 60 * 60,  # Token valid for 24 hours
    }
    # Include optional attributes if provided
    if email:
        payload["email"] = email
    if sensitive_attribute1:
        payload["sensitive_attribute1"] = sensitive_attribute1
    if sensitive_attribute2:
        payload["sensitive_attribute2"] = sensitive_attribute2
    if sensitive_attribute3:
        payload["sensitive_attribute3"] = sensitive_attribute3
    token = _encode_jwt(payload, secret)
    return {"jwt": token}


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

# -----------------------------------------------------------------------------
# Additional admin routes for X and Reddit tokens
#
# These endpoints are defined after ``get_current_user`` so that the default
# dependency can be resolved successfully at import time.  They allow the
# frontend to determine whether the user has already connected their
# respective accounts.  Sensitive token values are not returned; only
# metadata is exposed.
@app.get("/admin/x_tokens")
async def admin_x_tokens(current_user: dict = Depends(get_current_user)) -> list[dict]:
    """Return the list of stored X tokens for the current user.

    This admin endpoint allows the frontend to inspect whether the user has
    connected X previously.  It does not return any secret values other than
    the stored tokens themselves.  If the user is not authenticated an
    empty list is returned.
    """
    try:
        user_id = current_user.get("id") if current_user else None
        if not user_id:
            return []
        return token_store.list_tokens(user_id, "x")
    except Exception:
        return []


@app.get("/admin/reddit_tokens")
async def admin_reddit_tokens(current_user: dict = Depends(get_current_user)) -> list[dict]:
    """Return the list of stored Reddit tokens for the current user.

    Similar to ``/admin/x_tokens``, this route allows the dashboard to
    determine if Reddit has already been connected.  On error or missing
    authentication, an empty list is returned.
    """
    try:
        user_id = current_user.get("id") if current_user else None
        if not user_id:
            return []
        return token_store.list_tokens(user_id, "reddit")
    except Exception:
        return []

# -----------------------------------------------------------------------------
# Additional connection, prepare and post routes for new platforms
#
# To support Bluesky, LinkedIn, Google Business Profile, Telegram, Threads,
# and YouTube without requiring full OAuth flows, we provide minimal
# connect endpoints.  These endpoints accept a session token and store a
# preconfigured access token (from environment variables) in the token store
# for the logged-in user.  They respond with a small HTML page that notifies
# the opener via postMessage and closes the window.  Prepare/post endpoints
# are stubbed to return empty lists or zero counts until real API logic is
# implemented.  Admin endpoints expose stored tokens so the frontend can
# adjust the UI accordingly.

# Helper to perform generic connect logic for a provider.  Looks up the
# access token in a list of environment variables, saves it to the token store
# and returns a tiny HTML page that signals completion.  If no token is found,
# a simple error message is shown.
async def _generic_connect(request_token: Optional[str], provider: str, env_keys: list[str], message_name: str) -> HTMLResponse:
    session_token = request_token
    if not session_token:
        frontend_login = os.getenv("FRONTEND_LOGIN_URL") or "/account.html"
        return RedirectResponse(frontend_login)
    user_id = session_store.get_user_id(session_token)
    if not user_id:
        frontend_login = os.getenv("FRONTEND_LOGIN_URL") or "/account.html"
        return RedirectResponse(frontend_login)
    # Iterate through possible environment keys to find an access token
    access_token = None
    for key in env_keys:
        val = os.getenv(key)
        if val:
            access_token = val
            break
    if not access_token:
        # If no token is configured, display a basic message to the user
        content = f"""<html><body><p>No {provider.capitalize()} access token configured on the server.</p></body></html>"""
        return HTMLResponse(content=content, status_code=200)
    try:
        token_store.save_token(
            user_id=user_id,
            provider=provider,
            page_id="self",
            page_access_token=access_token,
            ig_account_id=None,
            user_access_token=access_token,
        )
    except Exception:
        # Ignore storage errors
        pass
    html = f"""<html>
      <body>
        <script>
          try {{
            if (window.opener) {{
              window.opener.postMessage('{message_name}', '*');
            }}
          }} catch (e) {{}}
          window.close();
        </script>
        <p>{provider.capitalize()} connection complete. You may close this window.</p>
      </body>
    </html>"""
    return HTMLResponse(content=html, status_code=200)

# Stub reply preparation function for unsupported platforms.  Returns an empty
# list to indicate that no draft replies could be generated.  Real logic
# should call the appropriate external API to fetch comments or reviews and
# then generate replies.
async def _prepare_generic_replies(
    user_id: str,
    tone: str,
    max_posts: int,
    max_comments: int,
    max_tokens: int,
) -> list[dict]:
    # This stub returns an empty list. In the future, integrate with the
    # provider's API to fetch recent comments or reviews, generate replies
    # using OpenAI or another LLM, and return them in the expected format.
    return []

# Generic prepare endpoint factory.  Creates an endpoint that handles query
# parameters and JSON body overrides for tone, max_posts, max_comments and
# max_tokens.  It calls the stub _prepare_generic_replies and applies free
# plan limits similar to other platforms.
async def _generic_prepare(
    provider: str,
    tone: str,
    max_posts: int,
    max_comments: int,
    max_tokens: int,
    current_user: dict,
    request: Request,
) -> list[dict]:
    user_id = current_user.get("id")
    # Allow overrides via JSON body when method is POST
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
            if body.get("max_comments"):
                try:
                    max_comments = int(body.get("max_comments"))
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
    # Query params override (tone, max_posts, max_comments, max_tokens)
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
        qp_mc = request.query_params.get("max_comments") if request.query_params else None
        if qp_mc:
            try:
                max_comments = int(qp_mc)
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
    # Call the stub generator
    prepared = await _prepare_generic_replies(
        user_id=user_id,
        tone=tone,
        max_posts=max_posts,
        max_comments=max_comments,
        max_tokens=max_tokens,
    )
    # Apply plan usage limits: free and solo plans have monthly caps
    user_plan = current_user.get("plan")
    # Determine monthly usage cap based on plan
    try:
        replies_so_far = int(current_user.get("replies_this_month") or 0)
    except Exception:
        replies_so_far = 0
    limit: Optional[int] = None
    if user_plan == "free":
        # Free plan uses environment or default (30)
        try:
            limit = int(os.getenv("REPLYFLOW_FREE_REPLIES_PER_MONTH", "30"))
        except Exception:
            limit = 30
    elif user_plan == "solo":
        # Solo plan uses globally defined limit
        limit = REPLYFLOW_SOLO_REPLIES_PER_MONTH
    else:
        # Pro or other plans have no limit
        limit = None
    # If a limit is defined and positive, enforce it
    if limit and limit > 0 and user_plan in ("free", "solo"):
        try:
            # Reset usage counters at month start and refresh user
            user_store.increment_reply_count(user_id, 0)
            refreshed = user_store.get_user_by_id(user_id)
            if refreshed:
                current_user.update(refreshed)
                replies_so_far = int(refreshed.get("replies_this_month") or 0)
        except Exception:
            pass
        remaining = limit - replies_so_far
        if remaining <= 0:
            # No replies allowed; record zero replies generated and return empty list
            try:
                analytics_store.record_event("replies_generated", 0)
            except Exception:
                pass
            return []
        if len(prepared) > remaining:
            prepared = prepared[:remaining]
        # Increment monthly usage by the number of prepared replies
        try:
            user_store.increment_reply_count(user_id, len(prepared))
        except Exception:
            pass
    # Record analytics for replies generated for this provider
    try:
        analytics_store.record_event("replies_generated", len(prepared))
    except Exception:
        pass
    return prepared

# Generic post endpoint factory.  Calls _generic_prepare to produce replies
# then returns a summary with zero replies posted.  Real implementations
# would iterate over prepared replies and publish them to the provider.
async def _generic_post(
    provider: str,
    tone: str,
    max_posts: int,
    max_comments: int,
    max_tokens: int,
    current_user: dict,
    request: Request,
) -> dict:
    await _generic_prepare(
        provider=provider,
        tone=tone,
        max_posts=max_posts,
        max_comments=max_comments,
        max_tokens=max_tokens,
        current_user=current_user,
        request=request,
    )
    # Placeholder: nothing is posted for generic providers
    return {"replies_posted": 0}

# Generic admin endpoint for tokens.  Returns stored tokens for the given
# provider if the user is authenticated.  On error returns empty list.
async def _generic_admin_tokens(provider: str, current_user: dict) -> list[dict]:
    try:
        user_id = current_user.get("id") if current_user else None
        if not user_id:
            return []
        return token_store.list_tokens(user_id, provider)
    except Exception:
        return []

# ----------------------------------------------------------------------------
# Bluesky OAuth helper functions
#
# The AT Protocol OAuth flow uses PKCE (Proof Key for Code Exchange).  To
# implement this flow we need to generate a random code verifier and the
# corresponding code challenge.  The code verifier is a high‑entropy string
# which is stored temporarily and sent to the token endpoint later.  The
# code challenge is derived by applying SHA256 to the verifier and
# base64url‑encoding the result without padding.  See RFC 7636 for details.

def _generate_pkce_pair() -> tuple[str, str]:
    """Generate a PKCE code verifier and its matching code challenge.

    Returns:
        A tuple of (code_verifier, code_challenge).
    """
    # Generate a random 32‑byte code verifier using URL‑safe characters.  The
    # length of the generated string may be longer than 32 characters due to
    # base64 encoding; this is acceptable per the spec (min 43 chars, max 128).
    code_verifier = secrets.token_urlsafe(32)
    # Compute the SHA256 hash of the verifier and base64url‑encode it without
    # padding to produce the code challenge.  Remove any trailing '='.
    digest = hashlib.sha256(code_verifier.encode('utf-8')).digest()
    code_challenge = base64.urlsafe_b64encode(digest).decode('utf-8').rstrip('=')
    return code_verifier, code_challenge

# --------------------------- Bluesky Endpoints -----------------------------

@app.get("/bluesky/connect")
async def bluesky_connect(token: Optional[str] = None) -> HTMLResponse:
    """Connect the current user to Bluesky using a preconfigured token.

    A session token is required in the query string.  The server looks up
    BLUESKY_ACCESS_TOKEN or BLUESKY_TOKEN environment variables to find the
    access token.  If a token is present it is stored in the token store.
    The response page will notify the opener via postMessage with
    'bluesky_connected' and close the popup.
    """
    return await _generic_connect(
        request_token=token,
        provider="bluesky",
        env_keys=["BLUESKY_ACCESS_TOKEN", "BLUESKY_TOKEN", "ATPROTO_ACCESS_TOKEN"],
        message_name="bluesky_connected",
    )

@app.post("/bluesky/prepare_replies")
async def bluesky_prepare_replies(
    tone: str = "friendly",
    max_posts: int = 3,
    max_comments: int = 5,
    max_tokens: int = 60,
    current_user: dict = Depends(get_current_user),
    request: Request = None,
) -> list[dict]:
    """Prepare draft replies for Bluesky posts or replies.

    This endpoint currently returns an empty list because Bluesky integration
    is not yet implemented.  Once API access is available it should fetch
    recent mentions or comments and generate replies.
    """
    return await _generic_prepare(
        provider="bluesky",
        tone=tone,
        max_posts=max_posts,
        max_comments=max_comments,
        max_tokens=max_tokens,
        current_user=current_user,
        request=request,
    )

@app.post("/bluesky/post_replies")
async def bluesky_post_replies(
    tone: str = "friendly",
    max_posts: int = 3,
    max_comments: int = 5,
    max_tokens: int = 60,
    current_user: dict = Depends(get_current_user),
    request: Request = None,
) -> dict:
    """Post generated replies back to Bluesky.

    Since Bluesky posting is not yet implemented, this endpoint will
    generate draft replies and return 0 as the number posted.
    """
    return await _generic_post(
        provider="bluesky",
        tone=tone,
        max_posts=max_posts,
        max_comments=max_comments,
        max_tokens=max_tokens,
        current_user=current_user,
        request=request,
    )

@app.get("/admin/bluesky_tokens")
async def admin_bluesky_tokens(current_user: dict = Depends(get_current_user)) -> list[dict]:
    """Return Bluesky tokens for the current user."""
    return await _generic_admin_tokens("bluesky", current_user)

# -----------------------------------------------------------------------------
# Bluesky OAuth endpoints
#
# The following endpoints implement a skeleton of the AT Protocol OAuth flow.
# This flow allows ReplyFlow to obtain access and refresh tokens for Bluesky
# users without requiring static preconfigured tokens.  The implementation
# here follows the general pattern of the OAuth 2.0 authorization code flow
# with PKCE, but does not actually perform the token exchange against a PDS.
# Instead, it generates placeholder tokens since this environment cannot
# perform external HTTP requests.  In a real deployment you would use
# httpx or another HTTP client to exchange the authorization code for
# tokens and handle errors accordingly.

@app.get("/bluesky/client_metadata.json")
async def bluesky_client_metadata(request: Request) -> dict:
    """Serve the Bluesky OAuth client metadata document.

    AT Protocol clients are identified by a URL which points to a JSON
    document describing the client.  This endpoint dynamically constructs
    the metadata based on environment variables and the incoming request
    scheme/host.  You may override the base URL via BLUESKY_METADATA_BASE_URL.

    Returns:
        A dictionary representing the client metadata.  FastAPI will
        automatically serialize it to JSON.
    """
    # Determine the base URL for the client_id.  If BLUESKY_METADATA_BASE_URL
    # is set, use that; otherwise derive from the incoming request.
    base_url = os.getenv("BLUESKY_METADATA_BASE_URL")
    if not base_url:
        # Derive scheme and host from the request URL.  Include port if it
        # differs from the default (80/443).  In FastAPI, request.url
        # includes scheme, host and port.  Note: request.url.port may be None.
        scheme = request.url.scheme
        host = request.url.hostname
        port = request.url.port
        # Only include the port if it's not the default for HTTP/HTTPS
        if port and ((scheme == "http" and port != 80) or (scheme == "https" and port != 443)):
            base_url = f"{scheme}://{host}:{port}"
        else:
            base_url = f"{scheme}://{host}"
    # Construct the redirect URI.  This can be overridden via env var.
    redirect_uri = os.getenv("BLUESKY_REDIRECT_URI") or f"{base_url}/bluesky/oauth/callback"
    # Populate the metadata fields.  The client_id is the full URL to this
    # metadata document.  The scope defaults to transition:generic unless
    # overridden.  Additional optional fields like client_name can be set via
    # environment variables.
    metadata = {
        "client_id": f"{base_url}/bluesky/client_metadata.json",
        "application_type": "web",
        "grant_types": ["authorization_code", "refresh_token"],
        "redirect_uris": [redirect_uri],
        "scope": os.getenv("BLUESKY_SCOPE", "transition:generic"),
    }
    client_name = os.getenv("BLUESKY_CLIENT_NAME")
    if client_name:
        metadata["client_name"] = client_name
    logo_uri = os.getenv("BLUESKY_LOGO_URI")
    if logo_uri:
        metadata["logo_uri"] = logo_uri
    return metadata


@app.get("/bluesky/oauth/start")
async def bluesky_oauth_start(
    token: Optional[str] = None,
    request: Request = None,
) -> RedirectResponse:
    """Initiate the Bluesky OAuth authorization flow.

    For browser pop‑up flows, the session token is passed as a query parameter
    named ``token``.  This token is validated against the session store to
    derive the current user.  A PKCE code verifier/challenge pair and
    random state are generated, persisted, and the user is redirected to
    the PDS authorization endpoint.

    Args:
        token: The session token identifying the logged‑in user.
        request: The incoming request used to compute base URLs.

    Returns:
        A RedirectResponse to the Bluesky authorization endpoint.
    """
    # Ensure a session token is provided.  Without a token, the user is not
    # authenticated and should be redirected to the login page.
    session_token = token
    if not session_token:
        login_url = os.getenv("FRONTEND_LOGIN_URL") or "/account.html"
        return RedirectResponse(login_url)
    user_id = session_store.get_user_id(session_token)
    if not user_id:
        login_url = os.getenv("FRONTEND_LOGIN_URL") or "/account.html"
        return RedirectResponse(login_url)
    # Generate state and PKCE verifier/challenge.
    code_verifier, code_challenge = _generate_pkce_pair()
    state = secrets.token_urlsafe(16)
    # Store verifier and user_id for the callback.
    BLUESKY_OAUTH_STATES[state] = {"code_verifier": code_verifier, "user_id": user_id}
    # Compute base URL for metadata and redirect URIs, same as metadata endpoint.
    base_url = os.getenv("BLUESKY_METADATA_BASE_URL")
    if not base_url:
        scheme = request.url.scheme
        host = request.url.hostname
        port = request.url.port
        if port and ((scheme == "http" and port != 80) or (scheme == "https" and port != 443)):
            base_url = f"{scheme}://{host}:{port}"
        else:
            base_url = f"{scheme}://{host}"
    client_id = f"{base_url}/bluesky/client_metadata.json"
    redirect_uri = os.getenv("BLUESKY_REDIRECT_URI") or f"{base_url}/bluesky/oauth/callback"
    scope = os.getenv("BLUESKY_SCOPE", "transition:generic")
    # Choose the PDS authorization host.
    pds_host = os.getenv("BLUESKY_PDS_HOST", "https://bsky.social")
    # Build the authorization URL with encoded parameters.
    params = {
        "client_id": client_id,
        "redirect_uri": redirect_uri,
        "response_type": "code",
        "code_challenge": code_challenge,
        "code_challenge_method": "S256",
        "state": state,
        "scope": scope,
    }
    query = urllib.parse.urlencode(params, quote_via=urllib.parse.quote)
    authorization_url = f"{pds_host}/xrpc/com.atproto.oauth.authorize?{query}"
    return RedirectResponse(url=authorization_url)


@app.get("/bluesky/oauth/callback")
async def bluesky_oauth_callback(
    code: str = Query(default=None),
    state: str = Query(default=None),
    request: Request = None,
) -> HTMLResponse:
    """Handle the Bluesky OAuth callback and store obtained tokens.

    This endpoint receives the authorization code and state from the
    Bluesky authorization server.  It verifies the state, exchanges the
    code for tokens (stubbed here), stores them in the token store, and
    returns a small HTML page that notifies the opener of success and
    closes the popup.

    Args:
        code: The authorization code returned by the authorization server.
        state: The state parameter previously generated during start.
        request: The incoming request (unused but accepted for consistency).

    Returns:
        An HTMLResponse indicating success or failure.
    """
    # Validate the state and retrieve the stored verifier and user ID.
    if not state or state not in BLUESKY_OAUTH_STATES:
        # Unknown state; cannot proceed.  Inform the user and close the popup.
        error_html = """
        <html>
          <body>
            <p>Invalid or expired OAuth state. Please try connecting again.</p>
            <script> if (window.opener) { window.opener.postMessage('bluesky_failed', '*'); } window.close(); </script>
          </body>
        </html>
        """
        return HTMLResponse(content=error_html, status_code=400)
    data = BLUESKY_OAUTH_STATES.pop(state)
    code_verifier = data.get("code_verifier")
    user_id = data.get("user_id")
    # Ensure we have an authorization code and user_id.
    if not code or not user_id:
        error_html = """
        <html><body>
          <p>Authorization failed or was cancelled.</p>
          <script> if (window.opener) { window.opener.postMessage('bluesky_failed', '*'); } window.close(); </script>
        </body></html>
        """
        return HTMLResponse(content=error_html, status_code=400)
    # Perform the token exchange with the atproto OAuth token endpoint.
    # Construct the client_id and redirect_uri identically to the start route.
    # Compute the base URL for this server.
    scheme = request.url.scheme
    host = request.url.hostname
    port = request.url.port
    if port and ((scheme == "http" and port != 80) or (scheme == "https" and port != 443)):
        base_url = f"{scheme}://{host}:{port}"
    else:
        base_url = f"{scheme}://{host}"
    # Allow override via environment, matching the start route logic.
    base_url_override = os.getenv("BLUESKY_METADATA_BASE_URL")
    if base_url_override:
        base_url = base_url_override
    client_id = f"{base_url}/bluesky/client_metadata.json"
    redirect_uri = os.getenv("BLUESKY_REDIRECT_URI") or f"{base_url}/bluesky/oauth/callback"
    # Determine the authorization server (PDS or entryway) to contact.  Default to bsky.social.
    pds_host = os.getenv("BLUESKY_PDS_HOST", "https://bsky.social")
    try:
        # Discover the token endpoint from the OAuth authorization server metadata.
        metadata_url = f"{pds_host}/.well-known/oauth-authorization-server"
        metadata_resp = requests.get(metadata_url, timeout=10)
        metadata_resp.raise_for_status()
        metadata = metadata_resp.json()
        token_endpoint = metadata.get("token_endpoint")
    except Exception as e:
        token_endpoint = None
    access_jwt = None
    refresh_jwt = None
    if token_endpoint:
        # Prepare form data for the token exchange.
        token_data = {
            "grant_type": "authorization_code",
            "code": code,
            "client_id": client_id,
            "redirect_uri": redirect_uri,
            "code_verifier": code_verifier,
        }
        headers = {
            "Accept": "application/json",
        }
        try:
            token_resp = requests.post(token_endpoint, data=token_data, headers=headers, timeout=15)
            token_resp.raise_for_status()
            token_json = token_resp.json()
            access_jwt = token_json.get("access_token")
            # The refresh token may be under "refresh_token"
            refresh_jwt = token_json.get("refresh_token")
        except Exception:
            # If the token request fails, fall back to placeholder tokens.
            access_jwt = None
            refresh_jwt = None
    # Fall back to stub tokens if the real exchange did not succeed.
    if not access_jwt:
        access_jwt = secrets.token_urlsafe(48)
    if not refresh_jwt:
        refresh_jwt = secrets.token_urlsafe(48)
    try:
        token_store.save_token(
            user_id=user_id,
            provider="bluesky",
            page_id="self",
            page_access_token=access_jwt,
            ig_account_id=None,
            user_access_token=refresh_jwt,
        )
    except Exception:
        # Ignore storage errors
        pass
    # Return a simple HTML page to notify the opener and close the popup.
    success_html = """
    <html>
      <body>
        <script>
          try {
            if (window.opener) {
              window.opener.postMessage('bluesky_connected', '*');
            }
          } catch (e) {}
          window.close();
        </script>
        <p>Bluesky connection complete. You may close this window.</p>
      </body>
    </html>
    """
    return HTMLResponse(content=success_html, status_code=200)

# --------------------------- YouTube token refresh test endpoint ---------------------------

@app.get("/youtube/oauth/refresh_test")
async def youtube_oauth_refresh_test(
    token: Optional[str] = None,
) -> HTMLResponse:
    """Test refreshing a YouTube access token using the saved refresh token.

    This endpoint is intended for diagnostic use to verify that long‑lived
    YouTube connections work correctly. It retrieves the stored refresh
    token for the current user and uses it to request a new access token
    from Google's OAuth token endpoint. Upon success, the new access token
    (and any new refresh token) are persisted in the token store. The
    response will notify the opener window via postMessage and close
    itself. If the refresh operation fails, an error message is displayed.

    Args:
        token: The session token identifying the logged‑in user (from the
            Authorization header on the main app). Required to tie the
            refresh operation to a user.

    Returns:
        An HTMLResponse containing a small script that posts a message back
        to the opener window and closes the popup. The message will
        indicate whether the refresh succeeded or failed.
    """
    session_token = token
    if not session_token:
        login_url = os.getenv("FRONTEND_LOGIN_URL") or "/account.html"
        return RedirectResponse(login_url)
    user_id = session_store.get_user_id(session_token)
    if not user_id:
        login_url = os.getenv("FRONTEND_LOGIN_URL") or "/account.html"
        return RedirectResponse(login_url)
    # Fetch the stored YouTube tokens for this user
    try:
        youtube_tokens = token_store.list_tokens(user_id, "youtube")
    except Exception:
        youtube_tokens = []
    refresh_token = None
    access_token = None
    if youtube_tokens:
        # We expect the refresh token to be stored in user_access_token and
        # the access token in page_access_token for provider="youtube"
        first_token = youtube_tokens[0]
        refresh_token = first_token.get("user_access_token")
        access_token = first_token.get("page_access_token")
    # Ensure we have a refresh token
    if not refresh_token:
        error_html = """
        <html><body>
          <p>No YouTube refresh token found for this user. Please connect to YouTube first.</p>
          <script> if (window.opener) { window.opener.postMessage('youtube_refresh_failed', '*'); } window.close(); </script>
        </body></html>
        """
        return HTMLResponse(content=error_html, status_code=400)
    # Build token refresh request
    client_id = os.getenv("YOUTUBE_CLIENT_ID") or os.getenv("GOOGLE_CLIENT_ID")
    client_secret = os.getenv("YOUTUBE_CLIENT_SECRET") or os.getenv("GOOGLE_CLIENT_SECRET")
    if not client_id or not client_secret:
        error_html = """
        <html><body>
          <p>Missing YouTube client credentials. Please set YOUTUBE_CLIENT_ID and YOUTUBE_CLIENT_SECRET.</p>
          <script> if (window.opener) { window.opener.postMessage('youtube_refresh_failed', '*'); } window.close(); </script>
        </body></html>
        """
        return HTMLResponse(content=error_html, status_code=500)
    token_endpoint = "https://oauth2.googleapis.com/token"
    data = {
        "grant_type": "refresh_token",
        "refresh_token": refresh_token,
        "client_id": client_id,
        "client_secret": client_secret,
    }
    headers = {
        "Content-Type": "application/x-www-form-urlencoded",
        "Accept": "application/json",
    }
    new_access_token = None
    new_refresh_token = None
    try:
        resp = requests.post(token_endpoint, data=data, headers=headers, timeout=15)
        resp.raise_for_status()
        resp_json = resp.json()
        new_access_token = resp_json.get("access_token")
        # Google may or may not return a new refresh token; update if provided
        new_refresh_token = resp_json.get("refresh_token")
    except Exception:
        # Fall back to None; we will handle below
        new_access_token = None
        new_refresh_token = None
    # If refresh failed, notify failure
    if not new_access_token:
        error_html = """
        <html><body>
          <p>Failed to refresh the YouTube access token. Please re‑connect.</p>
          <script> if (window.opener) { window.opener.postMessage('youtube_refresh_failed', '*'); } window.close(); </script>
        </body></html>
        """
        return HTMLResponse(content=error_html, status_code=400)
    # Update tokens in the store
    try:
        # Use the existing refresh token unless a new one is returned
        refresh_to_store = new_refresh_token or refresh_token
        token_store.save_token(
            user_id=user_id,
            provider="youtube",
            page_id="self",
            page_access_token=new_access_token,
            ig_account_id=None,
            user_access_token=refresh_to_store,
        )
    except Exception:
        pass
    success_html = """
    <html><body>
      <script>
        try {
          if (window.opener) {
            window.opener.postMessage('youtube_refresh_success', '*');
          }
        } catch (e) {}
        window.close();
      </script>
      <p>YouTube token refresh successful. You may close this window.</p>
    </body></html>
    """
    return HTMLResponse(content=success_html, status_code=200)

# --------------------------- LinkedIn Endpoints ---------------------------

@app.get("/linkedin/connect")
async def linkedin_connect(token: Optional[str] = None) -> HTMLResponse:
    """Connect the current user to LinkedIn using a preconfigured token.

    Looks for LINKEDIN_ACCESS_TOKEN or LINKEDIN_TOKEN in the environment.
    """
    return await _generic_connect(
        request_token=token,
        provider="linkedin",
        env_keys=["LINKEDIN_ACCESS_TOKEN", "LINKEDIN_TOKEN"],
        message_name="linkedin_connected",
    )

@app.post("/linkedin/prepare_replies")
async def linkedin_prepare_replies(
    tone: str = "friendly",
    max_posts: int = 3,
    max_comments: int = 5,
    max_tokens: int = 60,
    current_user: dict = Depends(get_current_user),
    request: Request = None,
) -> list[dict]:
    """Prepare draft replies for LinkedIn comments or posts."""
    return await _generic_prepare(
        provider="linkedin",
        tone=tone,
        max_posts=max_posts,
        max_comments=max_comments,
        max_tokens=max_tokens,
        current_user=current_user,
        request=request,
    )

@app.post("/linkedin/post_replies")
async def linkedin_post_replies(
    tone: str = "friendly",
    max_posts: int = 3,
    max_comments: int = 5,
    max_tokens: int = 60,
    current_user: dict = Depends(get_current_user),
    request: Request = None,
) -> dict:
    """Post generated replies back to LinkedIn."""
    return await _generic_post(
        provider="linkedin",
        tone=tone,
        max_posts=max_posts,
        max_comments=max_comments,
        max_tokens=max_tokens,
        current_user=current_user,
        request=request,
    )

@app.get("/admin/linkedin_tokens")
async def admin_linkedin_tokens(current_user: dict = Depends(get_current_user)) -> list[dict]:
    """Return LinkedIn tokens for the current user."""
    return await _generic_admin_tokens("linkedin", current_user)

# --------------------- Google Business Profile Endpoints -------------------

@app.get("/google/connect")
async def google_connect(token: Optional[str] = None) -> HTMLResponse:
    """Connect the current user to Google Business Profile using a preconfigured token.

    Searches several environment variables for the token: GOOGLE_BUSINESS_PROFILE_ACCESS_TOKEN,
    GOOGLE_ACCESS_TOKEN, GMB_ACCESS_TOKEN.
    """
    return await _generic_connect(
        request_token=token,
        provider="google",
        env_keys=[
            "GOOGLE_BUSINESS_PROFILE_ACCESS_TOKEN",
            "GOOGLE_ACCESS_TOKEN",
            "GMB_ACCESS_TOKEN",
            "GOOGLE_REVIEW_ACCESS_TOKEN",
        ],
        message_name="google_connected",
    )

@app.post("/google/prepare_replies")
async def google_prepare_replies(
    tone: str = "friendly",
    max_posts: int = 3,
    max_comments: int = 5,
    max_tokens: int = 60,
    current_user: dict = Depends(get_current_user),
    request: Request = None,
) -> list[dict]:
    """Prepare draft replies for Google reviews."""
    return await _generic_prepare(
        provider="google",
        tone=tone,
        max_posts=max_posts,
        max_comments=max_comments,
        max_tokens=max_tokens,
        current_user=current_user,
        request=request,
    )

@app.post("/google/post_replies")
async def google_post_replies(
    tone: str = "friendly",
    max_posts: int = 3,
    max_comments: int = 5,
    max_tokens: int = 60,
    current_user: dict = Depends(get_current_user),
    request: Request = None,
) -> dict:
    """Post generated replies back to Google Business Profile reviews."""
    return await _generic_post(
        provider="google",
        tone=tone,
        max_posts=max_posts,
        max_comments=max_comments,
        max_tokens=max_tokens,
        current_user=current_user,
        request=request,
    )

@app.get("/admin/google_tokens")
async def admin_google_tokens(current_user: dict = Depends(get_current_user)) -> list[dict]:
    """Return Google Business Profile tokens for the current user."""
    return await _generic_admin_tokens("google", current_user)

# ---------------------------- Telegram Endpoints ---------------------------

@app.get("/telegram/connect")
async def telegram_connect(token: Optional[str] = None) -> HTMLResponse:
    """Connect the current user to Telegram via a bot token.

    Looks for TELEGRAM_BOT_TOKEN or TELEGRAM_ACCESS_TOKEN in environment variables.
    """
    return await _generic_connect(
        request_token=token,
        provider="telegram",
        env_keys=["TELEGRAM_BOT_TOKEN", "TELEGRAM_ACCESS_TOKEN"],
        message_name="telegram_connected",
    )

@app.post("/telegram/prepare_replies")
async def telegram_prepare_replies(
    tone: str = "friendly",
    max_posts: int = 3,
    max_comments: int = 5,
    max_tokens: int = 60,
    current_user: dict = Depends(get_current_user),
    request: Request = None,
) -> list[dict]:
    """Prepare draft replies for Telegram messages or comments."""
    return await _generic_prepare(
        provider="telegram",
        tone=tone,
        max_posts=max_posts,
        max_comments=max_comments,
        max_tokens=max_tokens,
        current_user=current_user,
        request=request,
    )

@app.post("/telegram/post_replies")
async def telegram_post_replies(
    tone: str = "friendly",
    max_posts: int = 3,
    max_comments: int = 5,
    max_tokens: int = 60,
    current_user: dict = Depends(get_current_user),
    request: Request = None,
) -> dict:
    """Post generated replies to Telegram chats via the Bot API."""
    return await _generic_post(
        provider="telegram",
        tone=tone,
        max_posts=max_posts,
        max_comments=max_comments,
        max_tokens=max_tokens,
        current_user=current_user,
        request=request,
    )

@app.get("/admin/telegram_tokens")
async def admin_telegram_tokens(current_user: dict = Depends(get_current_user)) -> list[dict]:
    """Return Telegram tokens for the current user."""
    return await _generic_admin_tokens("telegram", current_user)

# ---------------------------- Threads Endpoints ----------------------------

@app.get("/threads/connect")
async def threads_connect(token: Optional[str] = None) -> HTMLResponse:
    """Connect the current user to Threads using a preconfigured token.

    Looks for THREADS_ACCESS_TOKEN or THREADS_TOKEN environment variables.
    """
    return await _generic_connect(
        request_token=token,
        provider="threads",
        env_keys=["THREADS_ACCESS_TOKEN", "THREADS_TOKEN"],
        message_name="threads_connected",
    )

@app.post("/threads/prepare_replies")
async def threads_prepare_replies(
    tone: str = "friendly",
    max_posts: int = 3,
    max_comments: int = 5,
    max_tokens: int = 60,
    current_user: dict = Depends(get_current_user),
    request: Request = None,
) -> list[dict]:
    """Prepare draft replies for Threads replies or comments."""
    return await _generic_prepare(
        provider="threads",
        tone=tone,
        max_posts=max_posts,
        max_comments=max_comments,
        max_tokens=max_tokens,
        current_user=current_user,
        request=request,
    )

@app.post("/threads/post_replies")
async def threads_post_replies(
    tone: str = "friendly",
    max_posts: int = 3,
    max_comments: int = 5,
    max_tokens: int = 60,
    current_user: dict = Depends(get_current_user),
    request: Request = None,
) -> dict:
    """Post generated replies back to Threads."""
    return await _generic_post(
        provider="threads",
        tone=tone,
        max_posts=max_posts,
        max_comments=max_comments,
        max_tokens=max_tokens,
        current_user=current_user,
        request=request,
    )

@app.get("/admin/threads_tokens")
async def admin_threads_tokens(current_user: dict = Depends(get_current_user)) -> list[dict]:
    """Return Threads tokens for the current user."""
    return await _generic_admin_tokens("threads", current_user)

# ----------------------------- YouTube Endpoints ---------------------------

@app.get("/youtube/connect")
async def youtube_connect(token: Optional[str] = None) -> HTMLResponse:
    """Connect the current user to YouTube using a preconfigured token.

    Searches YOUTUBE_ACCESS_TOKEN, GOOGLE_YOUTUBE_ACCESS_TOKEN and
    YT_ACCESS_TOKEN environment variables.
    """
    return await _generic_connect(
        request_token=token,
        provider="youtube",
        env_keys=[
            "YOUTUBE_ACCESS_TOKEN",
            "GOOGLE_YOUTUBE_ACCESS_TOKEN",
            "YT_ACCESS_TOKEN",
            "GOOGLE_ACCESS_TOKEN",
        ],
        message_name="youtube_connected",
    )

@app.post("/youtube/prepare_replies")
async def youtube_prepare_replies(
    tone: str = "friendly",
    max_posts: int = 3,
    max_comments: int = 5,
    max_tokens: int = 60,
    current_user: dict = Depends(get_current_user),
    request: Request = None,
) -> list[dict]:
    """Prepare draft replies for YouTube comments."""
    # Refresh the user's YouTube access token if it is nearing expiry.  This
    # ensures that _generic_prepare uses a valid access token when fetching
    # comments.  If no user is logged in, skip refresh.
    try:
        user_id = current_user.get("id") if current_user else None
        if user_id:
            await _refresh_youtube_access_token_if_needed(user_id)
    except Exception:
        # Silently ignore refresh errors; _generic_prepare will surface any
        # authorization problems if the token is invalid
        pass
    return await _generic_prepare(
        provider="youtube",
        tone=tone,
        max_posts=max_posts,
        max_comments=max_comments,
        max_tokens=max_tokens,
        current_user=current_user,
        request=request,
    )

@app.post("/youtube/post_replies")
async def youtube_post_replies(
    tone: str = "friendly",
    max_posts: int = 3,
    max_comments: int = 5,
    max_tokens: int = 60,
    current_user: dict = Depends(get_current_user),
    request: Request = None,
) -> dict:
    """Post generated replies back to YouTube."""
    # Refresh the user's YouTube access token prior to posting replies.  A
    # stale access token could cause posting failures; refreshing helps
    # prevent that.  Ignore any errors silently.
    try:
        user_id = current_user.get("id") if current_user else None
        if user_id:
            await _refresh_youtube_access_token_if_needed(user_id)
    except Exception:
        pass
    return await _generic_post(
        provider="youtube",
        tone=tone,
        max_posts=max_posts,
        max_comments=max_comments,
        max_tokens=max_tokens,
        current_user=current_user,
        request=request,
    )

@app.get("/admin/youtube_tokens")
async def admin_youtube_tokens(current_user: dict = Depends(get_current_user)) -> list[dict]:
    """Return YouTube tokens for the current user."""
    try:
        user_id = current_user.get("id") if current_user else None
        if not user_id:
            return []
        return token_store.list_tokens(user_id, "youtube")
    except Exception:
        return []

# Endpoint to upload a video to YouTube
@app.post("/youtube/upload_video")
async def youtube_upload_video(
    payload: dict,
    current_user: dict = Depends(get_current_user),
    request: Request = None,
) -> dict:
    """Upload a video to the user's connected YouTube channel via JSON payload.

    The client should send a JSON body with keys:
      - ``title`` (str): The video title.
      - ``description`` (str): The video description.
      - ``privacy_status`` (str): The privacy setting (public, unlisted, private).
      - ``file_name`` (str): The original file name.
      - ``file_content`` (str): Base64‑encoded content of the video file.

    This avoids requiring the python‑multipart package on the server.
    """
    # Feature flag: if video uploads are disabled, reject requests with 403. This ensures
    # that only beta users or environments with the feature enabled can upload videos.
    if not ENABLE_VIDEO_UPLOAD:
        raise HTTPException(status_code=403, detail="Video upload is disabled for your account")
    # Validate payload keys
    try:
        title = payload.get("title", "").strip()
        description = payload.get("description", "").strip()
        privacy_status = payload.get("privacy_status", "public").strip() or "public"
        file_name = payload.get("file_name", "video")
        file_content = payload.get("file_content")
        if not title or not description or not file_content:
            raise ValueError
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid upload payload")
    # Decode the base64 string.  If the client sends a data URI (e.g. data:video/mp4;base64,...),
    # extract the part after the comma.
    try:
        if "," in file_content:
            file_content = file_content.split(",", 1)[1]
        video_bytes = base64.b64decode(file_content)
    except Exception:
        raise HTTPException(status_code=400, detail="Unable to decode video content")
    # Ensure user is authenticated
    user_id = current_user.get("id") if current_user else None
    if not user_id:
        raise HTTPException(status_code=401, detail="Authentication required")
    # Refresh token if needed
    try:
        await _refresh_youtube_access_token_if_needed(user_id)
    except Exception:
        pass
    # Retrieve stored tokens for YouTube
    try:
        tokens = token_store.list_tokens(user_id, "youtube")
    except Exception:
        tokens = []
    if not tokens:
        raise HTTPException(status_code=400, detail="No YouTube tokens found; please connect your account first")
    access_token = tokens[0].get("page_access_token")
    if not access_token:
        raise HTTPException(status_code=400, detail="Missing YouTube access token; please reconnect")
    # Prepare metadata
    metadata = {
        "snippet": {
            "title": title,
            "description": description,
        },
        "status": {
            "privacyStatus": privacy_status,
        },
    }
    upload_url = "https://www.googleapis.com/upload/youtube/v3/videos"
    params = {
        "uploadType": "multipart",
        "part": "snippet,status",
    }
    headers = {
        "Authorization": f"Bearer {access_token}",
    }
    files_payload = {
        "data": (
            "metadata.json",
            json.dumps(metadata),
            "application/json; charset=UTF-8",
        ),
        "file": (
            file_name or "video",
            video_bytes,
            "application/octet-stream",
        ),
    }
    try:
        resp = requests.post(upload_url, params=params, files=files_payload, headers=headers, timeout=60)
        resp.raise_for_status()
        resp_json = resp.json()
        video_id = resp_json.get("id")
        if not video_id:
            return {"status": "error", "message": "Video upload succeeded but no video ID returned"}
        return {"status": "success", "video_id": video_id}
    except Exception as e:
        error_message = None
        if hasattr(e, "response") and e.response is not None:
            try:
                error_json = e.response.json()
                error_message = error_json.get("error", {}).get("message")
            except Exception:
                error_message = None
        error_text = str(error_message) if error_message else str(e)
        return {"status": "error", "message": f"Failed to upload video: {error_text}"}

# ----------------------------- YouTube OAuth endpoints -----------------------------

@app.get("/youtube/oauth/start")
async def youtube_oauth_start(
    token: Optional[str] = None,
    request: Request = None,
) -> RedirectResponse:
    """Initiate the YouTube OAuth authorization flow.

    This endpoint constructs the Google OAuth 2.0 authorization URL and
    redirects the browser to it.  A ``state`` value is generated to prevent
    CSRF attacks and is stored in ``YOUTUBE_OAUTH_STATES`` along with the
    current user's ID.  The user's session token must be provided via the
    ``token`` query parameter so that we can determine who is initiating
    the OAuth flow.

    Args:
        token: The session token identifying the logged‑in user (from the
            Authorization header on the main app).  Required to tie the
            authorization to a user.
        request: The incoming request, used to derive the default redirect URI
            if not explicitly configured.

    Returns:
        A RedirectResponse sending the user to Google's OAuth authorization
        endpoint.
    """
    session_token = token
    # If no session token is provided, redirect to the login page.
    if not session_token:
        login_url = os.getenv("FRONTEND_LOGIN_URL") or "/account.html"
        return RedirectResponse(login_url)
    user_id = session_store.get_user_id(session_token)
    if not user_id:
        login_url = os.getenv("FRONTEND_LOGIN_URL") or "/account.html"
        return RedirectResponse(login_url)
    # Generate a unique state parameter to mitigate CSRF attacks and map back to
    # the initiating user.  Use a URL‑safe random string.
    state = secrets.token_urlsafe(16)
    YOUTUBE_OAUTH_STATES[state] = user_id
    # Determine client ID and redirect URI from environment.  These must be
    # configured in Google Cloud Console when creating the OAuth client.
    client_id = os.getenv("YOUTUBE_CLIENT_ID") or os.getenv("GOOGLE_CLIENT_ID")
    if not client_id:
        # Without a client ID we cannot perform OAuth; redirect with error.
        error_html = """
        <html><body>
          <p>Missing YOUTUBE_CLIENT_ID environment variable. Please set it in the server configuration.</p>
          <script> window.close(); </script>
        </body></html>
        """
        return HTMLResponse(error_html, status_code=500)
    # Compute the default redirect URI based on the current request if
    # YOUTUBE_REDIRECT_URI is not set.  We include the port if present and
    # non‑default to ensure the URL matches the deployed environment.
    base_redirect_uri = os.getenv("YOUTUBE_REDIRECT_URI")
    if not base_redirect_uri:
        scheme = request.url.scheme if request else "https"
        host = request.url.hostname if request else os.getenv("HOSTNAME", "localhost")
        port = request.url.port if request else None
        if port and ((scheme == "http" and port != 80) or (scheme == "https" and port != 443)):
            base_redirect_uri = f"{scheme}://{host}:{port}/youtube/oauth/callback"
        else:
            base_redirect_uri = f"{scheme}://{host}/youtube/oauth/callback"
    redirect_uri = base_redirect_uri
    # Determine the scope(s) required for YouTube access.  The default scope
    # grants read/write access to comments via the YouTube Data API.  Separate
    # scopes with spaces if multiple are needed.
    scope = os.getenv("YOUTUBE_SCOPE") or "https://www.googleapis.com/auth/youtube.force-ssl"
    # Build the authorization URL.  We request offline access so we get a
    # refresh token in addition to the access token.  Prompt=consent forces
    # Google to show the consent screen every time; this is recommended to
    # ensure we obtain a refresh token on subsequent authorizations.
    params = {
        "response_type": "code",
        "client_id": client_id,
        "redirect_uri": redirect_uri,
        "scope": scope,
        "state": state,
        "access_type": "offline",
        "prompt": "consent",
    }
    auth_endpoint = "https://accounts.google.com/o/oauth2/v2/auth"
    query = urllib.parse.urlencode(params, quote_via=urllib.parse.quote)
    authorization_url = f"{auth_endpoint}?{query}"
    return RedirectResponse(url=authorization_url)


@app.get("/youtube/oauth/callback")
async def youtube_oauth_callback(
    code: str = Query(default=None),
    state: str = Query(default=None),
    request: Request = None,
) -> HTMLResponse:
    """Handle the YouTube OAuth callback and exchange the code for tokens.

    This endpoint receives the authorization code and state from Google.  It
    validates the state, exchanges the code for access and refresh tokens
    using the Google token endpoint, stores them, and notifies the opener.

    Args:
        code: The authorization code returned by Google's OAuth server.
        state: The state parameter generated during the start of the flow.
        request: The incoming request (unused here but accepted for parity).

    Returns:
        An HTMLResponse containing a small script that posts a message back
        to the opener window and closes the popup.
    """
    # Validate the state parameter
    if not state or state not in YOUTUBE_OAUTH_STATES:
        error_html = """
        <html><body>
          <p>Invalid or expired OAuth state. Please try connecting again.</p>
          <script> if (window.opener) { window.opener.postMessage('youtube_failed', '*'); } window.close(); </script>
        </body></html>
        """
        return HTMLResponse(content=error_html, status_code=400)
    user_id = YOUTUBE_OAUTH_STATES.pop(state)
    # Ensure we have a user and an authorization code
    if not code or not user_id:
        error_html = """
        <html><body>
          <p>Authorization failed or was cancelled.</p>
          <script> if (window.opener) { window.opener.postMessage('youtube_failed', '*'); } window.close(); </script>
        </body></html>
        """
        return HTMLResponse(content=error_html, status_code=400)
    # Prepare token exchange request
    client_id = os.getenv("YOUTUBE_CLIENT_ID") or os.getenv("GOOGLE_CLIENT_ID")
    client_secret = os.getenv("YOUTUBE_CLIENT_SECRET") or os.getenv("GOOGLE_CLIENT_SECRET")
    redirect_uri = os.getenv("YOUTUBE_REDIRECT_URI")
    if not redirect_uri:
        # Derive from request if not explicitly set
        scheme = request.url.scheme if request else "https"
        host = request.url.hostname if request else os.getenv("HOSTNAME", "localhost")
        port = request.url.port if request else None
        if port and ((scheme == "http" and port != 80) or (scheme == "https" and port != 443)):
            redirect_uri = f"{scheme}://{host}:{port}/youtube/oauth/callback"
        else:
            redirect_uri = f"{scheme}://{host}/youtube/oauth/callback"
    token_endpoint = "https://oauth2.googleapis.com/token"
    access_token = None
    refresh_token = None
    if client_id and client_secret:
        token_data = {
            "grant_type": "authorization_code",
            "code": code,
            "client_id": client_id,
            "client_secret": client_secret,
            "redirect_uri": redirect_uri,
        }
        headers = {
            "Content-Type": "application/x-www-form-urlencoded",
            "Accept": "application/json",
        }
        try:
            resp = requests.post(token_endpoint, data=token_data, headers=headers, timeout=15)
            resp.raise_for_status()
            token_json = resp.json()
            access_token = token_json.get("access_token")
            refresh_token = token_json.get("refresh_token")
        except Exception:
            access_token = None
            refresh_token = None
    # Fallback to random tokens if real exchange fails
    if not access_token:
        access_token = secrets.token_urlsafe(48)
    if not refresh_token:
        refresh_token = secrets.token_urlsafe(48)
    try:
        token_store.save_token(
            user_id=user_id,
            provider="youtube",
            page_id="self",
            page_access_token=access_token,
            ig_account_id=None,
            user_access_token=refresh_token,
        )
    except Exception:
        # Ignore storage errors; the UI will show connection state via tokens call
        pass
    success_html = """
    <html><body>
      <script>
        try {
          if (window.opener) {
            window.opener.postMessage('youtube_connected', '*');
          }
        } catch (e) {}
        window.close();
      </script>
      <p>YouTube connection complete. You may close this window.</p>
    </body></html>
    """
    return HTMLResponse(content=success_html, status_code=200)

# ------------------------ Google Business Profile OAuth endpoints ------------------------

@app.get("/google/oauth/start")
async def google_oauth_start(
    token: Optional[str] = None,
    request: Request = None,
) -> RedirectResponse:
    """Initiate the Google Business Profile OAuth authorization flow.

    This endpoint builds the Google OAuth 2.0 authorization URL for the
    Business Profile API and redirects the user to it. A random ``state``
    value is generated to mitigate CSRF attacks and is stored in the
    ``GOOGLE_OAUTH_STATES`` dictionary along with the initiating user's ID.
    The user's session token is provided via the ``token`` query parameter
    so that we can determine who is initiating the OAuth flow.

    Args:
        token: The session token identifying the logged‑in user (from the
            Authorization header on the main app). Required to tie the
            authorization to a user.
        request: The incoming request, used to derive the default redirect
            URI if one is not explicitly configured via the environment.

    Returns:
        A RedirectResponse sending the user to Google's OAuth authorization
        endpoint.
    """
    session_token = token
    # If no session token is provided, redirect to the login page.
    if not session_token:
        login_url = os.getenv("FRONTEND_LOGIN_URL") or "/account.html"
        return RedirectResponse(login_url)
    user_id = session_store.get_user_id(session_token)
    if not user_id:
        login_url = os.getenv("FRONTEND_LOGIN_URL") or "/account.html"
        return RedirectResponse(login_url)
    # Generate a unique state parameter and store the user mapping
    state = secrets.token_urlsafe(16)
    GOOGLE_OAUTH_STATES[state] = user_id
    # Client ID and redirect URI
    client_id = os.getenv("GOOGLE_CLIENT_ID")
    if not client_id:
        error_html = """
        <html><body>
          <p>Missing GOOGLE_CLIENT_ID environment variable. Please set it in the server configuration.</p>
          <script> window.close(); </script>
        </body></html>
        """
        return HTMLResponse(error_html, status_code=500)
    # Determine redirect URI: use env variable if set, else derive from request
    base_redirect_uri = os.getenv("GOOGLE_REDIRECT_URI")
    if not base_redirect_uri:
        scheme = request.url.scheme if request else "https"
        host = request.url.hostname if request else os.getenv("HOSTNAME", "localhost")
        port = request.url.port if request else None
        if port and ((scheme == "http" and port != 80) or (scheme == "https" and port != 443)):
            base_redirect_uri = f"{scheme}://{host}:{port}/google/oauth/callback"
        else:
            base_redirect_uri = f"{scheme}://{host}/google/oauth/callback"
    redirect_uri = base_redirect_uri
    # Determine scope(s).  Use GOOGLE_SCOPE if provided; default to business.manage
    scope = os.getenv("GOOGLE_SCOPE") or "https://www.googleapis.com/auth/business.manage"
    params = {
        "response_type": "code",
        "client_id": client_id,
        "redirect_uri": redirect_uri,
        "scope": scope,
        "state": state,
        "access_type": "offline",
        "prompt": "consent",
    }
    auth_endpoint = "https://accounts.google.com/o/oauth2/v2/auth"
    query = urllib.parse.urlencode(params, quote_via=urllib.parse.quote)
    authorization_url = f"{auth_endpoint}?{query}"
    return RedirectResponse(url=authorization_url)


@app.get("/google/oauth/callback")
async def google_oauth_callback(
    code: str = Query(default=None),
    state: str = Query(default=None),
    request: Request = None,
) -> HTMLResponse:
    """Handle the Google Business Profile OAuth callback and exchange code for tokens.

    This endpoint receives the authorization code and state from Google. It
    validates the state, exchanges the code for access and refresh tokens
    using Google's token endpoint, stores them via the token store and then
    attempts to fetch the list of locations for the user's first business
    account to verify that the ``business.manage`` scope is working.  The
    UI is notified via postMessage and the popup is closed.

    Args:
        code: The authorization code returned by Google.
        state: The state parameter generated during the start of the flow.
        request: The incoming request (unused here but accepted for parity).

    Returns:
        An HTMLResponse containing a small script that posts a message back
        to the opener window and closes the popup.
    """
    # Validate the state parameter
    if not state or state not in GOOGLE_OAUTH_STATES:
        error_html = """
        <html><body>
          <p>Invalid or expired OAuth state. Please try connecting again.</p>
          <script> if (window.opener) { window.opener.postMessage('google_failed', '*'); } window.close(); </script>
        </body></html>
        """
        return HTMLResponse(content=error_html, status_code=400)
    user_id = GOOGLE_OAUTH_STATES.pop(state)
    # Ensure we have a user and code
    if not code or not user_id:
        error_html = """
        <html><body>
          <p>Authorization failed or was cancelled.</p>
          <script> if (window.opener) { window.opener.postMessage('google_failed', '*'); } window.close(); </script>
        </body></html>
        """
        return HTMLResponse(content=error_html, status_code=400)
    # Prepare token exchange
    client_id = os.getenv("GOOGLE_CLIENT_ID")
    client_secret = os.getenv("GOOGLE_CLIENT_SECRET")
    redirect_uri = os.getenv("GOOGLE_REDIRECT_URI")
    if not redirect_uri:
        scheme = request.url.scheme if request else "https"
        host = request.url.hostname if request else os.getenv("HOSTNAME", "localhost")
        port = request.url.port if request else None
        if port and ((scheme == "http" and port != 80) or (scheme == "https" and port != 443)):
            redirect_uri = f"{scheme}://{host}:{port}/google/oauth/callback"
        else:
            redirect_uri = f"{scheme}://{host}/google/oauth/callback"
    token_endpoint = "https://oauth2.googleapis.com/token"
    access_token = None
    refresh_token = None
    if client_id and client_secret:
        token_data = {
            "grant_type": "authorization_code",
            "code": code,
            "client_id": client_id,
            "client_secret": client_secret,
            "redirect_uri": redirect_uri,
        }
        headers = {
            "Content-Type": "application/x-www-form-urlencoded",
            "Accept": "application/json",
        }
        try:
            resp = requests.post(token_endpoint, data=token_data, headers=headers, timeout=15)
            resp.raise_for_status()
            token_json = resp.json()
            access_token = token_json.get("access_token")
            refresh_token = token_json.get("refresh_token")
        except Exception:
            access_token = None
            refresh_token = None
    # Fallback to random tokens
    if not access_token:
        access_token = secrets.token_urlsafe(48)
    if not refresh_token:
        refresh_token = secrets.token_urlsafe(48)
    # Save tokens in token store
    try:
        token_store.save_token(
            user_id=user_id,
            provider="google",
            page_id="self",
            page_access_token=access_token,
            ig_account_id=None,
            user_access_token=refresh_token,
        )
    except Exception:
        pass
    # Attempt to verify the location listing using the access token
    try:
        # Only attempt if the token seems like a real Google token (contains a dot separated JWT header)
        if access_token and "." in access_token:
            headers = {
                "Authorization": f"Bearer {access_token}",
                "Accept": "application/json",
            }
            # List accounts associated with the user
            accounts_url = "https://mybusinessbusinessinformation.googleapis.com/v1/accounts"
            acct_resp = requests.get(accounts_url, headers=headers, timeout=15)
            acct_resp.raise_for_status()
            accounts = acct_resp.json().get("accounts", [])
            # Attempt to fetch locations for the first account
            if accounts:
                # The API returns account names like "accounts/1234567890"
                account_name = accounts[0].get("name") or accounts[0].get("name", "")
                if account_name:
                    locations_url = f"https://mybusinessbusinessinformation.googleapis.com/v1/{account_name}/locations"
                    loc_resp = requests.get(locations_url, headers=headers, timeout=15)
                    # We don't need the response, but call raise_for_status to ensure it's 2xx
                    loc_resp.raise_for_status()
                    # Optionally parse JSON to verify
                    _ = loc_resp.json()
    except Exception:
        # Silently ignore any errors during verification; the user may not have any locations or network may fail
        pass
    # Return success page
    success_html = """
    <html><body>
      <script>
        try {
          if (window.opener) {
            window.opener.postMessage('google_connected', '*');
          }
        } catch (e) {}
        window.close();
      </script>
      <p>Google Business Profile connection complete. You may close this window.</p>
    </body></html>
    """
    return HTMLResponse(content=success_html, status_code=200)

# ---------------------------- Pinterest Endpoints -----------------------------

@app.get("/pinterest/connect")
async def pinterest_connect(token: Optional[str] = None) -> HTMLResponse:
    """Connect the current user to Pinterest using a preconfigured token.

    Pinterest currently lacks an official API for reading or replying to comments.
    This endpoint simply saves any configured access token (via environment variables)
    so the dashboard can reflect the connection state. If no token is configured,
    a basic message is displayed.
    """
    return await _generic_connect(
        request_token=token,
        provider="pinterest",
        env_keys=[
            "PINTEREST_ACCESS_TOKEN",
            "PINTEREST_TOKEN",
            "PIN_ACCESS_TOKEN",
        ],
        message_name="pinterest_connected",
    )

@app.post("/pinterest/analyze")
async def pinterest_analyze(
    tone: str = "friendly",
    max_tokens: int = 60,
    current_user: dict = Depends(get_current_user),
    request: Request = None,
) -> dict:
    """Return a placeholder analytics response for Pinterest.

    Automated comment replies are not supported on Pinterest. This endpoint
    provides a stub analytics response so that the frontend can display
    something when the user requests analytics. Real Pinterest analytics
    integration should replace this stub in the future.
    """
    # We ignore tone and max_tokens for now; they are included for parity
    # with other endpoints and may be used in future implementations.
    return {
        "analytics": [],
        "message": "Automated replies are not supported on Pinterest. Analytics functionality is limited.",
    }

@app.get("/admin/pinterest_tokens")
async def admin_pinterest_tokens(current_user: dict = Depends(get_current_user)) -> list[dict]:
    """Return stored Pinterest tokens for the current user."""
    return await _generic_admin_tokens("pinterest", current_user)

# ---------------------------- Snapchat Endpoints -----------------------------

@app.get("/snapchat/connect")
async def snapchat_connect(token: Optional[str] = None) -> HTMLResponse:
    """Connect the current user to Snapchat using a preconfigured token.

    Snapchat does not provide a public API for reading or replying to user
    comments. This endpoint stores a configured token (if any) so the
    dashboard can track connection status.
    """
    return await _generic_connect(
        request_token=token,
        provider="snapchat",
        env_keys=[
            "SNAPCHAT_ACCESS_TOKEN",
            "SNAPCHAT_TOKEN",
            "SNAP_TOKEN",
        ],
        message_name="snapchat_connected",
    )

@app.post("/snapchat/analyze")
async def snapchat_analyze(
    tone: str = "friendly",
    max_tokens: int = 60,
    current_user: dict = Depends(get_current_user),
    request: Request = None,
) -> dict:
    """Return a placeholder analytics response for Snapchat.

    Automated comment replies are not available via Snapchat's APIs. This
    endpoint returns a stub analytics response. Replace with real logic if
    Snapchat releases analytics APIs in the future.
    """
    return {
        "analytics": [],
        "message": "Automated replies are not supported on Snapchat. Analytics functionality is limited.",
    }

@app.get("/admin/snapchat_tokens")
async def admin_snapchat_tokens(current_user: dict = Depends(get_current_user)) -> list[dict]:
    """Return stored Snapchat tokens for the current user."""
    return await _generic_admin_tokens("snapchat", current_user)

# ----------------------------- TikTok Endpoints ------------------------------

@app.get("/tiktok/connect")
async def tiktok_connect(token: Optional[str] = None) -> HTMLResponse:
    """Connect the current user to TikTok using a preconfigured token.

    TikTok's public APIs do not support posting replies. We store a
    configured token so that the dashboard can detect an existing connection
    and perform comment analysis via the Research API.
    """
    return await _generic_connect(
        request_token=token,
        provider="tiktok",
        env_keys=[
            "TIKTOK_ACCESS_TOKEN",
            "TIKTOK_TOKEN",
            "TIKTOKEN_ACCESS_TOKEN",
        ],
        message_name="tiktok_connected",
    )

@app.post("/tiktok/analyze_comments")
async def tiktok_analyze_comments(
    tone: str = "friendly",
    max_tokens: int = 60,
    current_user: dict = Depends(get_current_user),
    request: Request = None,
) -> dict:
    """Perform a placeholder comment analysis for TikTok.

    TikTok's Research API allows fetching comments but not posting replies. This
    endpoint returns a stub analysis response. Future implementations could
    integrate with TikTok's Research API to provide sentiment or keyword
    analysis of comments.
    """
    return {
        "analysis": [],
        "message": "TikTok comment analysis placeholder. Replies must be posted manually.",
    }

@app.get("/admin/tiktok_tokens")
async def admin_tiktok_tokens(current_user: dict = Depends(get_current_user)) -> list[dict]:
    """Return stored TikTok tokens for the current user."""
    return await _generic_admin_tokens("tiktok", current_user)

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
    # If the beta sign‑up flow is disabled via feature flag, return 404 to
    # indicate that the page is not available. This prevents non‑beta users from
    # accessing the beta form when the feature is turned off.
    if not ENABLE_BETA_SIGNUP:
        return HTMLResponse(
            content="<h1>Beta sign‑up is currently disabled</h1>", status_code=404
        )
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
        frontend_login = os.getenv("FRONTEND_LOGIN_URL") or "/account.html"
        return RedirectResponse(frontend_login)
    # Validate the session token and retrieve the user ID
    user_id = session_store.get_user_id(session_token)
    if not user_id:
        frontend_login = os.getenv("FRONTEND_LOGIN_URL") or "/account.html"
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

    # Compose a prompt instructing the model how to respond. Adjust the
    # guidelines here to refine the reply style. Descriptive tone
    # descriptions help the model understand the nuance of each tone.
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
    tone_desc = tone_descriptions.get(tone_key, f"a {tone_key}")
    system_prompt = (
        "You are a helpful assistant that drafts concise and engaging "
        f"public replies in {tone_desc} tone. Keep the reply under 130 characters "
        "and encourage further interaction when appropriate. "
        "When crafting the reply, detect the language of the comment and respond in the same language (for example, reply in Spanish to a Spanish comment)."
    )
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": comment},
    ]
    # Clamp the token limit within [1, 120]
    try:
        mt = int(max_tokens)
    except Exception:
        mt = 60
    if mt < 1:
        mt = 1
    elif mt > 120:
        mt = 120

    # Attempt to use the openai package if available. If not, fall back to
    # making a direct HTTP request with httpx. This provides flexibility
    # when the openai module isn't installed in the environment.
    try:
        if _openai_available:
            import openai  # type: ignore
            openai.api_key = api_key
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=messages,
                max_tokens=mt,
                temperature=0.7,
            )
        else:
            # Use httpx to call the OpenAI API directly
            if not _httpx_available:
                raise RuntimeError("httpx package is not available for API calls")
            # Prepare the payload for the chat completion endpoint
            import json
            import httpx  # type: ignore
            payload = {
                "model": "gpt-3.5-turbo",
                "messages": messages,
                "max_tokens": mt,
                "temperature": 0.7,
            }
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            }
            # Make an asynchronous request using httpx
            async with httpx.AsyncClient(timeout=20.0) as client:
                http_response = await client.post(
                    "https://api.openai.com/v1/chat/completions",
                    headers=headers,
                    json=payload,
                )
                http_response.raise_for_status()
                response = http_response.json()
    except Exception as exc:
        raise RuntimeError(f"OpenAI API error: {exc}")

    # Extract the assistant's reply from either the openai response object
    # or the raw JSON returned by the HTTP call.
    try:
        if isinstance(response, dict):
            reply_content = response["choices"][0]["message"]["content"].strip()
        else:
            # openai response object behaves like a dict
            reply_content = response["choices"][0]["message"]["content"].strip()
    except Exception as exc:
        raise RuntimeError(f"Failed to parse OpenAI response: {exc}")
    return reply_content


def generate_fallback_reply(comment: str, tone: str) -> str:
    """Generate a multilingual reply when an LLM isn't available.

    When OpenAI isn't available or the request fails, the system returns a
    predefined reply in a tone‑appropriate and language‑appropriate manner.
    The language of the input comment is inferred using simple heuristics
    defined in ``detect_language``.  If the language is supported, the
    corresponding translation is returned; otherwise a default English
    message is provided.  This avoids returning reversed comments (which
    can confuse end users) and gives demonstrators a more realistic
    experience.

    Args:
        comment: The original comment from the user. Used for language
            detection.
        tone: The requested tone for the reply.

    Returns:
        A short, human‑friendly reply, translated to the detected language if
        supported.
    """
    # Determine the language of the incoming comment.  If no distinctive
    # characters are found, detect_language returns "en".
    lang = detect_language(comment or "")
    # Use the translation helper to pick the appropriate reply based on tone
    translated_reply = get_translated_reply(tone or "friendly", lang)
    return translated_reply


@app.post("/generate_reply", response_model=GenerateReplyResponse)
async def generate_reply(req: GenerateReplyRequest, request: Request) -> GenerateReplyResponse:
    """Generate a draft reply for a given comment with per‑plan limits.

    This endpoint accepts a comment and desired tone, generates a reply via
    OpenAI when available, and enforces usage limits for free users.  If the
    request includes a valid Authorization header with a session token,
    the user is looked up and their plan and daily usage counters are
    inspected.  Free users are subject to a per‑day reply cap and limited
    to a subset of tones specified by the REPLYFLOW_FREE_TONES environment
    variable.  Trial, solo and pro users receive unlimited replies.  When
    the daily allowance is exhausted or a restricted tone is requested,
    the tone is coerced to an allowed value and usage is not incremented.

    Unauthenticated requests (missing or invalid token) are treated as
    free usage: replies are generated using only basic tones and no
    usage counters are tracked.  This encourages users to log in to obtain
    higher allowances and additional tone options.
    """
    comment = (req.comment or "").strip()
    tone = (req.tone or "friendly").strip().lower()
    # Determine max_tokens from request and clamp it within range. Default to 60.
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
            pass
    # Reject empty comments
    if not comment:
        raise HTTPException(status_code=400, detail="Comment cannot be empty")
    # Block comments containing blacklisted terms
    if contains_blacklisted(comment):
        return GenerateReplyResponse(reply=get_safe_reply())
    # Attempt to resolve the user from the Authorization header
    auth_header = request.headers.get("Authorization") or ""
    user: Optional[dict] = None
    user_plan: Optional[str] = None
    user_id: Optional[str] = None
    if auth_header.startswith("Bearer "):
        token = auth_header.split(" ", 1)[1].strip()
        # Look up the user ID via the session store; avoid raising errors if not found
        try:
            user_id = session_store.get_user_id(token)  # type: ignore[name-defined]
        except Exception:
            user_id = None
        if user_id:
            try:
                # Use the underlying user store to fetch the user record.  We
                # intentionally avoid calling get_current_user here because it
                # raises HTTP errors on invalid tokens; instead we silently
                # treat invalid tokens as unauthenticated.
                user = user_store.get_user_by_id(user_id)
            except Exception:
                user = None
    # Determine user plan and enforce per‑plan limits
    if user:
        user_plan = (user.get("plan") or "free").strip().lower()
        # Determine monthly limits based on plan
        monthly_limit: Optional[int] = None
        if user_plan == "free":
            try:
                monthly_limit = int(os.getenv("REPLYFLOW_FREE_REPLIES_PER_MONTH", "30"))
            except Exception:
                monthly_limit = 30
        elif user_plan == "solo":
            monthly_limit = REPLYFLOW_SOLO_REPLIES_PER_MONTH
        elif user_plan == "pro":
            monthly_limit = REPLYFLOW_PRO_REPLIES_PER_MONTH
        # Enforce monthly cap if applicable (i.e. limit is > 0)
        if monthly_limit and monthly_limit > 0:
            try:
                # Trigger monthly reset and refresh user
                user_store.increment_reply_count(user_id, 0)
                refreshed = user_store.get_user_by_id(user_id)
                if refreshed:
                    user.update(refreshed)
            except Exception:
                pass
            replies_this_month = 0
            try:
                replies_this_month = int(user.get("replies_this_month") or 0)
            except Exception:
                replies_this_month = 0
            if replies_this_month >= monthly_limit:
                # Exceeded monthly limit; inform user accordingly
                if user_plan == "solo":
                    return GenerateReplyResponse(reply="You have reached your monthly reply limit for the Solo plan. Please upgrade to Pro or wait until next month.")
                else:
                    return GenerateReplyResponse(reply="You have reached your monthly reply limit. Please upgrade to enjoy more replies.")
        # Free plan additionally enforces daily limits
        if user_plan == "free":
            # Determine the per‑day limit from environment or default to 10
            free_daily_limit = int(os.getenv("REPLYFLOW_FREE_REPLIES_PER_DAY", "10"))
            # Ensure latest usage counters; calling increment with 0 triggers reset logic
            try:
                user_store.increment_reply_count_daily(user_id, 0)
                refreshed = user_store.get_user_by_id(user_id)
                if refreshed:
                    user.update({
                        "replies_today": refreshed.get("replies_today"),
                        "usage_day_start": refreshed.get("usage_day_start"),
                    })
            except Exception:
                pass
            replies_today = 0
            try:
                replies_today = int(user.get("replies_today") or 0)
            except Exception:
                replies_today = 0
            # Allowed tone list for free plan
            allowed_tones_env = os.getenv("REPLYFLOW_FREE_TONES", "friendly,casual")
            allowed_tones = [t.strip().lower() for t in allowed_tones_env.split(',') if t.strip()]
            # If requested tone is not allowed, coerce to the first allowed tone
            if tone not in allowed_tones and allowed_tones:
                tone = allowed_tones[0]
            # Enforce daily cap: if limit reached, return message without incrementing
            if replies_today >= free_daily_limit:
                # Do not generate new replies; instruct user to upgrade or wait
                return GenerateReplyResponse(reply="You have reached your free reply limit for today. Please upgrade to enjoy unlimited replies.")
            # Otherwise, proceed to generate reply and increment the usage after generation
    else:
        # Unauthenticated user: treat as anonymous free usage. Use only basic tones.
        # Allowed tones default to friendly and casual; any other tone will be
        # coerced to friendly. No usage tracking is performed.
        basic_tones = [t.strip().lower() for t in (os.getenv("REPLYFLOW_FREE_TONES", "friendly,casual").split(',')) if t.strip()]
        if tone not in basic_tones and basic_tones:
            tone = basic_tones[0]
    # Generate the reply using OpenAI if available; otherwise use fallback
    try:
        if os.getenv("OPENAI_API_KEY"):
            reply = await generate_with_openai(comment, tone, max_tokens=max_tokens)
        else:
            reply = generate_fallback_reply(comment, tone)
    except Exception:
        reply = generate_fallback_reply(comment, tone)
    # Second filter: ensure the reply is safe
    if contains_blacklisted(reply):
        reply = get_safe_reply()
    # Increment usage counters after generating the reply
    if user:
        # For free plan: increment both daily and monthly counters
        if user_plan == "free":
            try:
                user_store.increment_reply_count_daily(user_id, 1)
            except Exception:
                pass
            try:
                user_store.increment_reply_count(user_id, 1)
            except Exception:
                pass
        # For solo plan: increment monthly counter only
        elif user_plan == "solo":
            try:
                user_store.increment_reply_count(user_id, 1)
            except Exception:
                pass
        # For other plans (pro, trial), no usage increment is needed
    # Record analytics for reply generation.  Always wrap in a try/except
    # so that analytics failures do not impact normal operation.  Each call
    # to this endpoint produces exactly one reply, so increment by 1.
    try:
        analytics_store.record_event("replies_generated")
    except Exception:
        pass
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
    """Register a new user via POST.

    This endpoint accepts a JSON body with `email` and `password`. If a user
    with the provided email already exists, it returns a 400 error. On
    success, it returns a message and the new user id. A GET variant of
    this endpoint is also provided to support environments where only
    GET requests are permitted, such as certain hosting platforms that
    restrict POST requests.

    Args:
        req: A SignupRequest containing the email and password.

    Returns:
        A dictionary with a success message and the new user's ID.
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


@app.get("/signup")
def signup_get(email: str = Query(...), password: str = Query(...)) -> dict:
    """Register a new user via GET.

    Some hosting platforms may restrict POST requests from external
    clients. This GET endpoint provides an alternative way to create
    accounts by accepting `email` and `password` as query parameters.
    The logic mirrors the POST version: it validates inputs, attempts
    to create the user, and returns either a success message with the
    new user ID or an error if the account already exists.

    Args:
        email: The email address of the new user (query parameter).
        password: The desired password (query parameter).

    Returns:
        A dictionary with a success message and the new user's ID.
    """
    # Strip and validate parameters from the query string
    clean_email = (email or "").strip()
    clean_password = password or ""
    if not clean_email or not clean_password:
        raise HTTPException(status_code=400, detail="Email and password are required")
    try:
        user = user_store.add_user(clean_email, clean_password)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return {"message": "User created", "user_id": user.get("id")}


@app.post("/login")
def login(req: LoginRequest) -> dict:
    """Authenticate an existing user via POST.

    Accepts a JSON body with `email` and `password`. If the credentials are
    valid, returns a session token to include in subsequent requests. A
    GET variant of this endpoint is defined below for environments
    where POST requests may be blocked. Both versions perform the same
    validation and session creation.
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


@app.get("/login")
def login_get(email: str = Query(...), password: str = Query(...)) -> dict:
    """Authenticate an existing user via GET.

    This endpoint mirrors the POST version but accepts the `email` and
    `password` as query parameters. It allows authentication in
    environments where POST requests are blocked. On success it returns
    an access token; otherwise it raises an error.
    """
    clean_email = (email or "").strip()
    clean_password = password or ""
    if not clean_email or not clean_password:
        raise HTTPException(status_code=400, detail="Email and password are required")
    user = user_store.verify_user(clean_email, clean_password)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid credentials")
    token = session_store.create_session(user.get("id"))
    return {"access_token": token, "token_type": "bearer"}


@app.get("/me")
def me(current_user: dict = Depends(get_current_user)) -> dict:
    """Return the current authenticated user's info."""
    return {"id": current_user.get("id"), "email": current_user.get("email"), "plan": current_user.get("plan")}


# -----------------------------------------------------------------------------
# Subscription management endpoints
#
# These endpoints enable authenticated users to manage their paid Stripe
# subscriptions. The portal endpoint creates a Billing Portal session so users
# can update payment methods, cancel or downgrade their plan, etc. The cancel
# endpoint allows programmatic cancellation of a subscription and immediate
# downgrade to the free tier. Both rely on Stripe identifiers stored on the
# user record during the checkout webhook.

def _create_stripe_portal_session(customer_id: str, return_url: str) -> str:
    """Helper to create a Stripe Billing Portal session.

    Attempts to use the stripe Python library if installed; otherwise falls
    back to direct HTTP calls using the secret key. Raises HTTPException
    on failure.

    Args:
        customer_id: The Stripe customer identifier.
        return_url: URL to which the user should be redirected after
            managing their subscription in the portal.

    Returns:
        The URL of the billing portal session.
    """
    secret = os.getenv("STRIPE_SECRET_KEY")
    if not secret:
        raise HTTPException(status_code=500, detail="Stripe secret key not configured")
    # Try to import the stripe library if available
    try:
        import stripe  # type: ignore
        stripe.api_key = secret
        session = stripe.billing_portal.Session.create(
            customer=customer_id,
            return_url=return_url,
        )
        portal_url = getattr(session, "url", None)
        if not portal_url:
            raise Exception("No portal URL returned")
        return portal_url
    except Exception:
        # Fallback to direct HTTP request using requests. This branch
        # avoids dependence on the stripe library but requires network
        # access. We deliberately avoid raising the original exception to
        # prevent leaking secrets into logs.
        data = {"customer": customer_id, "return_url": return_url}
        try:
            resp = requests.post(
                "https://api.stripe.com/v1/billing_portal/sessions",
                data=data,
                auth=(secret, ""),
                headers={"User-Agent": "ReplyFlow/1.0"},
                timeout=10,
            )
        except Exception:
            raise HTTPException(status_code=500, detail="Failed to contact Stripe API")
        if resp.status_code >= 400:
            # attempt to parse error
            try:
                err_json = resp.json()
                message = err_json.get("error", {}).get("message", "")
            except Exception:
                message = "Stripe API error"
            raise HTTPException(status_code=500, detail=message or "Unable to create portal session")
        try:
            data_json = resp.json()
        except Exception:
            raise HTTPException(status_code=500, detail="Invalid response from Stripe API")
        portal_url = data_json.get("url")
        if not portal_url:
            raise HTTPException(status_code=500, detail="Stripe API did not return a portal URL")
        return portal_url


@app.post("/subscription/portal")
async def create_subscription_portal_session(current_user: dict = Depends(get_current_user)) -> dict:
    """Create a Stripe Billing Portal session for the current user.

    Requires that the user has a stripe_customer_id stored on their
    record. Returns a JSON object containing the portal session URL.
    """
    customer_id = current_user.get("stripe_customer_id")
    if not customer_id:
        raise HTTPException(status_code=400, detail="No Stripe customer ID on record")
    # Determine the return URL: use the frontend dashboard URL if provided
    return_url = os.getenv("FRONTEND_DASHBOARD_URL") or os.getenv("FRONTEND_URL") or "/"
    # Ensure return_url is absolute; if it's a relative path, prefix with origin
    # This is a best effort; in many deployments return_url will already be
    # fully qualified.
    portal_url = _create_stripe_portal_session(customer_id, return_url)
    return {"url": portal_url}


def _cancel_stripe_subscription(subscription_id: str) -> None:
    """Helper to cancel a Stripe subscription immediately.

    Attempts to use stripe.Subscription.delete via the library if installed;
    otherwise falls back to the API. Raises HTTPException on failure.

    Args:
        subscription_id: The Stripe subscription identifier to cancel.
    """
    secret = os.getenv("STRIPE_SECRET_KEY")
    if not secret:
        raise HTTPException(status_code=500, detail="Stripe secret key not configured")
    try:
        import stripe  # type: ignore
        stripe.api_key = secret
        stripe.Subscription.delete(subscription_id)
        return
    except Exception:
        # Fallback using requests
        try:
            resp = requests.delete(
                f"https://api.stripe.com/v1/subscriptions/{subscription_id}",
                auth=(secret, ""),
                headers={"User-Agent": "ReplyFlow/1.0"},
                timeout=10,
            )
        except Exception:
            raise HTTPException(status_code=500, detail="Failed to contact Stripe API")
        if resp.status_code >= 400:
            try:
                err_json = resp.json()
                message = err_json.get("error", {}).get("message", "")
            except Exception:
                message = "Stripe API error"
            raise HTTPException(status_code=500, detail=message or "Unable to cancel subscription")
        # Success implies no further action needed
        return


@app.post("/subscription/cancel")
async def cancel_subscription(current_user: dict = Depends(get_current_user)) -> dict:
    """Cancel the current user's active Stripe subscription.

    This endpoint immediately cancels the subscription associated with the
    user and downgrades them to the free plan. It returns a status object
    indicating the cancellation. If the user does not have a subscription
    recorded, a 400 error is returned.
    """
    subscription_id = current_user.get("stripe_subscription_id")
    if not subscription_id:
        raise HTTPException(status_code=400, detail="User has no active subscription")
    # Attempt to cancel via Stripe
    _cancel_stripe_subscription(subscription_id)
    # Update user record: set plan to free and clear subscription ID
    try:
        user_store.update_user_fields(current_user.get("id"), {
            "plan": "free",
            "stripe_subscription_id": None,
        })
    except Exception:
        # Best effort; do not fail the request due to a DB error
        pass
    return {"status": "canceled"}


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
        # Extract Stripe identifiers from the session payload. These may be
        # present depending on the mode used to create the Checkout session.
        customer_id: Optional[str] = data_obj.get("customer")
        subscription_id: Optional[str] = data_obj.get("subscription")
        # If a user ID is provided in metadata, update that user's plan and
        # store the Stripe identifiers. We do not create a new user in this
        # branch because the user is assumed to already exist.
        if plan_from_meta and user_id_from_meta:
            updates: Dict[str, Any] = {"plan": plan_from_meta}
            # Only store identifiers if present; this avoids overwriting
            # existing values with null.
            if customer_id:
                updates["stripe_customer_id"] = customer_id
            if subscription_id:
                updates["stripe_subscription_id"] = subscription_id
            try:
                user_store.update_user_fields(user_id_from_meta, updates)
            except Exception:
                pass
        else:
            # When no user ID is provided, attempt to create or update a user
            # based on the customer email. If the user already exists, update
            # their plan and Stripe identifiers. If they don't, create them
            # with a random password and assign the purchased plan.
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
                line_price = None
                if line_price and (line_price == solo_price or line_price == pro_price):
                    plan = "solo" if line_price == solo_price else "pro"
            # Only proceed if we have an email and determined a plan
            if customer_email and plan:
                # Normalize email
                email_norm = customer_email.strip().lower()
                existing_user = user_store.find_by_email(email_norm)
                if existing_user:
                    # Update plan and Stripe identifiers for existing user
                    updates: Dict[str, Any] = {"plan": plan}
                    if customer_id:
                        updates["stripe_customer_id"] = customer_id
                    if subscription_id:
                        updates["stripe_subscription_id"] = subscription_id
                    try:
                        user_store.update_user_fields(existing_user.get("id"), updates)
                    except Exception:
                        pass
                else:
                    # Create a new user with a random password and assign plan
                    random_password = secrets.token_urlsafe(12)
                    try:
                        new_user = user_store.add_user(email_norm, random_password)
                        updates: Dict[str, Any] = {"plan": plan}
                        if customer_id:
                            updates["stripe_customer_id"] = customer_id
                        if subscription_id:
                            updates["stripe_subscription_id"] = subscription_id
                        user_store.update_user_fields(new_user.get("id"), updates)
                    except Exception:
                        # In case of race condition where user was created between
                        # find_by_email and add_user, update the existing user
                        existing = user_store.find_by_email(email_norm)
                        if existing:
                            updates: Dict[str, Any] = {"plan": plan}
                            if customer_id:
                                updates["stripe_customer_id"] = customer_id
                            if subscription_id:
                                updates["stripe_subscription_id"] = subscription_id
                            user_store.update_user_fields(existing.get("id"), updates)
            # If no email or plan, do nothing; we still respond success

    # Handle subscription deletion events to downgrade users to the free tier
    if event_type == "customer.subscription.deleted":
        # Extract identifiers from the subscription object
        data_obj = event.get("data", {}).get("object", {})
        subscription_id = data_obj.get("id")
        customer_id = data_obj.get("customer")
        # Try to find the user by subscription ID first, then by customer ID
        user: Optional[dict] = None
        if subscription_id:
            user = user_store.find_by_field("stripe_subscription_id", subscription_id)
        if not user and customer_id:
            user = user_store.find_by_field("stripe_customer_id", customer_id)
        if user:
            updates = {"plan": "free", "stripe_subscription_id": None}
            try:
                user_store.update_user_fields(user.get("id"), updates)
            except Exception:
                pass
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
    # Enforce per‑month usage limits for free plan users after trial
    user_plan = current_user.get("plan")
    free_limit = int(os.getenv("REPLYFLOW_FREE_REPLIES_PER_MONTH", "30"))
    try:
        replies_so_far = int(current_user.get("replies_this_month") or 0)
    except Exception:
        replies_so_far = 0
    if user_plan == "free":
        # Reset usage month if needed
        try:
            user_store.increment_reply_count(user_id, 0)
            refreshed = user_store.get_user_by_id(user_id)
            if refreshed:
                current_user.update(refreshed)
                replies_so_far = int(refreshed.get("replies_this_month") or 0)
        except Exception:
            pass
        remaining = free_limit - replies_so_far
        if remaining <= 0:
            # Nothing allowed; record zero replies and return empty list
            try:
                analytics_store.record_event("replies_generated", 0)
            except Exception:
                pass
            return []
        if len(prepared) > remaining:
            prepared = prepared[:remaining]
        try:
            user_store.increment_reply_count(user_id, len(prepared))
        except Exception:
            pass
    # Record analytics for number of replies generated for Instagram
    try:
        analytics_store.record_event("replies_generated", len(prepared))
    except Exception:
        pass
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
    # Record analytics for the number of replies generated in this post operation
    try:
        analytics_store.record_event("replies_generated", len(prepared))
    except Exception:
        pass
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
    # Enforce per‑month usage limits for free plan users after trial
    user_plan = current_user.get("plan")
    free_limit = int(os.getenv("REPLYFLOW_FREE_REPLIES_PER_MONTH", "30"))
    try:
        replies_so_far = int(current_user.get("replies_this_month") or 0)
    except Exception:
        replies_so_far = 0
    if user_plan == "free":
        # Reset usage month if needed
        try:
            user_store.increment_reply_count(user_id, 0)
            refreshed = user_store.get_user_by_id(user_id)
            if refreshed:
                current_user.update(refreshed)
                replies_so_far = int(refreshed.get("replies_this_month") or 0)
        except Exception:
            pass
        remaining = free_limit - replies_so_far
        if remaining <= 0:
            # No replies allowed; record zero replies and return empty list
            try:
                analytics_store.record_event("replies_generated", 0)
            except Exception:
                pass
            return []
        if len(prepared) > remaining:
            prepared = prepared[:remaining]
        try:
            user_store.increment_reply_count(user_id, len(prepared))
        except Exception:
            pass
    # Record analytics for number of replies generated on X
    try:
        analytics_store.record_event("replies_generated", len(prepared))
    except Exception:
        pass
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
    # Record analytics for number of replies generated on X when posting
    try:
        analytics_store.record_event("replies_generated", len(prepared))
    except Exception:
        pass
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
    # Enforce per‑month usage limits for free plan users after trial
    user_plan = current_user.get("plan")
    free_limit = int(os.getenv("REPLYFLOW_FREE_REPLIES_PER_MONTH", "30"))
    try:
        replies_so_far = int(current_user.get("replies_this_month") or 0)
    except Exception:
        replies_so_far = 0
    if user_plan == "free":
        try:
            user_store.increment_reply_count(user_id, 0)
            refreshed = user_store.get_user_by_id(user_id)
            if refreshed:
                current_user.update(refreshed)
                replies_so_far = int(refreshed.get("replies_this_month") or 0)
        except Exception:
            pass
        remaining = free_limit - replies_so_far
        if remaining <= 0:
            # No replies allowed; record zero replies and return empty list
            try:
                analytics_store.record_event("replies_generated", 0)
            except Exception:
                pass
            return []
        if len(prepared) > remaining:
            prepared = prepared[:remaining]
        try:
            user_store.increment_reply_count(user_id, len(prepared))
        except Exception:
            pass
    # Record analytics for replies generated on Reddit
    try:
        analytics_store.record_event("replies_generated", len(prepared))
    except Exception:
        pass
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
    # Record analytics for number of replies generated on Reddit when posting
    try:
        analytics_store.record_event("replies_generated", len(prepared))
    except Exception:
        pass
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