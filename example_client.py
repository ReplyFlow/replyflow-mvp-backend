"""
Example client for the ReplyFlow MVP API.

This script illustrates how to interact with the backend endpoints from a
Python environment. You can use it to test generating replies for arbitrary
comments as well as preparing and posting replies to Facebook. Before
running the script ensure that your backend server is up and running and
that you have completed the Facebook OAuth flow to store at least one
Page token. See README.md for details on starting the server and
authorizing Facebook pages.

Usage:
    python3 example_client.py --comment "Great job!" --tone friendly

    python3 example_client.py --prepare

    python3 example_client.py --post

By default the script assumes the API is running on http://127.0.0.1:8000.
You can override this with the --base-url option.
"""

import argparse
import json
from typing import Any, Dict

import requests


def generate_reply(base_url: str, comment: str, tone: str) -> str:
    """Call the /generate_reply endpoint and return the reply.

    Args:
        base_url: Base URL of the API (e.g. http://127.0.0.1:8000)
        comment: The social media comment to reply to
        tone: The tone of the desired reply

    Returns:
        The generated reply string.
    """
    url = f"{base_url.rstrip('/')}/generate_reply"
    payload = {"comment": comment, "tone": tone}
    resp = requests.post(url, json=payload, timeout=10)
    resp.raise_for_status()
    data = resp.json()
    return data.get("reply", "")


def prepare_facebook_replies(base_url: str, tone: str, max_posts: int, max_comments: int) -> list[Dict[str, Any]]:
    """Call the /facebook/prepare_replies endpoint.

    Args:
        base_url: Base URL of the API
        tone: Desired tone for replies
        max_posts: Number of recent posts to fetch per page
        max_comments: Number of recent comments per post

    Returns:
        A list of dictionaries containing prepared replies.
    """
    url = f"{base_url.rstrip('/')}/facebook/prepare_replies"
    params = {"tone": tone, "max_posts": max_posts, "max_comments": max_comments}
    resp = requests.post(url, params=params, timeout=30)
    resp.raise_for_status()
    return resp.json()


def post_facebook_replies(base_url: str, tone: str, max_posts: int, max_comments: int) -> Dict[str, Any]:
    """Call the /facebook/post_replies endpoint to publish the replies.

    Args:
        base_url: Base URL of the API
        tone: Desired tone for replies
        max_posts: Number of recent posts to fetch per page
        max_comments: Number of comments per post

    Returns:
        A dictionary summarizing how many replies were posted per page.
    """
    url = f"{base_url.rstrip('/')}/facebook/post_replies"
    params = {"tone": tone, "max_posts": max_posts, "max_comments": max_comments}
    resp = requests.post(url, params=params, timeout=30)
    resp.raise_for_status()
    return resp.json()


def main() -> None:
    parser = argparse.ArgumentParser(description="Interact with the ReplyFlow MVP API")
    parser.add_argument(
        "--base-url",
        type=str,
        default="http://127.0.0.1:8000",
        help="Base URL of the ReplyFlow API service",
    )
    parser.add_argument(
        "--comment",
        type=str,
        help="Generate a reply for this comment",
    )
    parser.add_argument(
        "--tone",
        type=str,
        default="friendly",
        help="Desired tone for the reply",
    )
    parser.add_argument(
        "--prepare",
        action="store_true",
        help="Prepare Facebook replies without posting",
    )
    parser.add_argument(
        "--post",
        action="store_true",
        help="Fetch comments and post replies to Facebook (external side effect)",
    )
    parser.add_argument(
        "--max-posts",
        type=int,
        default=3,
        help="Maximum number of recent posts to fetch per page",
    )
    parser.add_argument(
        "--max-comments",
        type=int,
        default=5,
        help="Maximum number of recent comments to fetch per post",
    )

    args = parser.parse_args()

    base_url = args.base_url
    tone = args.tone

    if args.comment:
        reply = generate_reply(base_url, args.comment, tone)
        print(f"Reply: {reply}")

    if args.prepare:
        prepared = prepare_facebook_replies(base_url, tone, args.max_posts, args.max_comments)
        print("Prepared replies:")
        print(json.dumps(prepared, indent=2))

    if args.post:
        results = post_facebook_replies(base_url, tone, args.max_posts, args.max_comments)
        print("Posted replies summary:")
        print(json.dumps(results, indent=2))

    if not (args.comment or args.prepare or args.post):
        parser.print_help()


if __name__ == "__main__":
    main()