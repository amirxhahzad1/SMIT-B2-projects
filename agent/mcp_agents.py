"""MCP-backed sub-agents: GitHub, Google Calendar, Gmail."""
import os

from langchain_mcp_adapters.client import MultiServerMCPClient

from .config import GITHUB_TOKEN, GOOGLE_OAUTH_CREDENTIALS
from .scheduler_tool import build_scheduler_tools

GITHUB_PROMPT = """You are the GitHub specialist, connected to GitHub through MCP.
Answer any question about repositories, code, issues, pull requests, commits, branches, workflows and users
by calling the GitHub tools. Never guess: look things up. For write actions (creating issues/PRs/comments,
pushing files) make sure the user clearly asked for them. Summarise results in a short, meaningful manner with links."""

CALENDAR_PROMPT = """You are the Google Calendar specialist, connected through MCP.
You can list calendars, read/search events, check free/busy, and create, update and delete events.
- Resolve relative dates against the current date/time. Always set explicit timezones.
- Before creating an event, make sure you have title, start, and end/duration; default duration is 30 minutes
  if unspecified, and say so. Check for conflicts first when scheduling.
- Add attendees only if the user names them with email addresses. Keep responses short, direct, and meaningful."""

EMAIL_PROMPT = """You are the email specialist, connected to Gmail through MCP, plus a local email scheduler.
- To WRITE an email: compose a clear, appropriately-toned message for the named recipient.
- DRAFT: if the user asks for a draft (or is unsure), save it with the draft tool and DO NOT send.
- SEND NOW: send only when the user clearly says to send.
- SEND LATER: use schedule_email with an ISO-8601 time and timezone; confirm the exact time back.
- If the recipient's email address is unknown, ask for it rather than inventing one.
- You can also search/read the inbox when asked. Keep all summaries short, direct, and meaningful."""


def github_server():
    if not GITHUB_TOKEN:
        raise RuntimeError("GITHUB_PERSONAL_ACCESS_TOKEN not set in .env")
    return {"github": {
        "transport": "stdio",
        "command": "npx",
        "args": ["-y", "@modelcontextprotocol/server-github"],
        "env": {**os.environ, "GITHUB_PERSONAL_ACCESS_TOKEN": GITHUB_TOKEN},
    }}


def calendar_server():
    creds = GOOGLE_OAUTH_CREDENTIALS
    if not creds and os.path.exists("gcp-oauth.keys.json"):
        creds = os.path.abspath("gcp-oauth.keys.json")
    if not creds:
        raise RuntimeError("GOOGLE_OAUTH_CREDENTIALS not set in .env (or gcp-oauth.keys.json missing)")
    if not os.path.exists(creds):
        raise RuntimeError(f"Credentials file not found at: {creds}")

    token_paths = [
        os.environ.get("GOOGLE_CALENDAR_MCP_TOKEN_PATH", ""),
        os.path.expanduser("~/.config/google-calendar-mcp/tokens.json"),
        os.path.join(os.environ.get("APPDATA", ""), "google-calendar-mcp", "tokens.json"),
        os.path.abspath(".gcp-saved-tokens.json"),
    ]
    if not any(p and os.path.exists(p) for p in token_paths):
        raise RuntimeError('Calendar not authorized yet. Run: node "$env:APPDATA\\npm\\node_modules\\@cocal\\google-calendar-mcp\\build\\auth-server.js"')

    return {"google_calendar": {
        "transport": "stdio",
        "command": "npx",
        "args": ["-y", "@cocal/google-calendar-mcp"],
        "env": {**os.environ, "GOOGLE_OAUTH_CREDENTIALS": os.path.abspath(creds)},
    }}


def gmail_server():
    # Uses ~/.gmail-mcp/gcp-oauth.keys.json + credentials created by the `auth` step (see README)
    gmail_mcp_path = os.path.expanduser("~/.gmail-mcp/gcp-oauth.keys.json")
    local_path = "gcp-oauth.keys.json"
    if not (os.path.exists(gmail_mcp_path) or os.path.exists(local_path)):
        raise RuntimeError("gcp-oauth.keys.json not found in ~/.gmail-mcp or current directory (see README)")

    gmail_creds = os.path.expanduser("~/.gmail-mcp/credentials.json")
    if not os.path.exists(gmail_creds):
        raise RuntimeError("Gmail not authorized yet. Run: npx @gongrzhe/server-gmail-autoauth-mcp auth")

    return {"gmail": {
        "transport": "stdio",
        "command": "npx",
        "args": ["-y", "@gongrzhe/server-gmail-autoauth-mcp"],
        "env": {**os.environ},
        "cwd": os.path.dirname(os.path.abspath(__file__)),
    }}


async def load_github_tools():
    return await MultiServerMCPClient(github_server()).get_tools()


async def load_calendar_tools():
    return await MultiServerMCPClient(calendar_server()).get_tools()


async def load_email_tools():
    tools = await MultiServerMCPClient(gmail_server()).get_tools()
    send = next((t for t in tools if "send" in t.name.lower() and "email" in t.name.lower()), None)
    if send is None:
        raise RuntimeError(f"Gmail MCP has no send_email tool; found: {[t.name for t in tools]}")
    return tools + build_scheduler_tools(send)
