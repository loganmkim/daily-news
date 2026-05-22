"""Daily news brief — Phase 1.

Fetches recent articles from Yahoo Finance and TechCrunch RSS feeds,
filters to the last 24 hours, and asks Claude to summarize them into
a structured brief printed to the terminal.
"""

import base64
import os
import sys
from collections import Counter
from datetime import datetime, timedelta, timezone
from email.utils import parseaddr, parsedate_to_datetime

import feedparser
import html2text
import pytz
from anthropic import Anthropic
from dotenv import load_dotenv
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

# Real environment variables take priority over .env values.
load_dotenv(override=False)

MODEL = "claude-sonnet-4-6"
LOOKBACK_HOURS = 24
MAX_ARTICLES_TO_CLAUDE = 30

LOW_SIGNAL_PATTERNS = [
    "how to",
    "best of",
    "top 10",
    "top 5",
    "sponsored",
    "presented by",
    "guide to",
    "everything you need to know",
]

FEEDS = [
    ("MarketWatch", "https://feeds.content.dowjones.io/public/rss/mw_topstories"),
    ("Seeking Alpha", "https://seekingalpha.com/market_currents.xml"),
    ("TechCrunch", "https://techcrunch.com/feed/"),
]

# Gmail integration — read-only access to a single Gmail label.
GMAIL_SCOPES = ["https://www.googleapis.com/auth/gmail.readonly"]
GMAIL_LABEL = "Newsletters"
GMAIL_QUERY = f"label:{GMAIL_LABEL} newer_than:1d"
GMAIL_CREDENTIALS_FILE = "credentials.json"
GMAIL_TOKEN_FILE = "token.json"
NEWSLETTER_BODY_CAP = 4000

# ---------------------------------------------------------------------------
# SUMMARIZATION_PROMPT — edit freely to tune the brief's voice and structure.
# ---------------------------------------------------------------------------
SUMMARIZATION_PROMPT = """ABSOLUTE RULES — READ FIRST. Violating either of these makes the brief unusable:

1. NO DUPLICATION ACROSS SECTIONS. Every story appears in EXACTLY ONE section. If a story is in "Top 3 Stories of the Day", it MUST NOT appear in any other section. The Markets, Tech, and Other sections must explicitly skip the Top 3 headlines — do not re-summarize them, do not reference them, do not include them.

2. SIGNIFICANCE IS NOT THE SAME AS DOLLAR SIZE. Do not equate the biggest dollar figure with the most important story. A $16M seed from a16z to a notable founder can be more significant than a $1B late-stage round if the seed signals a new thesis.

---

NOTE ON INPUT: Some items are excerpts from newsletters (source begins with 'Newsletter:'). These are typically pre-curated by editors — weight their signal highly when picking Top 3 stories. When summarizing them, distill the editorial point, not just the news event.

---

You are a sharp markets and tech analyst writing a daily intelligence brief for an analyst-investor (Centerview IB → tech investing track). Tone: direct, opinionated when warranted, no filler.

You will receive a set of news items from RSS feeds. Each item has a title, source, description, and timestamp.

Produce a brief in exactly these 4 sections, in markdown:

## 🎯 Top 3 Stories of the Day
Pick the 3 most SIGNIFICANT stories across ALL sources — meaning the ones that change how a sharp investor or operator should think about a market, company, or trend tomorrow.

Significance is NOT the same as size of dollar number. A $16M seed from a16z to a notable founder can be more significant than a $1B late-stage round if the seed signals a new thesis. Lean toward stories with second-order implications, narrative shifts, or genuine surprise. Avoid defaulting to whichever stories have the biggest numbers attached. A small but unusual move (a non-obvious investor entering a category, a strategic pivot, a regulatory shift, a notable founder reappearing) often beats a large but predictable one (another mega-round to an already-well-capitalized incumbent).

For each: bold headline, source in italics, 1-sentence "why it matters" framed in market/strategic terms.

REMINDER: These three items MUST NOT appear in any section below. Sections 2–4 must skip these headlines entirely.

## 📈 Markets & Macro
Items relevant to public markets, macro, monetary policy, major company moves — EXCLUDING anything already in Top 3. For each: headline, source in italics, 1–2 sentence "what + why." Skip generic earnings recaps and analyst rating changes UNLESS they reflect a material thesis shift. Order most-important first. If nothing notable, write "Nothing material today."

## 🚀 Tech, Investing, Venture, Startups
Funding rounds, M&A, founder moves, product launches that move a category, AI/regulation, big tech strategy — EXCLUDING anything already in Top 3. For each: headline, source in italics, 1–2 sentence takeaway. Order most-important first.

## 💡 Other Interesting
Anything notable that doesn't fit above — cultural shifts, second-order effects, contrarian signals — EXCLUDING anything already in Top 3. Be selective; if nothing fits, omit this section entirely.

Rules:
- Do NOT include items that are pure SEO bait, listicles, podcast episode announcements, or thinly-disguised vendor PR.
- Do NOT pad sections to hit a number — if there's nothing material, say so.
- Each item gets one bullet — no nested bullets, no preambles like "In other news".
- Always include the source in italics next to the headline.
- Before writing each section after Top 3, mentally check: "Did I already cover any of these in Top 3? If yes, skip them here." If you catch yourself about to repeat a Top 3 story, drop it.
"""


def _parse_entry_dt(entry):
    """Best-effort parse of an entry's publish time → tz-aware UTC datetime, or None."""
    ts = entry.get("published_parsed") or entry.get("updated_parsed")
    if ts:
        return datetime(*ts[:6], tzinfo=pytz.UTC)
    for field in ("published", "updated", "pubDate"):
        raw = entry.get(field, "")
        if not raw:
            continue
        try:
            dt = parsedate_to_datetime(raw)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=pytz.UTC)
            return dt.astimezone(pytz.UTC)
        except (TypeError, ValueError):
            continue
    return None


def fetch_feed(name: str, url: str) -> list[dict]:
    """Fetch and parse a single RSS feed. Returns a list of normalized entries."""
    print(f"Fetching {name}...")
    parsed = feedparser.parse(url)
    if parsed.bozo and not parsed.entries:
        print(f"  ! Failed to parse {name}: {parsed.bozo_exception}")
        return []
    entries = []
    for e in parsed.entries:
        entries.append({
            "source": name,
            "title": (e.get("title") or "").strip(),
            "link": e.get("link") or "",
            "summary": (e.get("summary") or "").strip(),
            "published_dt": _parse_entry_dt(e),
            "_raw_dates": {
                "published": e.get("published", ""),
                "updated": e.get("updated", ""),
                "pubDate": e.get("pubDate", ""),
                "published_parsed": e.get("published_parsed"),
                "updated_parsed": e.get("updated_parsed"),
            },
        })
    print(f"  Found {len(entries)} entries in feed")
    return entries


def filter_recent(entries: list[dict], hours: int = LOOKBACK_HOURS) -> list[dict]:
    """Keep only entries published within the last `hours` hours (UTC-aware)."""
    cutoff = datetime.now(pytz.UTC) - timedelta(hours=hours)
    return [e for e in entries if e["published_dt"] and e["published_dt"] >= cutoff]


def dedupe(entries: list[dict]) -> list[dict]:
    """Remove cross-source duplicates by URL OR normalized title.

    Different sources cover the same story under different URLs, so a URL-only
    key would miss those. We drop an entry if EITHER its URL or its normalized
    title has already been seen.
    """
    seen_urls: set[str] = set()
    seen_titles: set[str] = set()
    out = []
    for e in entries:
        url = e["link"]
        title_norm = e["title"].lower().strip()
        if url and url in seen_urls:
            continue
        if title_norm and title_norm in seen_titles:
            continue
        if url:
            seen_urls.add(url)
        if title_norm:
            seen_titles.add(title_norm)
        out.append(e)
    return out


def drop_low_signal(entries: list[dict]) -> list[dict]:
    """Drop entries whose title contains a low-signal pattern (case-insensitive)."""
    return [
        e for e in entries
        if not any(p in e["title"].lower() for p in LOW_SIGNAL_PATTERNS)
    ]


def cap_to_max(entries: list[dict], n: int = MAX_ARTICLES_TO_CLAUDE) -> list[dict]:
    """Sort by published date descending and keep only the top `n`."""
    return sorted(entries, key=lambda e: e["published_dt"], reverse=True)[:n]


def diagnose_zero_recent(all_entries: list[dict], recent_by_source: Counter) -> None:
    """For any source with >0 fetched but 0 passing the 24h filter, dump first 3 entries."""
    by_source_fetched = Counter(a["source"] for a in all_entries)
    now_utc = datetime.now(pytz.UTC)
    for source, fetched_count in by_source_fetched.items():
        if fetched_count > 0 and recent_by_source.get(source, 0) == 0:
            print(f"\n[debug] {source}: {fetched_count} fetched, 0 passed 24h filter. First 3 entries:")
            sample = [a for a in all_entries if a["source"] == source][:3]
            for i, e in enumerate(sample, 1):
                print(f"  [{i}] {e['title'][:80]}")
                print(f"      raw published: {e['_raw_dates']['published']!r}")
                print(f"      raw pubDate:   {e['_raw_dates']['pubDate']!r}")
                print(f"      parsed dt:     {e['published_dt']}")
                if e["published_dt"]:
                    print(f"      age:           {now_utc - e['published_dt']}")


# ---------------------------------------------------------------------------
# Gmail integration (read-only).
# ---------------------------------------------------------------------------

def _get_gmail_credentials():
    """Return valid Gmail credentials. Triggers OAuth browser flow on first run."""
    creds = None
    if os.path.exists(GMAIL_TOKEN_FILE):
        try:
            creds = Credentials.from_authorized_user_file(GMAIL_TOKEN_FILE, GMAIL_SCOPES)
        except Exception as e:
            print(f"  ! Could not load {GMAIL_TOKEN_FILE} ({e}); will re-run OAuth.")
            creds = None

    if creds and creds.valid:
        return creds

    if creds and creds.expired and creds.refresh_token:
        try:
            creds.refresh(Request())
            with open(GMAIL_TOKEN_FILE, "w") as f:
                f.write(creds.to_json())
            return creds
        except Exception as e:
            print(f"  ! Token refresh failed ({e}); re-running OAuth flow.")

    if not os.path.exists(GMAIL_CREDENTIALS_FILE):
        print(
            f"ERROR: {GMAIL_CREDENTIALS_FILE} not found. Download it from Google Cloud Console.",
            file=sys.stderr,
        )
        sys.exit(1)

    flow = InstalledAppFlow.from_client_secrets_file(GMAIL_CREDENTIALS_FILE, GMAIL_SCOPES)
    bar = "=" * 72
    prompt = (
        f"\n{bar}\n"
        "OAUTH CONSENT REQUIRED — copy this URL into your Windows browser:\n\n"
        "    {url}\n\n"
        f"After you approve, the page will redirect to a localhost URL —\n"
        f"that's the local callback server completing the flow. You can then\n"
        f"close that tab.\n"
        f"{bar}\n"
    )
    creds = flow.run_local_server(
        port=0,
        open_browser=False,
        authorization_prompt_message=prompt,
        success_message="Auth complete. You can close this browser tab.",
    )
    with open(GMAIL_TOKEN_FILE, "w") as f:
        f.write(creds.to_json())
    print(f"  OAuth complete; {GMAIL_TOKEN_FILE} saved for future runs.")
    return creds


def _decode_b64url(data: str) -> bytes:
    """Gmail API returns body data as URL-safe base64, sometimes without padding."""
    if not data:
        return b""
    padding = (-len(data)) % 4
    return base64.urlsafe_b64decode(data + "=" * padding)


def _extract_plain_body(payload: dict) -> str | None:
    """Walk a Gmail message MIME tree and return a plain-text body, or None."""
    plain_chunks: list[str] = []
    html_chunks: list[str] = []

    stack = [payload]
    while stack:
        node = stack.pop()
        mime = node.get("mimeType", "")
        body = node.get("body") or {}
        data = body.get("data")
        if data:
            try:
                text = _decode_b64url(data).decode("utf-8", errors="replace")
                if mime == "text/plain":
                    plain_chunks.append(text)
                elif mime == "text/html":
                    html_chunks.append(text)
            except Exception:
                pass
        for part in node.get("parts", []) or []:
            stack.append(part)

    if plain_chunks:
        return "\n\n".join(plain_chunks).strip()
    if html_chunks:
        h = html2text.HTML2Text()
        h.ignore_images = True
        h.body_width = 0  # don't hard-wrap
        return h.handle("\n\n".join(html_chunks)).strip()
    return None


def _truncate_clean(text: str, cap: int = NEWSLETTER_BODY_CAP) -> str:
    """Truncate to `cap` chars at a paragraph boundary if possible, else hard cut."""
    if len(text) <= cap:
        return text
    paragraphs = text.split("\n\n")
    out: list[str] = []
    used = 0
    for p in paragraphs:
        added = len(p) + (2 if out else 0)
        if used + added > cap:
            break
        out.append(p)
        used += added
    if out:
        return "\n\n".join(out) + "\n\n[...truncated]"
    return text[:cap].rstrip() + "…"


def _sender_domain(from_header: str) -> str:
    """Pull the domain out of an email From header. Returns 'unknown' on failure."""
    _, addr = parseaddr(from_header or "")
    if "@" not in addr:
        return "unknown"
    domain = addr.split("@", 1)[1].lower().strip()
    return domain or "unknown"


def fetch_gmail_newsletters() -> list[dict]:
    """Fetch newsletters from the last 24h. Returns entries in the same shape as RSS."""
    print(f"Fetching Gmail (label:{GMAIL_LABEL})...")
    try:
        creds = _get_gmail_credentials()
        service = build("gmail", "v1", credentials=creds, cache_discovery=False)

        message_ids: list[str] = []
        page_token = None
        while True:
            resp = service.users().messages().list(
                userId="me",
                q=GMAIL_QUERY,
                pageToken=page_token,
                maxResults=100,
            ).execute()
            for m in resp.get("messages", []):
                message_ids.append(m["id"])
            page_token = resp.get("nextPageToken")
            if not page_token:
                break

        if not message_ids:
            print("  Found 0 newsletter messages")
            return []

        entries: list[dict] = []
        for mid in message_ids:
            try:
                msg = service.users().messages().get(
                    userId="me", id=mid, format="full"
                ).execute()
            except HttpError as e:
                print(f"  ! Failed to fetch message {mid}: {e}")
                continue

            payload = msg.get("payload", {}) or {}
            headers = {h["name"].lower(): h["value"] for h in payload.get("headers", [])}
            sender = headers.get("from", "")
            subject = (headers.get("subject", "(no subject)") or "(no subject)").strip()
            domain = _sender_domain(sender)

            body = _extract_plain_body(payload)
            if not body:
                print(f"  ! Skipping {mid}: body extraction failed (subject: {subject[:60]!r})")
                continue
            body = _truncate_clean(body)

            internal_ms = int(msg.get("internalDate", "0"))
            published_dt = (
                datetime.fromtimestamp(internal_ms / 1000, tz=pytz.UTC)
                if internal_ms else None
            )

            entries.append({
                "source": f"Newsletter: {domain}",
                "title": subject,
                "link": "",
                "summary": body,
                "published_dt": published_dt,
                "_raw_dates": {
                    "published": "",
                    "updated": "",
                    "pubDate": "",
                    "published_parsed": None,
                    "updated_parsed": None,
                    "internalDate": internal_ms,
                },
            })

        print(f"  Found {len(entries)} newsletter messages (out of {len(message_ids)} ids)")
        return entries

    except HttpError as e:
        print(f"  ! Gmail API error: {e}")
        return []


def format_for_claude(articles: list[dict]) -> str:
    """Render the filtered articles into a plain-text block for the user message."""
    lines = []
    for i, a in enumerate(articles, 1):
        published = a["published_dt"].strftime("%Y-%m-%d %H:%M UTC")
        lines.append(f"[{i}] {a['title']}")
        lines.append(f"    Source: {a['source']}")
        lines.append(f"    Published: {published}")
        if a["summary"]:
            lines.append(f"    Summary: {a['summary']}")
        lines.append("")
    return "\n".join(lines)


def summarize(articles: list[dict]) -> str:
    """Call Claude to produce the structured brief."""
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("ERROR: ANTHROPIC_API_KEY is not set (checked env and .env).", file=sys.stderr)
        sys.exit(1)

    client = Anthropic(api_key=api_key)
    payload = format_for_claude(articles)

    print(f"Calling Claude ({MODEL})...")
    resp = client.messages.create(
        model=MODEL,
        max_tokens=2048,
        system=SUMMARIZATION_PROMPT,
        messages=[{
            "role": "user",
            "content": (
                f"Today is {datetime.now(timezone.utc).strftime('%Y-%m-%d')} (UTC). "
                f"Here are {len(articles)} articles from the last {LOOKBACK_HOURS} hours:\n\n"
                f"{payload}"
            ),
        }],
    )

    parts = [b.text for b in resp.content if getattr(b, "type", None) == "text"]
    return "\n".join(parts).strip()


def main() -> int:
    all_entries: list[dict] = []
    for name, url in FEEDS:
        all_entries.extend(fetch_feed(name, url))

    all_entries.extend(fetch_gmail_newsletters())

    fetched_total = len(all_entries)
    by_source_fetched = Counter(a["source"] for a in all_entries)
    print(f"\nPer-source fetch counts: {dict(by_source_fetched)}")
    newsletter_total = sum(c for s, c in by_source_fetched.items() if s.startswith("Newsletter:"))
    print(f"Newsletters: {newsletter_total} in last 24h")

    deduped = dedupe(all_entries)
    print(f"Total after dedup:       {len(deduped)} (removed {fetched_total - len(deduped)} duplicates)")

    recent = filter_recent(deduped)
    by_source_recent = Counter(a["source"] for a in recent)
    print(f"After 24h filter, by source: {dict(by_source_recent)}")

    diagnose_zero_recent(deduped, by_source_recent)

    filtered = drop_low_signal(recent)
    capped = cap_to_max(filtered)

    print(f"\nFetched: {fetched_total}. After filter: {len(filtered)}. Sent to Claude: {len(capped)}.\n")

    if not capped:
        print("No articles to summarize. Exiting.")
        return 0

    brief = summarize(capped)
    print("\n" + "=" * 72)
    print("DAILY BRIEF")
    print("=" * 72 + "\n")
    print(brief)
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
