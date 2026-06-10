"""Daily news brief — Phase 1.

Fetches recent articles from Yahoo Finance and TechCrunch RSS feeds,
filters to the last 24 hours, and asks Claude to summarize them into
a structured brief printed to the terminal.
"""

import base64
import json
import os
import sys
from collections import Counter
from datetime import datetime, timedelta, timezone
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.utils import parseaddr, parsedate_to_datetime

import feedparser
import html2text
import markdown as md
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
GMAIL_SCOPES = [
    "https://www.googleapis.com/auth/gmail.readonly",
    "https://www.googleapis.com/auth/gmail.send",
]
GMAIL_LABEL = "Newsletters"
GMAIL_QUERY = f"label:{GMAIL_LABEL} newer_than:1d"
GMAIL_CREDENTIALS_FILE = "credentials.json"
GMAIL_TOKEN_FILE = "token.json"
GMAIL_USER_EMAIL = os.getenv("GMAIL_USER_EMAIL", "loganmatthewkim@gmail.com")
DETECTED_CLOUD           = "GITHUB_ACTIONS" in os.environ
_GOOGLE_CREDENTIALS_JSON = os.environ.get("GOOGLE_CREDENTIALS_JSON")
_GMAIL_TOKEN_JSON        = os.environ.get("GMAIL_TOKEN_JSON")
NEWSLETTER_BODY_CAP = 4000

# ---------------------------------------------------------------------------
# SUMMARIZATION_PROMPT — edit freely to tune the brief's voice and structure.
# ---------------------------------------------------------------------------
SUMMARIZATION_PROMPT = """
You are writing a daily intelligence brief for ONE specific reader. Internalize who he is before you write a word:

READER PROFILE
- UC Irvine finance sophomore (3.92 GPA), CS minor, headed to Centerview Partners Menlo Park Technology Group as an IB Summer Analyst in 2027
- Long-term target: growth equity, then venture capital
- Reads Stratechery, Pro Rata, SemiAnalysis, Bessemer/a16z content, Khosla, Sequoia
- Already knows: dual-class shares, ARR vs revenue, dilution mechanics, multiples, capital stack, term sheet basics, common VC strategies, hyperscaler dynamics, basic macro (Fed, yields, BOJ), software business models
- He does NOT need you to explain these. Spend your words on insight, not 101-level definitions.
- He is bright but pre-professional. Write to him as a peer analyst, not as a beginner or a layperson.

FACTUAL DISCIPLINE -- strict, non-negotiable:

You will be given source items from RSS feeds and email newsletters. Each item has a title, source, description, and a content snippet. You may ONLY cite facts that are explicitly present in those source items.

- Do NOT invent specific numbers, dates, percentages, dollar amounts, ticker prices, or growth rates that are not in the source.
- Do NOT invent company ownership relationships, executive names, or competitor lists. If you don't know who owns a brand or who a competitor is from the source, don't say it.
- Do NOT invent historical comparisons or precedents unless they are well-known canonical facts that any analyst would accept without sourcing (e.g., 'AWS launched in 2006' is fine; 'AWS hit $X revenue in 2014' requires the number to be either in the source OR you must omit it).
- When the source gives a range or an approximation, preserve that -- don't tighten an 'about $10B' into 'exactly $9.8B.'
- If a 'So what' line requires comparison to a benchmark you don't have confident knowledge of, either omit the comparison or use a hedge ('comparable in scale to past Meta capex cycles' rather than 'exactly matches Meta's 2022 capex of $35B').
- Better to be vague-but-correct than precise-but-wrong. Your reader will catch wrong claims and lose trust permanently.

If you find yourself reaching for a stat to make a sentence punchier and you can't ground it in the source or in canonical common knowledge, REWRITE the sentence without the stat or CUT the item.

WRITING RULES (strict — violations = bad brief)

1. SPECIFIC NUMBERS OR DROP IT. Every item must contain at least one concrete figure: dollar amount, percentage, date, ticker price, basis point move, share count, growth rate, market share, multiple, etc. If the source doesn't provide a number worth citing, the item probably isn't material — drop it or replace it.

2. COMPARE TO A BENCHMARK. "Why it matters" lines must compare to something — a precedent (e.g., "Zuck's voting structure at FB IPO was 58%; Musk is locking in >50% with no sunset"), a peer (e.g., "Cursor at $XB vs Cognition at $YB on similar ARR"), or a historical base rate (e.g., "down rounds at this stage historically signal a 30-50% reset"). Numbers alone aren't analysis — relative position is.

3. NAME SPECIFIC AFFECTED ENTITIES. Don't say "the sector," "competitors," "the industry." Name 2-3 specific tickers, companies, funds, or people whose situation changes because of this news. E.g., not "AV competitors gain ammo" but "Cruise, Zoox, and Tesla Robotaxi all benefit from Waymo's stumble."

4. BANNED WORDS — never use any of these. They are filler that signals you have nothing to say: "key," "important," "significant," "notable," "compelling," "meaningful," "material" (unless quantified), "noteworthy," "worth watching" (without specifying what specifically), "interesting," "robust," "strategic" (without object), "directionally," "thesis-level."
   If you'd use one of these, you don't have the underlying insight yet. Either dig deeper or drop the item.

5. NO RESTATING THE HEADLINE. If your "what happened" line just rewords the headline, you've added nothing. Add a fact the headline omits — a number, a counterparty, a precedent, a context.

6. IF YOU CAN'T ADD INFORMATION, CUT THE ITEM. Better to publish 8 sharp items than 16 padded ones. Quality over quantity. Default to fewer items.

7. CONCRETE > ABSTRACT. "Watch for governance discount of 5-15% in the IPO range" beats "Watch for governance discount." Always commit to a specific magnitude, direction, or window.

OUTPUT FORMAT -- strict:
Each item must be formatted in markdown EXACTLY like this, with literal blank lines between sections:

**[Headline]** | *[Source]*

**What:** [one sentence with at least one specific number from the source]

**So what:** [one sentence with a comparison or named affected entity]

**Watch:** [short phrase -- specific data point or threshold]

The blank lines between What/So what/Watch must be present in your raw markdown output. Do not collapse into one paragraph.

OUTPUT SECTIONS (4 total, in this order)

## 🎯 Top 3 Stories
The 3 most important items today by impact + novelty + relevance to this reader. NOT the items with the biggest dollar numbers — the ones that most change how he should think about a market, company, or trend. NEWSLETTER ITEMS GET WEIGHTED 2x in selection because they are pre-curated. These three items appear ONLY here, never repeated below.

## 📈 Markets & Macro
Items affecting public markets, macro, credit, monetary policy, major company moves. Max 6 items.

## 🚀 Tech, Investing, Venture, Startups
Funding rounds, M&A, founder moves, AI/regulation, big tech strategy, fund launches, exits. Max 6 items.

## 💡 Other Interesting
Second-order signals, contrarian framings, cross-industry implications, things that don't fit elsewhere. Max 3 items. Omit the section entirely if you have nothing that meets the writing rules.

FINAL CHECK BEFORE SUBMITTING
Before you output, re-read each bullet and ask:
- Does it contain a specific number?
- Does it compare to something a sharp reader can recognize?
- Are any banned words still in there? (If yes, rewrite or cut.)
- Could a Bloomberg terminal also tell me this? If your output is generic enough that mainstream financial media already covered it the same way, you are failing this reader.

Total output should not exceed roughly 2500 words. Keep items tight. Better to publish 12 sharp items than 18 padded ones running into a length cap.

BEFORE OUTPUTTING: scan every bullet for the banned words list above ('key,' 'important,' 'significant,' 'notable,' 'compelling,' 'meaningful,' 'material' unless quantified, 'noteworthy,' 'worth watching,' 'interesting,' 'robust,' 'strategic,' 'directionally,' 'thesis-level'). If any remain, rewrite or cut. Hard rule, no exceptions.

Now write the brief.
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
    """Return valid Gmail credentials. Supports both file (local) and env-var (cloud) modes."""
    creds = None

    # ── Load token ────────────────────────────────────────────────────────
    if _GMAIL_TOKEN_JSON:
        try:
            _td = json.loads(_GMAIL_TOKEN_JSON)
            _scopes_raw = _td.get("scopes") or []
            if isinstance(_scopes_raw, str):
                _ts = set(_scopes_raw.split())
            else:
                _ts = set(_scopes_raw)
            if not set(GMAIL_SCOPES).issubset(_ts):
                print(f"  ! GMAIL_TOKEN_JSON missing scopes {set(GMAIL_SCOPES) - _ts}; re-running OAuth.")
            else:
                creds = Credentials.from_authorized_user_info(_td, GMAIL_SCOPES)
        except Exception as e:
            print(f"  ! Could not parse GMAIL_TOKEN_JSON ({e}); will re-run OAuth.")
            creds = None
    elif os.path.exists(GMAIL_TOKEN_FILE):
        try:
            with open(GMAIL_TOKEN_FILE) as _tf:
                _token_data = json.load(_tf)
            _scopes_raw2 = _token_data.get("scopes") or []
            if isinstance(_scopes_raw2, str):
                _token_scopes = set(_scopes_raw2.split())
            else:
                _token_scopes = set(_scopes_raw2)
            if not set(GMAIL_SCOPES).issubset(_token_scopes):
                print(f"  ! token.json missing required scopes {set(GMAIL_SCOPES) - _token_scopes}; re-running OAuth.")
            else:
                creds = Credentials.from_authorized_user_file(GMAIL_TOKEN_FILE, GMAIL_SCOPES)
        except Exception as e:
            print(f"  ! Could not load {GMAIL_TOKEN_FILE} ({e}); will re-run OAuth.")
            creds = None

    if creds and creds.valid:
        return creds

    if creds and creds.expired and creds.refresh_token:
        try:
            creds.refresh(Request())
            if _GMAIL_TOKEN_JSON:
                print("  ! Token refreshed in env-var mode — not persisted. Update GMAIL_TOKEN_JSON secret before it expires.")
            else:
                with open(GMAIL_TOKEN_FILE, "w") as f:
                    f.write(creds.to_json())
            return creds
        except Exception as e:
            print(f"  ! Token refresh failed ({e}); re-running OAuth flow.")

    # ── Build OAuth flow ──────────────────────────────────────────────────
    if DETECTED_CLOUD:
        print(
            "ERROR: GMAIL_TOKEN_JSON is invalid or expired. "
            "Re-run locally to refresh the token, then update the GMAIL_TOKEN_JSON GitHub Secret.",
            file=sys.stderr,
        )
        sys.exit(1)

    if _GOOGLE_CREDENTIALS_JSON:
        try:
            flow = InstalledAppFlow.from_client_config(
                json.loads(_GOOGLE_CREDENTIALS_JSON), GMAIL_SCOPES
            )
        except Exception as e:
            print(f"ERROR: Could not parse GOOGLE_CREDENTIALS_JSON ({e})", file=sys.stderr)
            sys.exit(1)
    else:
        if not os.path.exists(GMAIL_CREDENTIALS_FILE):
            print(
                f"ERROR: {GMAIL_CREDENTIALS_FILE} not found. Download it from Google Cloud Console.",
                file=sys.stderr,
            )
            sys.exit(1)
        flow = InstalledAppFlow.from_client_secrets_file(GMAIL_CREDENTIALS_FILE, GMAIL_SCOPES)

    flow.redirect_uri = "urn:ietf:wg:oauth:2.0:oob"
    auth_url, _ = flow.authorization_url(prompt="consent")
    bar = "=" * 72
    print(f"\n{bar}")
    print("OAUTH CONSENT REQUIRED -- copy this URL into your Windows browser:\n")
    print(f"    {auth_url}\n")
    print("After clicking Allow, Google will show you a CODE on the page.")
    print("Copy that CODE (not a URL) and paste it back here.")
    print(f"{bar}\n")
    code = input("Paste the authorization code here: ").strip()
    flow.fetch_token(code=code)
    creds = flow.credentials
    if not _GMAIL_TOKEN_JSON:
        with open(GMAIL_TOKEN_FILE, "w") as f:
            f.write(creds.to_json())
        print(f"  OAuth complete; {GMAIL_TOKEN_FILE} saved for future runs.")
    else:
        print("  OAuth complete (env-var mode; token not written to file).")
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



# ---------------------------------------------------------------------------
# Email delivery.
# ---------------------------------------------------------------------------

_EMAIL_HTML = (
    '<div style="font-family:-apple-system,BlinkMacSystemFont,Helvetica,Arial,sans-serif;'
    'max-width:640px;margin:0 auto;color:#1a1a1a;line-height:1.7;background:#fff;">'
    '<div style="background:#0f172a;padding:24px 32px;border-radius:8px 8px 0 0;">'
    '<h1 style="color:#fff;margin:0;font-size:20px;font-weight:700;letter-spacing:-0.3px;">Daily Brief</h1>'
    '<p style="color:#64748b;margin:6px 0 0;font-size:13px;">__DATE_STR__&nbsp;&middot;&nbsp;__TIME_STR__</p>'
    '</div>'
    '<div style="padding:28px 32px;border:1px solid #e2e8f0;border-top:none;border-radius:0 0 8px 8px;">'
    '<style>'
    'h2{font-size:17px;font-weight:700;margin:28px 0 10px;padding-bottom:8px;'
    'border-bottom:2px solid #e2e8f0;color:#0f172a;line-height:1.3;}'
    'ul{margin:8px 0;padding-left:20px;}'
    'li{margin:6px 0;}'
    'strong{color:#0f172a;}'
    'em{color:#475569;}'
    'p{margin:8px 0;}'
    'blockquote{border-left:3px solid #e2e8f0;margin:12px 0;padding:8px 16px;'
    'color:#475569;font-style:italic;background:#f8fafc;border-radius:0 4px 4px 0;}'
    '</style>'
    '__HTML_BODY__'
    '</div>'
    '<div style="padding:14px 32px;text-align:center;">'
    '<p style="color:#94a3b8;font-size:11px;margin:0;">'
    'Your AI briefing&nbsp;&middot;&nbsp;Sources: MarketWatch, Seeking Alpha, TechCrunch, Gmail newsletters'
    '</p></div></div>'
)



def send_brief_email(brief: str) -> bool:
    """Send the daily brief as plain text + HTML via Gmail API."""
    now = datetime.now()
    date_str = now.strftime(f"%A, %B {now.day}")
    time_str = now.strftime("%I:%M %p")

    html_body = md.markdown(brief, extensions=["extra", "nl2br"])
    full_html = _EMAIL_HTML.replace("__DATE_STR__", date_str).replace("__TIME_STR__", time_str).replace("__HTML_BODY__", html_body)

    msg = MIMEMultipart("alternative")
    msg["Subject"] = f"Daily Brief — {date_str}"
    msg["From"] = GMAIL_USER_EMAIL
    msg["To"] = GMAIL_USER_EMAIL
    msg.attach(MIMEText(brief, "plain", "utf-8"))
    msg.attach(MIMEText(full_html, "html", "utf-8"))

    raw = base64.urlsafe_b64encode(msg.as_bytes()).decode()

    try:
        creds = _get_gmail_credentials()
        service = build("gmail", "v1", credentials=creds, cache_discovery=False)
        service.users().messages().send(userId="me", body={"raw": raw}).execute()
        return True
    except HttpError as e:
        print(f"  ! Gmail API error sending email: {e}")
        return False
    except Exception as e:
        print(f"  ! Failed to send email: {e}")
        return False


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
        max_tokens=8000,
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
    print(f"Environment: {'cloud' if DETECTED_CLOUD else 'local'}")
    print(f"Credentials source: {'env var' if _GOOGLE_CREDENTIALS_JSON else 'file'}")
    print(f"Token source: {'env var' if _GMAIL_TOKEN_JSON else 'file'}")
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

    print("Sending email...")
    if send_brief_email(brief):
        print(f"  Sent to {GMAIL_USER_EMAIL}")
    else:
        print("  Email send failed â see error above. Brief printed above is complete.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
