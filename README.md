# LangGraph Multi-Agent Assistant

Supervisor graph routing to four sub-agents:

| Agent | Backend | Can do |
|---|---|---|
| `rag_agent` | Qdrant Cloud (hosted) + local FastEmbed embeddings | Ingest any PDF, answer from it with file/page citations |
| `github_agent` | Official GitHub MCP server (remote, streamable HTTP) | Any repo/issue/PR/code/commit question |
| `calendar_agent` | Google Calendar MCP server (stdio via npx) | Read meetings, check free/busy, create/update/delete events |
| `email_agent` | Gmail MCP server (stdio via npx) + local scheduler | Write, draft, send, search, and schedule emails |

```
user -> supervisor -> {rag | github | calendar | email} -> supervisor -> ... -> final answer
```
The supervisor can chain agents ("check my Friday meetings and email Sara a summary").
An agent whose credentials are missing is disabled at startup; the rest still work.

## Setup
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env      # fill in keys
```
Requires Node.js (for `npx`).

### Pick your model provider (Claude, OpenAI, or Gemini)
Everything in the app calls one function, `get_llm()` in `agent/llm.py`, which reads
`LLM_PROVIDER` from `.env` and returns the matching LangChain chat model. Nothing else
in the code changes between providers — same tools, same graph, same prompts.

```bash
# .env
LLM_PROVIDER=anthropic   # or: openai | gemini
LLM_MODEL=               # optional, e.g. gpt-4.1 / gemini-2.5-pro / claude-sonnet-5
ANTHROPIC_API_KEY=...    # set only the key matching your provider
OPENAI_API_KEY=...
GOOGLE_API_KEY=...       # from aistudio.google.com/apikey
```

### 1. Qdrant Cloud (RAG)
Create a free cluster at cloud.qdrant.io, put URL + API key in `.env`. The collection is created automatically.
```bash
python -m scripts.ingest_pdf ~/docs/paper.pdf     # or just ask the agent to ingest a path
```

### 2. GitHub
Create a Personal Access Token (github.com/settings/tokens) and set `GITHUB_PERSONAL_ACCESS_TOKEN`.
Use read-only scopes if you don't want the agent to write.

### 3. Google Calendar and Gmail
In Google Cloud Console: create a project, enable **Google Calendar API** and **Gmail API**,
configure the OAuth consent screen (add yourself as a test user), and create an
**OAuth client ID of type "Desktop app"**. Download the JSON.

- Calendar: set `GOOGLE_OAUTH_CREDENTIALS=/path/to/that.json`, then authorise once:
  `GOOGLE_OAUTH_CREDENTIALS=/path/to/that.json npx @cocal/google-calendar-mcp auth`
- Gmail: save the JSON as `~/.gmail-mcp/gcp-oauth.keys.json`, then authorise once:
  `npx @gongrzhe/server-gmail-autoauth-mcp auth`

(Both are community MCP servers; check their READMEs if flags change.)

## Run
```bash
python main.py
```
Examples:
- "What does the ingested paper say about limitations?"
- "List open issues labelled bug in owner/repo and summarise them"
- "What meetings do I have tomorrow?" / "Book a 30 min call with a@b.com on Friday at 3pm"
- "Draft an email to a@b.com thanking them for the meeting"
- "Email a@b.com the agenda tomorrow at 9am"

## Notes / limitations
- Scheduled emails live in process memory: the program must still be running at send time.
  Swap APScheduler's jobstore for SQLAlchemy or a queue for durability.
- Drafts are the default unless you explicitly say "send".
- MCP tool names/args come from the servers; the scheduler expects the Gmail server's `send_email(to[], subject, body)`.
- Pinned `mcp<2`, `langchain-mcp-adapters<0.2`, `langchain-openai<1.0`, `langchain-google-genai<3.0`
  (newer releases of each need `langchain-core>=1.0`, which conflicts with the rest of the stack).
- `langgraph<1.0` is used because the code relies on `langgraph.prebuilt.create_react_agent`.
