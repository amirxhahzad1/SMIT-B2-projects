"""Schedule emails to be sent later. Uses the Gmail MCP `send_email` tool at fire time.

NOTE: jobs live in this process's memory (APScheduler MemoryJobStore). If the program
exits before the send time, the email is NOT sent. For production, swap in a persistent
jobstore (SQLAlchemyJobStore) or an external queue.
"""
from datetime import datetime
from zoneinfo import ZoneInfo

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from langchain_core.tools import tool

from .config import DEFAULT_TIMEZONE

scheduler = AsyncIOScheduler()


def ensure_scheduler_started():
    if not scheduler.running:
        scheduler.start()  # must be called inside a running event loop


def build_scheduler_tools(send_tool):
    async def _send(to: str, subject: str, body: str):
        try:
            await send_tool.ainvoke({"to": [to], "subject": subject, "body": body})
            print(f"\n[scheduler] Sent scheduled email to {to}: {subject}")
        except Exception as e:
            print(f"\n[scheduler] FAILED to send scheduled email to {to}: {e}")

    @tool
    async def schedule_email(to: str, subject: str, body: str, send_at: str, timezone: str = DEFAULT_TIMEZONE) -> str:
        """Schedule an email to be sent automatically at a future time.
        `send_at` is ISO-8601 local time, e.g. '2026-10-01T09:00:00'. `timezone` is an IANA name."""
        dt = datetime.fromisoformat(send_at)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=ZoneInfo(timezone))
        if dt <= datetime.now(dt.tzinfo):
            return "Error: send_at is in the past."
        ensure_scheduler_started()
        job = scheduler.add_job(_send, "date", run_date=dt, args=[to, subject, body])
        return f"Scheduled job {job.id}: email to {to} at {dt.isoformat()}."

    @tool
    def list_scheduled_emails() -> str:
        """List pending scheduled emails."""
        jobs = scheduler.get_jobs()
        if not jobs:
            return "No scheduled emails."
        return "\n".join(f"{j.id}: to={j.args[0]} subject={j.args[1]!r} at {j.next_run_time}" for j in jobs)

    @tool
    def cancel_scheduled_email(job_id: str) -> str:
        """Cancel a scheduled email by job id."""
        try:
            scheduler.remove_job(job_id)
            return f"Cancelled {job_id}."
        except Exception as e:
            return f"Could not cancel: {e}"

    return [schedule_email, list_scheduled_emails, cancel_scheduled_email]
