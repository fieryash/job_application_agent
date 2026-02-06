from __future__ import annotations

from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple
from urllib.parse import parse_qs, urlparse

from .models import AutoApplyResult, Job, Profile

SUPPORTED_ATS = {
    "greenhouse",
    "lever",
    "ashby",
    "workday",
    "smartrecruiters",
    "jobvite",
    "workable",
    "icims",
    "bamboohr",
}

AGGREGATOR_DOMAINS = {
    "dice.com",
    "linkedin.com",
    "glassdoor.com",
    "ziprecruiter.com",
    "builtin.com",
    "jobright.ai",
    "virtualvocations.com",
}

APPLY_ENTRY_SELECTORS = [
    "button:has-text('Apply')",
    "a:has-text('Apply')",
    "button:has-text('Apply now')",
    "a:has-text('Apply now')",
    "button:has-text('Start application')",
    "a:has-text('Start application')",
    "button:has-text('Easy Apply')",
    "a:has-text('Easy Apply')",
    "button:has-text('Submit Resume')",
    "a:has-text('Submit Resume')",
    "button:has-text('Upload Resume')",
    "a:has-text('Upload Resume')",
    "button:has-text('Submit Application')",
    "a:has-text('Submit Application')",
    "[aria-label*='Apply']",
    "[data-test*='apply']",
]

NEXT_STEP_SELECTORS = [
    "button:has-text('Next')",
    "button:has-text('Continue')",
    "button:has-text('Save and Continue')",
    "button:has-text('Review')",
    "a:has-text('Next')",
    "a:has-text('Continue')",
]

FIRST_NAME_SELECTORS = [
    "input[name='first_name']",
    "input[name='firstname']",
    "input[id*='first']",
    "input[placeholder*='First']",
]
LAST_NAME_SELECTORS = [
    "input[name='last_name']",
    "input[name='lastname']",
    "input[id*='last']",
    "input[placeholder*='Last']",
]
EMAIL_SELECTORS = [
    "input[type='email']",
    "input[name='email']",
    "input[id*='email']",
]
PHONE_SELECTORS = [
    "input[type='tel']",
    "input[name*='phone']",
    "input[id*='phone']",
]
RESUME_SELECTORS = [
    "input[type='file']",
    "input[name*='resume']",
    "input[id*='resume']",
]


def _split_name(full_name: str) -> Tuple[str, str]:
    parts = (full_name or "").strip().split()
    if not parts:
        return "", ""
    if len(parts) == 1:
        return parts[0], ""
    return parts[0], " ".join(parts[1:])


def _source_ats(job: Job) -> Optional[str]:
    if job.source and job.source.ats:
        return job.source.ats.lower()
    return None


def _source_url(job: Job) -> Optional[str]:
    if job.source and job.source.url:
        return job.source.url
    return None


def _domain(url: str) -> str:
    host = urlparse(url).netloc.lower()
    if host.startswith("www."):
        host = host[4:]
    return host


def _looks_like_listing_page(url: str) -> bool:
    parsed = urlparse(url)
    host = _domain(url)
    path = (parsed.path or "").lower()
    query = parse_qs(parsed.query or "")

    if "viewjob" in path or "/job/" in path or "/jobs/" in path and any(ch.isdigit() for ch in path):
        return False

    if host in AGGREGATOR_DOMAINS:
        if "q-" in path or path.endswith("/jobs") or "/jobs/" in path:
            return True
        if "query" in query or "q" in query or "keyword" in query:
            return True
        if "search" in path:
            return True

    if "/jobs/q-" in path or "/jobs/search" in path:
        return True
    if "q" in query and len(query.get("q", [])) > 0:
        return True
    return False


def _fill_first(page, selectors: List[str], value: str) -> bool:
    if not value:
        return False
    for selector in selectors:
        try:
            locator = page.locator(selector).first
            if locator.count() > 0:
                input_type = (locator.get_attribute("type") or "").lower()
                if input_type in {"checkbox", "radio"}:
                    if not locator.is_checked():
                        locator.check()
                        return True
                    continue
                existing = (locator.input_value() or "").strip()
                if existing:
                    continue
                locator.fill(value)
                return True
        except Exception:
            continue
    return False


def _click_first(page, selectors: List[str]) -> bool:
    for selector in selectors:
        try:
            locator = page.locator(selector)
            count = min(locator.count(), 6)
            for idx in range(count):
                target = locator.nth(idx)
                if not target.is_visible():
                    continue
                disabled = target.get_attribute("disabled")
                aria_disabled = (target.get_attribute("aria-disabled") or "").lower()
                if disabled is not None or aria_disabled == "true":
                    continue
                target.click()
                return True
        except Exception:
            continue
    return False


def _has_any(page, selectors: List[str]) -> bool:
    for selector in selectors:
        try:
            if page.locator(selector).count() > 0:
                return True
        except Exception:
            continue
    return False


def _required_field_name(node) -> str:
    keys = ["name", "id", "aria-label", "placeholder", "type"]
    for key in keys:
        try:
            value = (node.get_attribute(key) or "").strip()
            if value:
                return value
        except Exception:
            continue
    return "required_field"


def _detect_unfilled_required_fields(page, cap: int = 12) -> List[str]:
    missing: List[str] = []
    for selector in ("input[required]", "textarea[required]", "select[required]"):
        try:
            locator = page.locator(selector)
            count = min(locator.count(), 80)
        except Exception:
            continue

        for idx in range(count):
            node = locator.nth(idx)
            try:
                if not node.is_visible():
                    continue
                input_type = (node.get_attribute("type") or "").lower()
                if input_type in {"hidden", "submit", "button", "image"}:
                    continue

                if input_type in {"checkbox", "radio"}:
                    filled = bool(node.is_checked())
                else:
                    filled = bool((node.input_value() or "").strip())

                if not filled:
                    missing.append(_required_field_name(node))
                    if len(missing) >= cap:
                        return missing
            except Exception:
                continue
    return missing


def _request_user_attention(page, message: str) -> None:
    try:
        page.bring_to_front()
    except Exception:
        pass
    try:
        page.evaluate(
            """
            (msg) => {
              try {
                const id = '__job_agent_attention_banner';
                const old = document.getElementById(id);
                if (old) old.remove();
                const bar = document.createElement('div');
                bar.id = id;
                bar.textContent = msg;
                bar.style.position = 'fixed';
                bar.style.top = '0';
                bar.style.left = '0';
                bar.style.right = '0';
                bar.style.padding = '10px 14px';
                bar.style.background = '#dc2626';
                bar.style.color = '#fff';
                bar.style.fontSize = '14px';
                bar.style.fontWeight = '700';
                bar.style.zIndex = '2147483647';
                bar.style.textAlign = 'center';
                document.body.appendChild(bar);
                const title = document.title || 'Application';
                let ticks = 0;
                const h = setInterval(() => {
                  document.title = (ticks % 2 ? '⚠ ACTION REQUIRED | ' : '') + title;
                  ticks += 1;
                  if (ticks > 20) {
                    clearInterval(h);
                    document.title = title;
                  }
                }, 450);
              } catch (e) {}
            }
            """,
            message,
        )
    except Exception:
        pass


def _pause_for_user(
    *,
    page,
    emit: Callable[[str], None],
    pause_cb: Optional[Callable[[Dict[str, object]], None]],
    wait_for_close_cb: Optional[Callable[[], None]],
    reason: str,
    message: str,
    missing_fields: Optional[List[str]] = None,
) -> None:
    attention_msg = "User action required: fill missing fields / complete OTP / submit manually."
    _request_user_attention(page, attention_msg)
    payload = {
        "reason": reason,
        "prompt": message,
        "missing_fields": list(missing_fields or []),
        "url": page.url,
    }
    if pause_cb:
        try:
            pause_cb(payload)
        except Exception:
            pass
    emit(message)
    if wait_for_close_cb:
        emit("Waiting for user confirmation to close browser")
        wait_for_close_cb()


def auto_apply_job(
    *,
    profile: Profile,
    job: Job,
    resume_path: Path,
    auto_submit: bool = False,
    headless: bool = True,
    progress_cb: Optional[Callable[[str], None]] = None,
    pause_cb: Optional[Callable[[Dict[str, object]], None]] = None,
    wait_for_close_cb: Optional[Callable[[], None]] = None,
    max_steps: int = 6,
) -> AutoApplyResult:
    """
    Conservative auto-apply flow:
    - auto-fill known fields
    - never auto-submit
    - pause for manual review / OTP / submit
    - browser closes only when caller explicitly confirms
    """
    apply_url = _source_url(job)
    if not apply_url:
        return AutoApplyResult(
            job_id=job.id,
            profile_tag=profile.tag,
            status="unsupported",
            submitted=False,
            apply_url=None,
            ats=_source_ats(job),
            message="Job has no application URL.",
            steps=["Missing source URL"],
        )

    ats = _source_ats(job)
    if ats and ats not in SUPPORTED_ATS:
        return AutoApplyResult(
            job_id=job.id,
            profile_tag=profile.tag,
            status="unsupported",
            submitted=False,
            apply_url=apply_url,
            ats=ats,
            message=f"ATS '{ats}' is not yet automated.",
            steps=["Unsupported ATS"],
        )

    source_type = (job.source.source_type or "").lower() if job.source else ""
    if source_type == "aggregator" or _looks_like_listing_page(apply_url):
        return AutoApplyResult(
            job_id=job.id,
            profile_tag=profile.tag,
            status="unsupported",
            submitted=False,
            apply_url=apply_url,
            ats=ats,
            message="This URL appears to be a job listing/search page. Use a direct job application posting URL.",
            steps=["Listing page detected", f"Domain: {_domain(apply_url)}"],
        )

    if not resume_path.exists():
        return AutoApplyResult(
            job_id=job.id,
            profile_tag=profile.tag,
            status="failed",
            submitted=False,
            apply_url=apply_url,
            ats=ats,
            message=f"Resume file missing: {resume_path}",
            steps=["Resume export not found"],
        )

    try:
        from playwright.sync_api import TimeoutError as PlaywrightTimeoutError
        from playwright.sync_api import sync_playwright
    except Exception:
        return AutoApplyResult(
            job_id=job.id,
            profile_tag=profile.tag,
            status="failed",
            submitted=False,
            apply_url=apply_url,
            ats=ats,
            message="Playwright is not installed. Install playwright and browser binaries.",
            steps=[
                "pip install playwright",
                "python -m playwright install chromium",
            ],
        )

    contact = profile.identity.contact
    email = contact.email if contact else ""
    phone = contact.phone if contact else ""
    first_name, last_name = _split_name(profile.identity.name)
    steps: List[str] = []

    def emit(step: str) -> None:
        steps.append(step)
        if progress_cb:
            try:
                progress_cb(step)
            except Exception:
                pass

    emit(f"Opening application page ({_domain(apply_url)})")
    if auto_submit:
        emit("Auto-submit request ignored by safety policy; manual submit is required")
    if not headless:
        emit("Visible browser mode enabled")

    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=headless, slow_mo=(250 if not headless else 0))
            context = browser.new_context()
            page = context.new_page()
            page.goto(apply_url, wait_until="domcontentloaded", timeout=45000)

            # Some boards open form after clicking Apply.
            if _click_first(page, APPLY_ENTRY_SELECTORS):
                emit("Clicked Apply button")
                page.wait_for_timeout(1000)
            else:
                emit("Apply button not required or not found")
            contact_inputs_available = sum(
                1
                for selectors in [FIRST_NAME_SELECTORS, LAST_NAME_SELECTORS, EMAIL_SELECTORS, PHONE_SELECTORS]
                if _has_any(page, selectors)
            )
            resume_input_available = _has_any(page, RESUME_SELECTORS)
            emit(
                f"Detected {contact_inputs_available} contact inputs and "
                f"{'a' if resume_input_available else 'no'} resume upload input"
            )
            if contact_inputs_available == 0 and not resume_input_available:
                emit("No application form detected on current page")
                _pause_for_user(
                    page=page,
                    emit=emit,
                    pause_cb=pause_cb,
                    wait_for_close_cb=wait_for_close_cb if not headless else None,
                    reason="form_not_detected",
                    message=(
                        "Application form was not detected automatically. "
                        "Navigate manually, complete OTP if required, and submit."
                    ),
                )
                final_url = page.url
                context.close()
                browser.close()
                return AutoApplyResult(
                    job_id=job.id,
                    profile_tag=profile.tag,
                    status="ready_to_submit" if not headless else "unsupported",
                    submitted=False,
                    apply_url=final_url or apply_url,
                    ats=ats,
                    message=(
                        "Application form was not detected automatically. "
                        "Please navigate/submit manually."
                        if not headless
                        else "Could not find an application form on this page. Try a direct posting URL."
                    ),
                    steps=steps,
                )

            filled_total = 0
            uploaded = False
            for step_idx in range(1, max_steps + 1):
                filled_step = 0
                if _fill_first(page, FIRST_NAME_SELECTORS, first_name):
                    filled_step += 1
                if _fill_first(page, LAST_NAME_SELECTORS, last_name):
                    filled_step += 1
                if _fill_first(page, EMAIL_SELECTORS, email):
                    filled_step += 1
                if _fill_first(page, PHONE_SELECTORS, phone or ""):
                    filled_step += 1
                filled_total += filled_step
                emit(f"Step {step_idx}: filled {filled_step} contact fields ({filled_total} total)")

                if not uploaded:
                    for selector in RESUME_SELECTORS:
                        try:
                            locator = page.locator(selector).first
                            if locator.count() > 0:
                                locator.set_input_files(str(resume_path))
                                uploaded = True
                                emit("Uploaded tailored resume")
                                break
                        except Exception:
                            continue

                missing_required = _detect_unfilled_required_fields(page)
                if missing_required:
                    _pause_for_user(
                        page=page,
                        emit=emit,
                        pause_cb=pause_cb,
                        wait_for_close_cb=wait_for_close_cb,
                        reason="missing_required_fields",
                        message=(
                            "Unfilled required fields detected. "
                            "Please complete them, handle OTP if prompted, and submit manually."
                        ),
                        missing_fields=missing_required,
                    )
                    final_url = page.url or apply_url
                    context.close()
                    browser.close()
                    return AutoApplyResult(
                        job_id=job.id,
                        profile_tag=profile.tag,
                        status="ready_to_submit",
                        submitted=False,
                        apply_url=final_url,
                        ats=ats,
                        message="Manual action required: complete required fields/OTP and submit manually.",
                        steps=steps,
                    )

                if _click_first(page, NEXT_STEP_SELECTORS):
                    emit(f"Step {step_idx}: moved to next application step")
                    page.wait_for_timeout(1200)
                    continue

                emit(f"Step {step_idx}: no next-step button detected; reached final/manual stage")
                break

            if not uploaded:
                emit("Resume upload input not found")

            _pause_for_user(
                page=page,
                emit=emit,
                pause_cb=pause_cb,
                wait_for_close_cb=wait_for_close_cb,
                reason="manual_review",
                message="Verify details, complete OTP if required, and submit manually.",
            )

            final_url = page.url or apply_url
            context.close()
            browser.close()

            return AutoApplyResult(
                job_id=job.id,
                profile_tag=profile.tag,
                status="ready_to_submit",
                submitted=False,
                apply_url=final_url,
                ats=ats,
                message="Verify details, complete OTP if required, and submit manually.",
                steps=steps,
            )
    except PlaywrightTimeoutError:
        return AutoApplyResult(
            job_id=job.id,
            profile_tag=profile.tag,
            status="failed",
            submitted=False,
            apply_url=apply_url,
            ats=ats,
            message="Timed out while loading or interacting with the application page.",
            steps=steps + ["Timeout reached"],
        )
    except Exception as exc:
        return AutoApplyResult(
            job_id=job.id,
            profile_tag=profile.tag,
            status="failed",
            submitted=False,
            apply_url=apply_url,
            ats=ats,
            message=f"Auto-apply failed: {exc}",
            steps=steps,
        )
