#!/usr/bin/env python3
"""noahmp-doctor — assemble a copy-paste LLM debug report for a failed Noah-MP run.

Read-only. Run it from the host-model run directory (or build directory) where a
Noah-MP / HRLDAS / WRF / LIS / ERF run just failed:

    python3 /path/to/noahmp-doctor.py            # auto-detect the log
    python3 /path/to/noahmp-doctor.py run.err    # explicit log(s)
    python3 /path/to/noahmp-doctor.py --output report.txt

It gathers the context an expert would ask for (the failing log tail, the namelist,
the build options, compiler/MPI + module provenance, scheduler env, git state), redacts
home/user/host/scratch paths, and prints ONE block you paste into the LLM you trust.

Data flow:

    cwd ──► detect_logs ──► read_tail (seek-from-end) ──► scan_errors (first fatal)
        ├─► find_build_options (parse Makefile `include` line, walk up)
        ├─► find_namelists (known-name priority)
        ├─► capture_environment (FC/MPI/module/scheduler, sandboxed subprocess)
        └─► git_info
                         │
                         ▼
                 redact (ordered, stable per-path placeholder) + REDACTION SUMMARY
                         │
                         ▼
                 render_report ──► stdout (default) | --output file
"""

from __future__ import annotations

import argparse
import os
import re
import socket
import subprocess
import sys
from pathlib import Path

# ── tunables ────────────────────────────────────────────────────────────────
TAIL_LINES = 200
TAIL_BYTES = 1_000_000          # seek-from-end window; constant memory on huge logs
EMBED_CAP_BYTES = 500_000       # max embedded log bytes (paste-ability guard)
SUBPROCESS_TIMEOUT = 5          # seconds; never hang the user's terminal
UPWARD_LEVELS = 6               # how far to walk up for Makefile / user_build_options
MAX_LOGS = 12                   # cap inspected logs (a WRF run can have 1000s of rsl.*)
SCAN_BYTES = 5_000_000          # how far into each log to stream-scan for first fatal

NOT_FOUND = "not found"
ERROR_KEYWORDS = re.compile(
    r"\b(error|fatal|segmentation|segfault|mpi_?abort|forrtl|nan|abort|killed|"
    r"oom|out of memory|backtrace)\b",
    re.IGNORECASE,
)
# log filename patterns, most-specific (rank/ESMF) first so MPI first-fatal wins
LOG_PATTERNS = [
    "rsl.error.*", "rsl.out.*", "PET*.ESMF_LogFile", "*.ESMF_LogFile",
    "*.err", "slurm-*.out", "*.log", "*.out",
]
NAMELIST_PRIORITY = ["namelist.hrldas", "namelist.input", "namelist.noahmp"]
SCRATCH_ROOTS = ["/scratch", "/glade", "/work", "/lustre", "/project", "/gpfs"]
FORTRAN_SRC_REF = re.compile(r"([A-Za-z0-9_./-]+\.[fF]9?0?):(\d+)")


# ── log discovery & reading ──────────────────────────────────────────────────
def detect_logs(cwd: Path, explicit: list[str]) -> tuple[list[Path], int, list[str]]:
    """Return (logs_to_inspect, omitted_count, missing_explicit).

    Explicit paths win and are resolved against `cwd` (so `-C run foo.out` works).
    Otherwise pattern-match cwd. Rank/ESMF logs are kept as a group so the
    first-fatal scan can look across ranks (rank-0 logs often show only abort
    noise). Inspected logs are capped at MAX_LOGS, preferring non-empty files,
    so a run dir with thousands of rsl.* files stays usable.
    """
    if explicit:
        resolved, missing = [], []
        for p in explicit:
            cand = Path(p)
            if not cand.is_absolute():
                cand = cwd / cand
            (resolved if cand.is_file() else missing).append(cand)
        return resolved[:MAX_LOGS], max(0, len(resolved) - MAX_LOGS), [str(m) for m in missing]

    found: list[Path] = []
    seen: set[Path] = set()
    for pat in LOG_PATTERNS:
        for p in sorted(cwd.glob(pat)):
            rp = p.resolve()
            if p.is_file() and rp not in seen and p.name != "noahmp_doctor_report.txt":
                seen.add(rp)
                found.append(p)
    # prefer non-empty logs when capping
    found.sort(key=lambda p: (p.stat().st_size == 0, p.name))
    omitted = max(0, len(found) - MAX_LOGS)
    return found[:MAX_LOGS], omitted, []


def read_tail(path: Path) -> str:
    """Last TAIL_LINES lines via a bounded seek-from-end read. Constant memory."""
    try:
        size = path.stat().st_size
        with open(path, "rb") as fh:
            if size > TAIL_BYTES:
                fh.seek(-TAIL_BYTES, os.SEEK_END)
                truncated = True
            else:
                truncated = False
            data = fh.read()
        text = data.decode("utf-8", errors="replace")
        lines = text.splitlines()[-TAIL_LINES:]
        body = "\n".join(lines)
        if len(body.encode("utf-8")) > EMBED_CAP_BYTES:
            body = body.encode("utf-8")[-EMBED_CAP_BYTES:].decode("utf-8", "replace")
            truncated = True
        prefix = f"(showing last {TAIL_LINES} lines"
        prefix += "; earlier output truncated)\n" if truncated else ")\n"
        return prefix + body
    except OSError as exc:
        return f"{NOT_FOUND} (could not read {path.name}: {exc})"


def scan_errors(logs: list[Path]) -> list[str]:
    """First keyword-matching lines per log, scanned from the TOP of each file.

    The real first fatal often precedes the MPI abort noise that lands in the
    displayed tail, so this streams each file from the start (bounded by
    SCAN_BYTES) rather than reusing the tail.
    """
    hits: list[str] = []
    for path in logs:
        try:
            with open(path, "rb") as fh:
                raw = fh.read(SCAN_BYTES)
        except OSError:
            continue
        text = raw.decode("utf-8", errors="replace")
        per_log = 0
        for line in text.splitlines():
            if ERROR_KEYWORDS.search(line):
                hits.append(f"[{path.name}] {line.strip()}")
                per_log += 1
                if per_log >= 5 or len(hits) >= 40:
                    break
        if len(hits) >= 40:
            break
    return hits


def source_context(logs: list[tuple[Path, str]], cwd: Path) -> str:
    """Best-effort: quote source near a `file.F90:line` ref if reachable."""
    for _, body in logs:
        m = FORTRAN_SRC_REF.search(body)
        if not m:
            continue
        fname, lineno = m.group(1), int(m.group(2))
        for cand in (cwd / fname, cwd / Path(fname).name):
            if cand.is_file():
                try:
                    src = cand.read_text(errors="replace").splitlines()
                    lo, hi = max(0, lineno - 4), min(len(src), lineno + 3)
                    quoted = "\n".join(f"{i+1}: {src[i]}" for i in range(lo, hi))
                    return f"{fname}:{lineno}\n{quoted}"
                except OSError:
                    pass
        return (f"backtrace names {fname}:{lineno}, but the source is not reachable "
                f"from here (optimized build may also lack a usable backtrace)")
    return f"{NOT_FOUND} (no file.F90:line backtrace; build likely lacked -g -fbacktrace)"


# ── build options (cross-repo, variable include depth) ───────────────────────
def find_build_options(cwd: Path) -> tuple[str, str]:
    """Find user_build_options by reading a Makefile `include` line, walking up.

    Handles both depths (../../hrldas, ../../../hrldas) and the absent-hrldas case.
    Returns (source_path_or_notfound, contents_or_empty).
    """
    inc_re = re.compile(r"^\s*include\s+(\S*user_build_options)\s*$", re.MULTILINE)
    base = cwd
    for _ in range(UPWARD_LEVELS):
        for mk in ("Makefile", "makefile"):
            mkpath = base / mk
            if mkpath.is_file():
                try:
                    text = mkpath.read_text(errors="replace")
                except OSError:
                    text = ""
                m = inc_re.search(text)
                if m:
                    resolved = (base / m.group(1)).resolve()
                    if resolved.is_file():
                        try:
                            return str(resolved), resolved.read_text(errors="replace")
                        except OSError:
                            pass
                    return (f"{NOT_FOUND} (Makefile points to {m.group(1)} but it is "
                            f"not present; hrldas repo not located)", "")
        # also try a direct sibling user_build_options at each level
        direct = base / "user_build_options"
        if direct.is_file():
            try:
                return str(direct), direct.read_text(errors="replace")
            except OSError:
                pass
        if base.parent == base:
            break
        base = base.parent
    return f"{NOT_FOUND} (no Makefile include or user_build_options within {UPWARD_LEVELS} levels)", ""


def parse_fc(build_options: str) -> str | None:
    """Pull the Fortran compiler (FC/F90/COMPILERF90) from user_build_options."""
    for key in ("COMPILERF90", "FC", "F90", "MPIFC", "MPIF90"):
        m = re.search(rf"^\s*{key}\s*=\s*(\S+)", build_options, re.MULTILINE)
        if m:
            return m.group(1)
    return None


# ── namelists ────────────────────────────────────────────────────────────────
def find_namelists(cwd: Path) -> list[tuple[str, str]]:
    """Known-name priority; embed all found, each labeled. Empty list if none."""
    out: list[tuple[str, str]] = []
    seen: set[str] = set()
    for name in NAMELIST_PRIORITY:
        p = cwd / name
        if p.is_file():
            seen.add(name)
            out.append((name, _read_capped(p)))
    for p in sorted(cwd.glob("namelist*")):
        if p.is_file() and p.name not in seen:
            out.append((p.name, _read_capped(p)))
    return out


def _read_capped(p: Path, cap: int = 50_000) -> str:
    try:
        data = p.read_bytes()[:cap]
        return data.decode("utf-8", errors="replace")
    except OSError as exc:
        return f"{NOT_FOUND} (could not read: {exc})"


# ── environment provenance (sandboxed subprocess) ────────────────────────────
# Only these probe binaries are ever executed. We do NOT execute an FC value
# parsed from user_build_options (a project file): running an arbitrary path from
# a file we read would violate the read-only contract. We allow the common,
# well-known compiler/MPI driver names by exact basename match only.
_ALLOWED_PROBES = {
    "gfortran", "ifort", "ifx", "nvfortran", "pgfortran", "flang", "xlf", "xlf90",
    "mpif90", "mpifort", "mpiifort", "ftn",
}
# Scheduler/MPI env keys whose VALUE may carry a secret or sensitive endpoint.
_SECRET_KEY = re.compile(r"(TOKEN|SECRET|KEY|PASSWORD|PASSWD|CRED|AUTH)", re.IGNORECASE)
# Scheduler env keys whose value is a host / node list / IP we should not leak.
_HOSTY_KEY = re.compile(r"(HOST|NODE|NODELIST|ADDR|IP|SUBMIT)", re.IGNORECASE)


def _run(cmd: list[str]) -> str:
    """Run a probe command read-only with a hard timeout; never raise.

    Captures bytes and decodes with errors='replace' so non-UTF/locale output
    from a compiler or `module` cannot raise UnicodeDecodeError.
    """
    try:
        res = subprocess.run(
            cmd, capture_output=True, timeout=SUBPROCESS_TIMEOUT, check=False,
        )
        raw = res.stdout or res.stderr or b""
        out = raw.decode("utf-8", errors="replace").strip()
        return out if out else "not detected (empty output)"
    except FileNotFoundError:
        return "not detected (command not on PATH; module may not be loaded)"
    except subprocess.TimeoutExpired:
        return "not detected (timed out)"
    except OSError as exc:
        return f"not detected ({exc})"


def capture_environment(fc: str | None) -> dict[str, str]:
    env = {}
    env["compiler_name"] = fc or NOT_FOUND
    # Execute --version ONLY for an allowlisted, bare compiler name. Never run an
    # arbitrary path read from user_build_options.
    if fc and os.path.basename(fc) in _ALLOWED_PROBES and "/" not in fc:
        env["compiler"] = _run([fc, "--version"])
    elif fc:
        env["compiler"] = (f"not executed (value '{fc}' from user_build_options is "
                           f"not an allowlisted bare compiler name; reported as-is)")
    else:
        env["compiler"] = "not detected (no FC in user_build_options)"
    # MPI wrapper identity: -show reveals the underlying compiler+libs
    for wrapper in ("mpif90", "mpifort", "mpiifort"):
        show = _run([wrapper, "-show"])
        if not show.startswith("not detected"):
            env["mpi_wrapper"] = f"{wrapper} -show: {show}"
            break
    else:
        env["mpi_wrapper"] = "not detected (no mpi wrapper on PATH)"
    env["modules"] = _run(["/bin/bash", "-lc", "module list 2>&1"])
    # Scheduler env: redact secret-bearing and host/node/IP-bearing values.
    lines = []
    for k, v in sorted(os.environ.items()):
        if not k.startswith(("SLURM_", "PBS_", "LSB_", "OMPI_", "MPI_")):
            continue
        if _SECRET_KEY.search(k):
            v = "<REDACTED_SECRET>"
        elif _HOSTY_KEY.search(k):
            v = "<REDACTED_HOST>"
        lines.append(f"{k}={v}")
    env["scheduler"] = ("\n".join(lines) if lines
                        else "not detected (no SLURM_/PBS_ env vars set)")
    return env


def git_info(cwd: Path) -> str:
    head = _run(["git", "-C", str(cwd), "rev-parse", "HEAD"])
    if head.startswith("not detected"):
        return "not detected (not inside a git repo)"
    dirty = _run(["git", "-C", str(cwd), "status", "--porcelain"])
    state = "dirty" if dirty and not dirty.startswith("not detected") else "clean"
    return f"HEAD={head} ({state})"


# ── redaction (ordered, stable per-path placeholder) ─────────────────────────
def build_redactor():
    """Return (redact_fn, summary_dict). Order: HOME → USER → host → scratch → generic."""
    summary = {"home": 0, "user": 0, "hostname": 0, "scratch": 0, "generic": 0}
    path_map: dict[str, str] = {}

    home = os.path.expanduser("~")
    user = os.environ.get("USER") or os.environ.get("LOGNAME") or ""
    try:
        host = socket.gethostname()
    except OSError:
        host = ""

    def stable_path(_match: re.Match) -> str:
        original = _match.group(0)
        if original not in path_map:
            path_map[original] = f"<PATH_{len(path_map) + 1}>"
        summary["scratch"] += 1
        return path_map[original]

    scratch_re = re.compile(
        r"(?:" + "|".join(re.escape(r) for r in SCRATCH_ROOTS) + r")/[^\s:'\"]*",
        re.IGNORECASE,
    )
    # broad, case-insensitive generic home/scratch roots seen across HPC sites
    generic_re = re.compile(
        r"(?:/home/|/u/|/users/|/Users/|/projects/|/pscratch/|/global/|/cluster/|/nobackup/)"
        r"[^\s:'\"]*",
        re.IGNORECASE,
    )

    def redact(text: str) -> str:
        nonlocal summary
        if home and home != "/":
            text, n = re.subn(re.escape(home), "<HOME>", text)
            summary["home"] += n
        if user:
            text, n = re.subn(rf"\b{re.escape(user)}\b", "<USER>", text)
            summary["user"] += n
        if host:
            text, n = re.subn(re.escape(host), "<HOST>", text)
            summary["hostname"] += n
        text = scratch_re.sub(stable_path, text)
        text, n = generic_re.subn("<PATH_GENERIC>", text)
        summary["generic"] += n
        return text

    return redact, summary


# ── report rendering ─────────────────────────────────────────────────────────
def render_report(sections: dict[str, str], redact, summary) -> str:
    out = []
    out.append("Copy everything below into the LLM you trust.")
    out.append("=" * 70)
    out.append("Noah-MP failed-run debug report (generated by noahmp-doctor, read-only)")
    out.append("")
    for title, body in sections.items():
        out.append(f"### {title}")
        out.append(redact(body) if body else NOT_FOUND)
        out.append("")
    # summary computed AFTER all redaction so counts are accurate
    out.append("### REDACTION SUMMARY")
    out.append(", ".join(f"{k}: {v}" for k, v in summary.items())
               + "  (eyeball the report before pasting externally)")
    out.append("=" * 70)
    return "\n".join(out)


# ── orchestration ────────────────────────────────────────────────────────────
def build_sections(cwd: Path, explicit: list[str]) -> dict[str, str]:
    logs, omitted, missing = detect_logs(cwd, explicit)
    log_bodies = [(p, read_tail(p)) for p in logs]
    bo_src, bo_text = find_build_options(cwd)
    fc = parse_fc(bo_text) if bo_text else None
    env = capture_environment(fc)
    namelists = find_namelists(cwd)

    sections: dict[str, str] = {}
    sections["run directory"] = str(cwd)
    inspected = ", ".join(p.name for p in logs) if logs else f"{NOT_FOUND} (no log matched in cwd)"
    if omitted:
        inspected += f"  (+{omitted} more logs omitted; cap is {MAX_LOGS})"
    if missing:
        inspected += f"  (explicit logs not found: {', '.join(missing)})"
    sections["logs inspected"] = inspected
    sections["first fatal lines (scanned from top of each log)"] = (
        "\n".join(scan_errors(logs)) or "no keyword matches; see log tail")
    for p, body in log_bodies:
        sections[f"log tail: {p.name}"] = body
    sections["source context"] = source_context(log_bodies, cwd)
    for name, body in namelists:
        sections[f"namelist: {name}"] = body
    if not namelists:
        sections["namelist"] = f"{NOT_FOUND} (no namelist* in cwd)"
    sections["user_build_options"] = (
        f"# from {bo_src}\n{bo_text}" if bo_text else bo_src)
    sections["compiler"] = f"{env['compiler_name']}\n{env['compiler']}"
    sections["mpi wrapper"] = env["mpi_wrapper"]
    sections["loaded modules"] = env["modules"]
    sections["scheduler env"] = env["scheduler"]
    sections["git"] = git_info(cwd)
    return sections


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Assemble a copy-paste LLM debug report for a failed Noah-MP run.")
    parser.add_argument("logs", nargs="*",
                        help="explicit log file(s); omit to auto-detect in the current dir")
    parser.add_argument("-o", "--output",
                        help="write the report to this file (default: print to stdout)")
    parser.add_argument("-C", "--directory", default=".",
                        help="run directory to inspect (default: current dir)")
    args = parser.parse_args(argv)

    cwd = Path(args.directory).resolve()
    if not cwd.is_dir():
        print(f"error: -C/--directory '{args.directory}' is not a directory",
              file=sys.stderr)
        return 2

    redact, summary = build_redactor()
    sections = build_sections(cwd, args.logs)
    report = render_report(sections, redact, summary)

    if args.output:
        try:
            Path(args.output).write_text(report, encoding="utf-8")
            print(f"wrote {args.output}")
        except OSError as exc:
            print(f"could not write {args.output} ({exc}); printing to stdout instead\n",
                  file=sys.stderr)
            print(report)
    else:
        print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
