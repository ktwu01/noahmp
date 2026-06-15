"""Tests for utility/noahmp-doctor.py.

No network, no real HPC, no real compilers (subprocess probes are monkeypatched).
Every path builds a fake run directory under tmp_path. First test suite in this repo.
`detect_logs` returns (logs, omitted_count, missing_explicit); tests unpack accordingly.
"""
from __future__ import annotations

import importlib.util
import os
from pathlib import Path

import pytest

# load the hyphenated module file by path
_SPEC = importlib.util.spec_from_file_location(
    "noahmp_doctor", Path(__file__).resolve().parent.parent / "noahmp-doctor.py")
doctor = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(doctor)


# ── fixtures ─────────────────────────────────────────────────────────────────
@pytest.fixture
def rundir(tmp_path: Path) -> Path:
    (tmp_path / "namelist.hrldas").write_text("&noahmp_offline\n NSOIL=4\n/\n")
    (tmp_path / "slurm-123.out").write_text("starting run\nstep 1 ok\n")
    return tmp_path


# ── log detection ────────────────────────────────────────────────────────────
def test_detect_single_log(rundir):
    logs, omitted, missing = doctor.detect_logs(rundir, [])
    assert any(p.name == "slurm-123.out" for p in logs)
    assert omitted == 0 and missing == []


def test_detect_explicit_override(rundir):
    custom = rundir / "my.log"
    custom.write_text("boom")
    logs, _, missing = doctor.detect_logs(rundir, [str(custom)])
    assert [p.name for p in logs] == ["my.log"] and missing == []


def test_detect_explicit_relative_to_C(tmp_path):
    # explicit relative log resolves against the -C dir, not process cwd
    run = tmp_path / "run"
    run.mkdir()
    (run / "boom.out").write_text("forrtl: error")
    logs, _, missing = doctor.detect_logs(run, ["boom.out"])
    assert [p.name for p in logs] == ["boom.out"] and missing == []


def test_detect_explicit_missing_reported(tmp_path):
    logs, _, missing = doctor.detect_logs(tmp_path, ["nope.out"])
    assert logs == [] and len(missing) == 1


def test_detect_no_log(tmp_path):
    assert doctor.detect_logs(tmp_path, []) == ([], 0, [])


def test_detect_skips_own_report(rundir):
    (rundir / "noahmp_doctor_report.txt").write_text("old report")
    logs, _, _ = doctor.detect_logs(rundir, [])
    assert all(p.name != "noahmp_doctor_report.txt" for p in logs)


def test_rank_logs_detected_first(tmp_path):
    (tmp_path / "rsl.error.0001").write_text("forrtl: severe segmentation fault")
    (tmp_path / "slurm-1.out").write_text("MPI_ABORT was invoked")
    logs, _, _ = doctor.detect_logs(tmp_path, [])
    assert logs[0].name.startswith("rsl.error")  # rank logs ranked first


def test_detect_caps_many_logs(tmp_path):
    for i in range(50):
        (tmp_path / f"rsl.error.{i:04d}").write_text("forrtl: error here")
    logs, omitted, _ = doctor.detect_logs(tmp_path, [])
    assert len(logs) == doctor.MAX_LOGS
    assert omitted == 50 - doctor.MAX_LOGS


# ── tail reading ─────────────────────────────────────────────────────────────
def test_read_tail_small(rundir):
    body = doctor.read_tail(rundir / "slurm-123.out")
    assert "step 1 ok" in body


def test_read_tail_huge_is_bounded(tmp_path):
    big = tmp_path / "big.log"
    big.write_bytes(b"x\n" * 2_000_000)  # ~4MB
    body = doctor.read_tail(big)
    assert "truncated" in body
    assert len(body.encode()) <= doctor.EMBED_CAP_BYTES + 200


def test_read_tail_binary_no_crash(tmp_path):
    b = tmp_path / "b.log"
    b.write_bytes(b"\xff\xfe\x00 fatal error \x80")
    body = doctor.read_tail(b)
    assert "fatal error" in body


def test_read_tail_empty(tmp_path):
    e = tmp_path / "e.log"
    e.write_text("")
    assert isinstance(doctor.read_tail(e), str)


# ── error scan (streams each file from the top) ──────────────────────────────
def test_scan_errors_finds_keywords(tmp_path):
    a = tmp_path / "slurm-1.out"
    a.write_text("all good\nMPI_ABORT invoked\n")
    b = tmp_path / "rsl.error.0001"
    b.write_text("forrtl: severe: segmentation fault\n")
    hits = doctor.scan_errors([a, b])
    assert any("rsl.error" in h for h in hits)
    assert any("MPI_ABORT" in h for h in hits)


def test_scan_errors_finds_fatal_above_tail(tmp_path):
    # the true first fatal is far above the last 200 lines; scan must still find it
    log = tmp_path / "rsl.error.0000"
    lines = ["forrtl: severe (174): SIGSEGV at step 3"] + [f"noise {i}" for i in range(1000)]
    log.write_text("\n".join(lines))
    hits = doctor.scan_errors([log])
    assert any("SIGSEGV" in h for h in hits)


def test_scan_errors_no_hits(tmp_path):
    log = tmp_path / "x.log"
    log.write_text("everything fine\n")
    assert doctor.scan_errors([log]) == []


# ── source context ───────────────────────────────────────────────────────────
def test_source_context_reachable(tmp_path):
    (tmp_path / "EnergyMainMod.F90").write_text("\n".join(f"line{i}" for i in range(20)))
    logs = [(Path("l"), "crash at EnergyMainMod.F90:10")]
    out = doctor.source_context(logs, tmp_path)
    assert "EnergyMainMod.F90:10" in out and "line9" in out


def test_source_context_unreachable(tmp_path):
    logs = [(Path("l"), "crash at Missing.F90:5")]
    assert "not reachable" in doctor.source_context(logs, tmp_path)


def test_source_context_no_backtrace(tmp_path):
    assert doctor.NOT_FOUND in doctor.source_context([(Path("l"), "no refs")], tmp_path)


# ── build options: BOTH include depths + missing repo ────────────────────────
def _make_tree(root: Path, depth_up: int):
    """Create rundir/Makefile that includes user_build_options `depth_up` levels up."""
    run = root
    for i in range(depth_up):
        run = run / f"d{i}"
    run.mkdir(parents=True)
    rel = "/".join([".."] * depth_up) + "/user_build_options"
    (run / "Makefile").write_text(f"include {rel}\n")
    (root / "user_build_options").write_text("COMPILERF90 = gfortran\nFC = mpif90\n")
    return run


def test_build_options_depth_two(tmp_path):
    run = _make_tree(tmp_path, 2)  # ../../user_build_options
    src, text = doctor.find_build_options(run)
    assert "gfortran" in text and doctor.NOT_FOUND not in src


def test_build_options_depth_three(tmp_path):
    run = _make_tree(tmp_path, 3)  # ../../../user_build_options
    src, text = doctor.find_build_options(run)
    assert "gfortran" in text


def test_build_options_missing_repo(tmp_path):
    (tmp_path / "Makefile").write_text("include ../../hrldas/user_build_options\n")
    src, text = doctor.find_build_options(tmp_path)
    assert doctor.NOT_FOUND in src and text == ""


def test_build_options_none_at_all(tmp_path):
    src, text = doctor.find_build_options(tmp_path)
    assert doctor.NOT_FOUND in src


def test_parse_fc():
    assert doctor.parse_fc("COMPILERF90 = gfortran\n") == "gfortran"
    assert doctor.parse_fc("nothing here") is None


# ── namelists ────────────────────────────────────────────────────────────────
def test_namelist_known_name_priority(rundir):
    nls = doctor.find_namelists(rundir)
    assert nls[0][0] == "namelist.hrldas"


def test_namelist_multiple_each_labeled(tmp_path):
    (tmp_path / "namelist.hrldas").write_text("a")
    (tmp_path / "namelist.input").write_text("b")
    (tmp_path / "namelist.output").write_text("c")
    names = [n for n, _ in doctor.find_namelists(tmp_path)]
    assert "namelist.hrldas" in names and "namelist.input" in names and "namelist.output" in names


def test_namelist_none(tmp_path):
    assert doctor.find_namelists(tmp_path) == []


# ── environment capture (sandboxed; monkeypatched) ───────────────────────────
def test_env_compiler_ok(monkeypatch):
    monkeypatch.setattr(doctor, "_run", lambda cmd: "GNU Fortran 11.4.0")
    env = doctor.capture_environment("gfortran")
    assert "11.4.0" in env["compiler"]


def test_env_compiler_timeout(monkeypatch):
    monkeypatch.setattr(doctor, "_run", lambda cmd: "not detected (timed out)")
    env = doctor.capture_environment("gfortran")
    assert "not detected" in env["compiler"]


def test_env_no_fc(monkeypatch):
    monkeypatch.setattr(doctor, "_run", lambda cmd: "not detected")
    env = doctor.capture_environment(None)
    assert "no FC" in env["compiler"]


def test_env_fc_not_allowlisted_is_not_executed(monkeypatch):
    # an arbitrary path from user_build_options must NOT be executed (read-only contract)
    called = {"ran": False}

    def spy(cmd):
        called["ran"] = True
        return "should not run"
    monkeypatch.setattr(doctor, "_run", spy)
    env = doctor.capture_environment("/some/evil/path/to/compiler")
    assert "not executed" in env["compiler"]
    # spy may still run for mpi/module probes, but never for the bad FC value:
    assert "should not run" not in env["compiler"]


def test_env_scheduler_redacts_secrets_and_hosts(monkeypatch):
    monkeypatch.setattr(doctor, "_run", lambda cmd: "stub")
    monkeypatch.setenv("SLURM_JOB_ID", "12345")
    monkeypatch.setenv("SLURM_NODELIST", "node[001-004]")
    monkeypatch.setenv("MPI_AUTH_TOKEN", "supersecret")
    env = doctor.capture_environment(None)
    sched = env["scheduler"]
    assert "SLURM_JOB_ID=12345" in sched           # benign value kept
    assert "supersecret" not in sched              # secret value scrubbed
    assert "node[001-004]" not in sched            # host/node list scrubbed
    assert "<REDACTED_SECRET>" in sched and "<REDACTED_HOST>" in sched


def test_run_handles_missing_binary():
    # real _run against a guaranteed-absent command: must not raise
    out = doctor._run(["definitely_not_a_real_binary_xyz", "--version"])
    assert "not detected" in out


def test_git_not_a_repo(tmp_path, monkeypatch):
    monkeypatch.setattr(doctor, "_run", lambda cmd: "not detected (not on PATH)")
    assert "not detected" in doctor.git_info(tmp_path)


def test_git_in_repo(monkeypatch):
    calls = {"n": 0}

    def fake(cmd):
        calls["n"] += 1
        return "abc123" if "rev-parse" in cmd else ""  # clean tree
    monkeypatch.setattr(doctor, "_run", fake)
    assert "abc123" in doctor.git_info(Path("."))


# ── redaction (security-critical, ordered, stable placeholders) ──────────────
def test_redact_home_before_user(monkeypatch):
    monkeypatch.setenv("USER", "alice")
    monkeypatch.setattr(os.path, "expanduser", lambda p: "/home/alice")
    redact, summary = doctor.build_redactor()
    out = redact("path /home/alice/run and user alice")
    assert "/home/alice" not in out
    assert "alice" not in out
    assert summary["home"] >= 1 and summary["user"] >= 1


def test_redact_scratch_stable_placeholders():
    redact, summary = doctor.build_redactor()
    out = redact("/scratch/projA/run /glade/work/x /scratch/projA/run")
    # same path → same placeholder; different path → different
    assert out.count("<PATH_1>") == 2
    assert "<PATH_2>" in out
    assert summary["scratch"] == 3


def test_redact_each_scratch_root():
    redact, summary = doctor.build_redactor()
    text = " ".join(f"{root}/x" for root in doctor.SCRATCH_ROOTS)
    out = redact(text)
    for root in doctor.SCRATCH_ROOTS:
        assert f"{root}/x" not in out
    assert summary["scratch"] >= len(doctor.SCRATCH_ROOTS)


def test_redact_generic_home():
    redact, summary = doctor.build_redactor()
    out = redact("/u/abc/file and /users/bob/x")
    assert "<PATH_GENERIC>" in out and summary["generic"] >= 1


def test_redact_summary_counts_match():
    redact, summary = doctor.build_redactor()
    redact("/scratch/a /scratch/b")
    assert summary["scratch"] == 2


def test_redact_uppercase_and_broad_roots():
    redact, summary = doctor.build_redactor()
    out = redact("/Users/bob/run /pscratch/sd/x /global/u1/y /projects/abc/z")
    assert "/Users/bob/run" not in out
    assert "/pscratch/sd/x" not in out
    assert "/global/u1/y" not in out
    assert "/projects/abc/z" not in out
    assert summary["generic"] >= 4


# ── end-to-end integration ───────────────────────────────────────────────────
def test_end_to_end(tmp_path, monkeypatch):
    # synthetic failed-run dir two levels under a build root
    root = tmp_path
    (root / "user_build_options").write_text("COMPILERF90 = gfortran\n")
    run = root / "exp" / "case01"
    run.mkdir(parents=True)
    (run / "Makefile").write_text("include ../../user_build_options\n")
    (run / "namelist.hrldas").write_text("&noahmp_offline\n NSOIL=4\n/\n")
    (run / "rsl.error.0000").write_text("forrtl: severe (174): SIGSEGV\nbacktrace:\n")
    (run / "slurm-9.out").write_text("MPI_ABORT was invoked on rank 0\n")

    monkeypatch.setattr(doctor, "_run", lambda cmd: "stub-output")
    redact, summary = doctor.build_redactor()
    sections = doctor.build_sections(run, [])
    report = doctor.render_report(sections, redact, summary)

    assert "Copy everything below into the LLM you trust." in report
    assert "namelist.hrldas" in report
    assert "gfortran" in report           # build options located across 2 levels
    assert "forrtl" in report             # first fatal from rank log
    assert "REDACTION SUMMARY" in report


def test_main_stdout_default(rundir, monkeypatch, capsys):
    monkeypatch.setattr(doctor, "_run", lambda cmd: "stub")
    rc = doctor.main(["-C", str(rundir)])
    captured = capsys.readouterr()
    assert rc == 0
    assert "Copy everything below" in captured.out
    # default must NOT write a file into the run dir
    assert not (rundir / "noahmp_doctor_report.txt").exists()


def test_main_output_file(rundir, monkeypatch, capsys):
    monkeypatch.setattr(doctor, "_run", lambda cmd: "stub")
    target = rundir / "report.txt"
    rc = doctor.main(["-C", str(rundir), "-o", str(target)])
    assert rc == 0
    assert target.is_file() and "Copy everything below" in target.read_text()


def test_main_bad_directory_returns_nonzero(tmp_path, capsys):
    missing = tmp_path / "does_not_exist"
    rc = doctor.main(["-C", str(missing)])
    assert rc == 2
    assert "not a directory" in capsys.readouterr().err
