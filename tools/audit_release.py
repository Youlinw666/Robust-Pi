#!/usr/bin/env python3
"""Audit every publication-candidate byte, without installing/running a model.

Default writes nothing; --write-report updates the designated audit report.
This heuristic scan does not prove the absence of every possible secret.
Git metadata and Python/test caches are local runtime metadata, not release
content. They are excluded so the same check works in a fresh Git clone.
Review actual Git objects separately before publishing an existing history.
"""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import re
from urllib.parse import unquote

ROOT=Path(__file__).resolve().parents[1]
SPECIAL={"release-files.json","reports/G4_RELEASE_AUDIT.json"}
FORBIDDEN={".git","data","checkpoints","artifacts","videos","logs","environment","external","__pycache__",".pytest_cache"}
TEXT_EXT={".py",".md",".mmd",".json",".txt",".ini",""}
PRIVATE=re.compile(r"(?:/home/|/Users/)[A-Za-z0-9_.-]+/|[A-Za-z]:\\[A-Za-z0-9_.-]+\\")
SECRET={
    "private-key":re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----"),
    "github-token":re.compile(r"\b(?:gh[pousr]_[A-Za-z0-9]{25,}|github_pat_[A-Za-z0-9_]{40,})\b"),
    "api-token":re.compile(r"\bsk-(?:proj-|svcacct-)?[A-Za-z0-9_-]{30,}\b"),
    "aws-key":re.compile(r"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b"),
    "literal-secret":re.compile(r"(?i)(?:api[_-]?key|access[_-]?token|client[_-]?secret|password)\s*[:=]\s*['\"][A-Za-z0-9_./+=-]{24,}['\"]"),
}


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def audit():
    failures=[];links=0;total=0;count=0
    spec=json.loads((ROOT/"release-files.json").read_text());expected=spec["files"]
    for p in ROOT.rglob("*"):
        name=p.relative_to(ROOT).as_posix()
        if any(part in {".git","__pycache__",".pytest_cache"} for part in p.relative_to(ROOT).parts):
            continue
        if p.is_symlink():failures.append({"path":name,"reason":"symlink"});continue
        if any(part in FORBIDDEN or part.startswith(".venv") for part in p.relative_to(ROOT).parts):failures.append({"path":name,"reason":"private/generated directory"});continue
        if p.is_dir():continue
        if p.name in {"id_rsa","id_ed25519","id_ecdsa",".env"} or p.suffix in {".key",".pem",".ppk",".p12",".pfx"} or p.name.startswith(".env."):
            failures.append({"path":name,"reason":"credential filename"})
        count+=1;raw=p.read_bytes();total+=len(raw)
        if name not in expected and name not in SPECIAL:failures.append({"path":name,"reason":"outside allowlist"})
        if len(raw)>262144:failures.append({"path":name,"reason":"file over 256KiB"})
        if name in expected and (sha(p)!=expected[name]["sha256"] or len(raw)!=expected[name]["bytes"]):failures.append({"path":name,"reason":"content hash/size mismatch"})
        if p.suffix not in TEXT_EXT or b"\0" in raw:failures.append({"path":name,"reason":"binary/media/non-text type"});continue
        try:s=raw.decode("utf-8")
        except UnicodeDecodeError:failures.append({"path":name,"reason":"non UTF8 text"});continue
        if PRIVATE.search(s) or ("wsl"+"-pgy") in s:failures.append({"path":name,"reason":"personal absolute path/host alias"})
        for typ,pat in SECRET.items():
            if pat.search(s):failures.append({"path":name,"reason":"sensitive pattern: "+typ})
        if p.suffix==".py":
            try:ast.parse(s)
            except SyntaxError:failures.append({"path":name,"reason":"Python syntax"})
        if p.suffix==".md":
            if s.count("```")%2:failures.append({"path":name,"reason":"unbalanced code fences"})
            for t in re.findall(r"!?\[[^\]]*\]\(([^)]+)\)",s):
                if t.startswith(("https://","http://","mailto:")):continue
                target=(p.parent/unquote(t.split("#",1)[0])).resolve()
                if not target.is_relative_to(ROOT) or not target.exists():failures.append({"path":name,"reason":"invalid/non-local relative link","target":t})
                links+=1
    for name in expected:
        if not (ROOT/name).is_file():failures.append({"path":name,"reason":"allowlisted file missing"})
    if total>2*1024*1024:failures.append({"reason":"candidate over 2MiB"})
    license_file=ROOT/"LICENSE"
    if not license_file.exists() or "MIT License" not in license_file.read_text() or "Copyright (c) 2026 youlinw666" not in license_file.read_text():failures.append({"reason":"confirmed MIT license/attribution missing"})
    checks=json.loads((ROOT/"reports/CPU_CHECKS.json").read_text())
    if checks.get("exit_code")!=0 or checks.get("tests_passed",0)<6 or not checks.get("synthetic_only"):failures.append({"reason":"CPU validation not passed"})
    for n in ["README.md","README_EN.md"]:
        s=(ROOT/n).read_text()
        for fact in ["0/2","180","100"]:
            if fact not in s:failures.append({"path":n,"reason":"missing negative-result/pilot disclosure"})
    for p in (ROOT/"docs/diagrams").glob("*.mmd"):
        s=p.read_text()
        if not s.startswith("flowchart ") or s.count("[")!=s.count("]"):failures.append({"path":p.relative_to(ROOT).as_posix(),"reason":"Mermaid source structure"})
    return dict(status="G4_READY_FOR_PUBLICATION" if not failures else "G4_PUBLICATION_BLOCKED",scope="isolated allowlisted candidate only; original Git history excluded",files_checked=count,total_bytes=total,relative_links_checked=links,failures=failures,license="MIT",copyright="2026 youlinw666",media_files_included=0,git_history_included=False,new_gpu_training_inference_or_simulation=False,commit_push_or_upload=False,manifest_sha256=sha(ROOT/"release-files.json"),limits="Full candidate text scan with heuristic patterns; no mathematical guarantee that every possible secret format is detected")


def main():
    p=argparse.ArgumentParser();p.add_argument("--write-report",action="store_true");args=p.parse_args();r=audit()
    if args.write_report:(ROOT/"reports/G4_RELEASE_AUDIT.json").write_text(json.dumps(r,ensure_ascii=False,indent=2)+"\n")
    print(json.dumps(r,ensure_ascii=False,indent=2))
    raise SystemExit(0 if not r["failures"] else 2)


if __name__=="__main__":main()
