"""Offline editorial handoff; prose is authored externally, never inferred here."""
from __future__ import annotations

import difflib
import hashlib
import json
import re
import unicodedata
from pathlib import Path

from talk_journal.output import atomic_text, digest_file

AUTHOR = "openai-codex/gpt-6.1-sol-900k"
PAGE = re.compile(r"[a-z][a-z0-9-]{0,63}\.md\Z")
DIGEST = re.compile(r"[0-9a-f]{64}\Z")


def page_name(value: object) -> bool:
    return isinstance(value, str) and bool(PAGE.fullmatch(value)) and value not in {"index.md", "review.md"}


def safe_markdown(value: object, targets: set[str]) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("Expected nonempty editorial text")
    if any(unicodedata.category(char).startswith("C") and char not in "\n\t" for char in value):
        raise ValueError("Control characters in editorial text")
    if any(token in value.lower() for token in ("!", "<", ">", "\\", "://", "javascript:", "data:")):
        raise ValueError("Unsafe Markdown; only local ordinary page links are supported")
    def link(match):
        if match[2] not in targets:
            raise ValueError("Unknown or unsafe page link")
        return ""
    remainder = re.sub(r"\[([^\[\]\n]+)\]\(([^()\s]+)\)", link, value)
    if "[" in remainder or "]" in remainder or value.startswith("---"):
        raise ValueError("Unsupported Markdown link/frontmatter syntax")


def validate_packet(packet: object, identity: str, text: str, targets: set[str]) -> None:
    keys = {"schema_version", "source_sha256", "author", "dossier", "uncertainty", "quotes", "pages"}
    if not isinstance(packet, dict) or set(packet) != keys:
        raise ValueError("Malformed editorial packet fields")
    if type(packet["schema_version"]) is not int or packet["schema_version"] != 1 or packet["source_sha256"] != identity:
        raise ValueError("Packet schema/source mismatch")
    if packet["author"] != AUTHOR:
        raise ValueError("Unsupported editorial author declaration")
    if not isinstance(packet["pages"], list) or not packet["pages"]:
        raise ValueError("Packet needs full-page edits")
    names = []
    for page in packet["pages"]:
        if not isinstance(page, dict) or set(page) != {"path", "before_sha256", "markdown"} or not page_name(page["path"]):
            raise ValueError("Invalid page edit/path")
        before = page["before_sha256"]
        if before is not None and (not isinstance(before, str) or not DIGEST.fullmatch(before)):
            raise ValueError("Invalid before-image digest")
        names.append(page["path"])
    if len(names) != len(set(names)):
        raise ValueError("Duplicate page edits")
    targets = targets | set(names)
    safe_markdown(packet["dossier"], set())
    for key in ("uncertainty", "quotes"):
        if not isinstance(packet[key], list) or not packet[key]:
            raise ValueError(f"Nonempty {key} list required")
        for item in packet[key]:
            safe_markdown(item, set())
    if not all(quote in text for quote in packet["quotes"]):
        raise ValueError("Evidence quotes must occur in the source")
    for page in packet["pages"]:
        safe_markdown(page["markdown"], targets)


def no_symlinks(path: Path) -> None:
    absolute = path.absolute()
    if any(item.is_symlink() for item in (absolute, *absolute.parents)):
        raise ValueError("Symlink paths are not supported")


def load_base(base: Path) -> tuple[dict, dict, dict]:
    no_symlinks(base)
    no_symlinks(base / "manifest.json")
    previous = json.loads((base / "manifest.json").read_text(encoding="utf-8"))
    if not isinstance(previous, dict) or set(previous) != {"schema_version", "sources", "pages", "artifacts"}:
        raise ValueError("Malformed base manifest")
    sources, pages, artifacts = (previous[key] for key in ("sources", "pages", "artifacts"))
    if type(previous["schema_version"]) is not int or previous["schema_version"] != 1 or not all(isinstance(item, dict) and item for item in (sources, pages, artifacts)):
        raise ValueError("Malformed base schema")
    if not all(isinstance(key, str) and DIGEST.fullmatch(key) and isinstance(value, str) and DIGEST.fullmatch(value) for key, value in sources.items()):
        raise ValueError("Malformed base source identity")
    if not all(page_name(key) and isinstance(refs, list) and refs and all(isinstance(ref, str) and ref in sources for ref in refs) and len(refs) == len(set(refs)) for key, refs in pages.items()):
        raise ValueError("Malformed base page provenance")
    expected = {"index.md", "review.md"} | set(pages)
    expected |= {f"{folder}/{identity}.{extension}" for identity in sources for folder, extension in (("sources", "txt"), ("dossiers", "md"))}
    if set(artifacts) != expected:
        raise ValueError("Unrecognized base artifact paths")
    present = set()
    for path in base.rglob("*"):
        no_symlinks(path)
        if path.is_file():
            present.add(path.relative_to(base).as_posix())
    if present != expected | {"manifest.json"}:
        raise ValueError("Untracked or missing base artifacts")
    files = {}
    for name, expected_digest in artifacts.items():
        if not isinstance(expected_digest, str) or not DIGEST.fullmatch(expected_digest) or digest_file(base / name) != expected_digest:
            raise ValueError("Base artifact modified")
        files[name] = (base / name).read_bytes()
        if name.endswith(".md"):
            targets = set(artifacts) - {name}
            targets.add(name)
            if name.startswith("dossiers/"):
                targets = {"../" + item for item in targets}
            safe_markdown(files[name].decode("utf-8"), targets)
    for identity in sources:
        if hashlib.sha256(files[f"sources/{identity}.txt"]).hexdigest() != identity:
            raise ValueError("Base source identity mismatch")
    return files, sources.copy(), pages.copy()


def ingest(source: Path, packet_path: Path, destination: Path, *, base: Path | None = None) -> str:
    no_symlinks(destination)
    if base is None and destination.exists():
        base = destination
    if base is not None:
        no_symlinks(base)
        if base.resolve() in destination.resolve().parents:
            raise ValueError("Destination must not be inside base")
    raw = source.read_bytes()
    text = raw.decode("utf-8")
    identity = hashlib.sha256(raw).hexdigest()
    packet = json.loads(packet_path.read_text(encoding="utf-8"))

    packet_digest = digest_file(packet_path)
    files, sources, pages = {}, {}, {}
    if base is not None:
        files, sources, pages = load_base(base)
    validate_packet(packet, identity, text, set(pages))
    if base is not None:
        if identity in sources:
            if sources[identity] != packet_digest:
                raise ValueError("Same source has a different editorial packet")
            return "NOOP"
    sources[identity] = packet_digest
    dossier = f"# Dossier\n\n{packet['dossier']}\n\n## Uncertainty\n\n"
    dossier += "".join(f"- {item}\n" for item in packet["uncertainty"])
    dossier += f"\n[Raw source](../sources/{identity}.txt)\n\nAuthor declaration: {AUTHOR}\n"
    dossier += "\n## Source quotations\n\n" + "".join("    " + line + "\n" for quote in packet["quotes"] for line in quote.splitlines())
    files[f"dossiers/{identity}.md"] = dossier
    review = "# Editorial review\n\nSynthetic or real fidelity requires reviewer judgement.\n\n" + dossier.replace("../sources/", "sources/")
    for page in packet["pages"]:
        name, body = page["path"], page["markdown"]
        before = files.get(name, b"").decode("utf-8")
        if page["before_sha256"] != (hashlib.sha256(files[name]).hexdigest() if name in files else None):
            raise ValueError("Page before-image mismatch")
        page_sources = pages.get(name, []) + [identity]
        pages[name] = page_sources
        files[name] = body + "\n## Editorial provenance\n\n" + "".join(
            f"- [Dossier {item}](dossiers/{item}.md)\n" for item in page_sources) + "\n[Index](index.md)\n"
        diff = "".join(difflib.unified_diff(before.splitlines(keepends=True), files[name].splitlines(keepends=True), fromfile=name if before else "/dev/null", tofile=name))
        review += f"\n## Before page: {name}\n\n" + "".join("    " + line + "\n" for line in before.splitlines())
        review += f"\n## Full page: {name}\n\n" + "".join("    " + line + "\n" for line in files[name].splitlines())
        review += "\n## Diff\n\n" + "".join("    " + line + "\n" for line in diff.splitlines())
    files["index.md"] = "# Journal wiki\n\n" + "".join(f"- [{name}]({name})\n" for name in sorted(pages)) + "\n[Review](review.md)\n"
    files["review.md"] = review
    files[f"sources/{identity}.txt"] = raw
    destination.mkdir(parents=True, exist_ok=False)
    for name, body in files.items():
        path = destination / name
        path.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(body, bytes):
            path.write_bytes(body)
        else:
            atomic_text(path, body)
    manifest = {"schema_version": 1, "sources": sources, "pages": pages,
                "artifacts": {name: digest_file(destination / name) for name in files}}
    atomic_text(destination / "manifest.json", json.dumps(manifest, indent=2) + "\n")
    return "CREATED"


def main(argv: list[str]) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Apply an externally authored editorial packet; no inference")
    parser.add_argument("source", type=Path)
    parser.add_argument("packet", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--base", type=Path, help="Previous complete snapshot; never mutated")
    args = parser.parse_args(argv)
    try:
        print(ingest(args.source, args.packet, args.destination, base=args.base))
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.exit(1, f"Talk Journal editorial: {exc}\n")
    return 0
