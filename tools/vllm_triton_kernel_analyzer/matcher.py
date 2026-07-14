from __future__ import annotations

import difflib
from collections import defaultdict
from pathlib import PurePosixPath

from .model import Comparison, DiffStats, Inventory, Kernel, Match


def diff_stats(old: str, new: str) -> DiffStats:
    added = deleted = changed = 0
    matcher = difflib.SequenceMatcher(None, old.splitlines(), new.splitlines(), autojunk=False)
    for tag, old_start, old_end, new_start, new_end in matcher.get_opcodes():
        if tag == "equal":
            continue
        old_count, new_count = old_end - old_start, new_end - new_start
        deleted += old_count
        added += new_count
        changed += max(old_count, new_count)
    return DiffStats(added, deleted, changed)


def _path_similarity(left: str, right: str) -> float:
    a, b = PurePosixPath(left), PurePosixPath(right)
    name_score = difflib.SequenceMatcher(None, a.name, b.name).ratio()
    parts_score = difflib.SequenceMatcher(None, a.parts, b.parts).ratio()
    return 0.65 * name_score + 0.35 * parts_score


def _token_similarity(left: Kernel, right: Kernel) -> float:
    if not left.body_tokens and not right.body_tokens:
        return 1.0
    if not left.body_tokens or not right.body_tokens:
        return 0.0
    # SequenceMatcher retains ordering, which is more discriminating than a
    # bag of common Triton tokens such as tl.load/tl.store.
    return difflib.SequenceMatcher(
        None, left.body_tokens, right.body_tokens, autojunk=False
    ).ratio()


def _score(left: Kernel, right: Kernel) -> float:
    body = _token_similarity(left, right)
    name = difflib.SequenceMatcher(None, left.name, right.name).ratio()
    path = _path_similarity(left.path, right.path)
    return 0.68 * body + 0.20 * name + 0.12 * path


def _location_key(kernel: Kernel) -> str:
    return f"{kernel.path}:{kernel.definition_line} · {kernel.qualname}"


def _status(old: Kernel, new: Kernel) -> str:
    same_location = old.path == new.path and old.qualname == new.qualname
    same_source = old.source_sha256 == new.source_sha256
    same_semantics = old.semantic_sha256 == new.semantic_sha256
    if same_location and same_source:
        return "unchanged"
    moved = old.path != new.path
    renamed = old.name != new.name
    if moved and renamed:
        prefix = "moved_renamed"
    elif moved:
        prefix = "moved"
    elif renamed:
        prefix = "renamed"
    else:
        prefix = "modified"
    if same_source:
        return f"{prefix}_source_exact"
    if same_semantics:
        return f"{prefix}_semantic_exact"
    return prefix


def _confidence(method: str, score: float, alternatives: list[tuple[str, float]]) -> str:
    ambiguous = bool(alternatives and score - alternatives[0][1] < 0.04)
    if ambiguous:
        return "low"
    if method in {"location", "semantic_exact", "source_exact"}:
        return "high"
    if method == "same_name" and score >= 0.72:
        return "high"
    if score >= 0.72:
        return "medium"
    return "low"


def compare_inventories(old: Inventory, new: Inventory, fuzzy_threshold: float = 0.58) -> Comparison:
    old_left = {kernel.id: kernel for kernel in old.kernels}
    new_left = {kernel.id: kernel for kernel in new.kernels}
    matches: list[Match] = []

    def add_pair(left: Kernel, right: Kernel, method: str, score: float,
                 alternatives: list[tuple[str, float]] | None = None) -> None:
        alternatives = alternatives or []
        matches.append(Match(
            left, right, _status(left, right), method,
            _confidence(method, score, alternatives), round(score, 4),
            diff_stats(left.source, right.source), alternatives,
        ))
        old_left.pop(left.id, None)
        new_left.pop(right.id, None)

    # 1. Stable identity is authoritative even for a total rewrite. A file can
    # legally define the same qualified name in multiple conditional branches,
    # so location is a multimap rather than a (path, qualname) dictionary.
    by_location: dict[tuple[str, str], list[Kernel]] = defaultdict(list)
    for right in new_left.values():
        by_location[(right.path, right.qualname)].append(right)
    proposals: list[tuple[float, Kernel, Kernel, list[tuple[str, float]]]] = []
    for left in old_left.values():
        ranked = sorted(((_score(left, item), item)
                         for item in by_location.get((left.path, left.qualname), [])),
                        reverse=True, key=lambda pair: pair[0])
        if ranked:
            for candidate_score, candidate in ranked:
                alternatives = [
                    (_location_key(item), round(score, 4))
                    for score, item in ranked if item.id != candidate.id
                ][:3]
                proposals.append((candidate_score, left, candidate, alternatives))
    for score, left, right, alternatives in sorted(proposals, reverse=True, key=lambda x: x[0]):
        if left.id in old_left and right.id in new_left:
            add_pair(left, right, "location", max(0.99, score), alternatives)

    # 2. Exact source/semantic matches reliably identify moves and renames.
    for attribute, method in (("source_sha256", "source_exact"),
                              ("semantic_sha256", "semantic_exact")):
        index: dict[str, list[Kernel]] = defaultdict(list)
        for right in new_left.values():
            index[getattr(right, attribute)].append(right)
        for left in list(old_left.values()):
            candidates = index.get(getattr(left, attribute), [])
            candidates = [item for item in candidates if item.id in new_left]
            if candidates:
                ranked = sorted(((_score(left, item), item) for item in candidates), reverse=True,
                                key=lambda pair: pair[0])
                best_score, best = ranked[0]
                alternatives = [(_location_key(item), round(score, 4)) for score, item in ranked[1:4]]
                add_pair(left, best, method, max(0.98, best_score), alternatives)

    # 3. Same-name candidates are common after file/module refactors.
    by_name: dict[str, list[Kernel]] = defaultdict(list)
    for right in new_left.values():
        by_name[right.name].append(right)
    proposals: list[tuple[float, Kernel, Kernel, list[tuple[str, float]]]] = []
    for left in old_left.values():
        ranked = sorted(((_score(left, item), item) for item in by_name.get(left.name, [])
                        if item.id in new_left), reverse=True, key=lambda pair: pair[0])
        if ranked and ranked[0][0] >= 0.48:
            for candidate_score, candidate in ranked:
                if candidate_score < 0.48:
                    continue
                alternatives = [
                    (_location_key(item), round(score, 4))
                    for score, item in ranked if item.id != candidate.id
                ][:3]
                proposals.append((candidate_score, left, candidate, alternatives))
    for score, left, right, alternatives in sorted(proposals, reverse=True, key=lambda x: x[0]):
        if left.id in old_left and right.id in new_left:
            add_pair(left, right, "same_name", score, alternatives)

    # 4. Best-effort fuzzy matching for simultaneous move + rename + edits.
    proposals = []
    for left in old_left.values():
        ranked = sorted(((_score(left, right), right) for right in new_left.values()),
                        reverse=True, key=lambda pair: pair[0])
        if ranked and ranked[0][0] >= fuzzy_threshold:
            for candidate_score, candidate in ranked:
                if candidate_score < fuzzy_threshold:
                    break
                alternatives = [
                    (_location_key(item), round(score, 4)) for score, item in ranked
                    if item.id != candidate.id and score >= fuzzy_threshold - 0.05
                ][:3]
                proposals.append((candidate_score, left, candidate, alternatives))
    for score, left, right, alternatives in sorted(proposals, reverse=True, key=lambda x: x[0]):
        if left.id in old_left and right.id in new_left:
            add_pair(left, right, "fuzzy", score, alternatives)

    for left in old_left.values():
        matches.append(Match(left, None, "removed", "unmatched", "high", 0.0,
                             diff_stats(left.source, "")))
    for right in new_left.values():
        matches.append(Match(None, right, "added", "unmatched", "high", 0.0,
                             diff_stats("", right.source)))
    matches.sort(key=lambda item: (
        item.status == "unchanged",
        (item.old or item.new).path,
        (item.old or item.new).decorator_line,
    ))
    return Comparison(old, new, matches)
