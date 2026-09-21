#!/usr/bin/env python3
"""Deterministic structural retrieval for the Jian Ce routing index.

The index proposes candidates only. It never verifies quotations or replaces
opening the cited primary-source page.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parent.parent
INDEX_PATH = ROOT / "references" / "case-index.jsonl"

STAGES = {"choose", "explore", "wait", "exit", "scale"}
WORKS = {"史记", "资治通鉴"}
ARCHETYPES = {
    "platform-fit",
    "option-value",
    "timing-window",
    "information-quality",
    "feedback-loop",
    "trust-legitimacy",
    "reputation",
    "alliance",
    "coalition-stability",
    "power-asymmetry",
    "temporary-leverage",
    "talent-recognition",
    "delegation",
    "succession",
    "reform-change",
    "commitment",
    "escalation",
    "sunk-cost",
    "negotiation",
    "execution",
    "conflict-avoidance",
    "concentration-risk",
    "path-dependence",
    "principal-agent",
    "identity-status",
}
CONSTRAINTS = {
    "low-information",
    "low-feedback",
    "low-autonomy",
    "time-pressure",
    "irreversible",
    "resource-scarcity",
    "power-asymmetry",
    "reputation-at-stake",
    "coalition-dependent",
    "principal-agent",
    "winner-take-all",
    "temporary-leverage",
    "path-dependence",
    "hostile-environment",
    "repeated-failure",
    "mandate-change",
    "trust-deficit",
    "multiple-fronts",
    "moral-cost",
}
REQUIRED = {
    "id",
    "work",
    "title",
    "actor",
    "stage",
    "archetypes",
    "constraints",
    "action_pattern",
    "summary",
    "options",
    "known",
    "unknown",
    "outcome_hint",
    "cost_hints",
    "locator",
    "url",
    "anchors",
    "paired_with",
    "source_status",
}


def csv_set(value: str | None) -> set[str]:
    if not value:
        return set()
    return {part.strip() for part in value.split(",") if part.strip()}


def load_cases(path: Path = INDEX_PATH) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, raw in enumerate(handle, 1):
            line = raw.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"line {line_no}: invalid JSON: {exc}") from exc
            item["_line"] = line_no
            cases.append(item)
    return cases


def validate_cases(cases: list[dict[str, Any]]) -> list[str]:
    errors: list[str] = []
    ids: list[str] = [str(case.get("id", "")) for case in cases]
    id_set = set(ids)
    duplicate_ids = [case_id for case_id, count in Counter(ids).items() if count > 1]
    if duplicate_ids:
        errors.append(f"duplicate ids: {sorted(duplicate_ids)}")

    for case in cases:
        line = case.get("_line", "?")
        case_id = case.get("id", f"line-{line}")
        missing = sorted(REQUIRED - set(case))
        if missing:
            errors.append(f"{case_id}: missing {missing}")
        if case.get("stage") not in STAGES:
            errors.append(f"{case_id}: invalid stage {case.get('stage')!r}")
        if case.get("work") not in WORKS:
            errors.append(f"{case_id}: invalid work {case.get('work')!r}")
        if not isinstance(case.get("archetypes"), list) or not case.get("archetypes"):
            errors.append(f"{case_id}: archetypes must be a non-empty list")
        elif set(case["archetypes"]) - ARCHETYPES:
            errors.append(f"{case_id}: unknown archetypes {sorted(set(case['archetypes']) - ARCHETYPES)}")
        if not isinstance(case.get("constraints"), list):
            errors.append(f"{case_id}: constraints must be a list")
        elif set(case["constraints"]) - CONSTRAINTS:
            errors.append(f"{case_id}: unknown constraints {sorted(set(case['constraints']) - CONSTRAINTS)}")
        if not isinstance(case.get("options"), list) or len(case.get("options", [])) < 2:
            errors.append(f"{case_id}: at least two options required")
        if not isinstance(case.get("anchors"), list) or len(case.get("anchors", [])) < 2:
            errors.append(f"{case_id}: at least two anchors required")
        if case.get("source_status") != "route-only":
            errors.append(f"{case_id}: source_status must be 'route-only'")
        url = str(case.get("url", ""))
        if not url.startswith("https://zh.wikisource.org/zh-hans/"):
            errors.append(f"{case_id}: URL must use the primary source domain")
        costs = case.get("cost_hints", {})
        if not isinstance(costs, dict) or not isinstance(costs.get("explicit"), list) or not isinstance(costs.get("implicit"), list):
            errors.append(f"{case_id}: cost_hints needs explicit and implicit lists")
        pairs = case.get("paired_with", [])
        if not isinstance(pairs, list):
            errors.append(f"{case_id}: paired_with must be a list")
        else:
            for pair in pairs:
                if pair == case_id:
                    errors.append(f"{case_id}: cannot pair with itself")
                elif pair not in id_set:
                    errors.append(f"{case_id}: unknown pair {pair}")

    if {case.get("work") for case in cases} != WORKS:
        errors.append("index must contain both 史记 and 资治通鉴")
    return errors


def overlap_score(query: set[str], values: Iterable[str], weight: float) -> tuple[float, list[str]]:
    values_set = set(values)
    matched = sorted(query & values_set)
    if not query:
        return 0.0, matched
    union = query | values_set
    jaccard = len(matched) / len(union) if union else 0.0
    return weight * len(matched) + weight * 0.75 * jaccard, matched


def score_case(
    case: dict[str, Any],
    stage: str | None,
    archetypes: set[str],
    constraints: set[str],
    keywords: set[str],
) -> tuple[float, dict[str, Any]]:
    score = 0.0
    matched: dict[str, Any] = {"stage": False, "archetypes": [], "constraints": [], "keywords": []}

    if stage and case["stage"] == stage:
        score += 7.0
        matched["stage"] = True
    elif stage:
        score -= 1.25

    archetype_score, archetype_hits = overlap_score(archetypes, case["archetypes"], 3.0)
    constraint_score, constraint_hits = overlap_score(constraints, case["constraints"], 2.0)
    score += archetype_score + constraint_score
    matched["archetypes"] = archetype_hits
    matched["constraints"] = constraint_hits

    if keywords:
        haystack = " ".join(
            [
                case["title"],
                case["actor"],
                case["summary"],
                case["action_pattern"],
                " ".join(case["options"]),
                " ".join(case["anchors"]),
            ]
        ).lower()
        keyword_hits = sorted(word for word in keywords if word.lower() in haystack)
        score += 0.75 * len(keyword_hits)
        matched["keywords"] = keyword_hits

    # Require structural overlap when structural tags were supplied.
    structural_query = bool(archetypes or constraints)
    if structural_query and not (archetype_hits or constraint_hits):
        score -= 4.0

    return score, matched


def public_case(case: dict[str, Any], score: float | None = None, matched: dict[str, Any] | None = None) -> dict[str, Any]:
    result = {key: value for key, value in case.items() if key != "_line"}
    if score is not None:
        result = {
            "score": round(score, 3),
            "matched_on": matched or {},
            **result,
        }
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Search the Jian Ce structural case index.")
    parser.add_argument("--stage", choices=sorted(STAGES))
    parser.add_argument("--archetypes", help="Comma-separated ontology tags")
    parser.add_argument("--constraints", help="Comma-separated ontology tags")
    parser.add_argument("--keywords", help="Optional comma-separated routing words")
    parser.add_argument("--work", choices=["both", "史记", "资治通鉴"], default="both")
    parser.add_argument("--exclude", help="Comma-separated case ids to omit")
    parser.add_argument("--paired-for", help="Return the cases paired with this case id")
    parser.add_argument("--ids", help="Return exact comma-separated case ids")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--validate", action="store_true", help="Validate the catalog and exit")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        cases = load_cases()
    except (OSError, ValueError) as exc:
        print(json.dumps({"ok": False, "errors": [str(exc)]}, ensure_ascii=False, indent=2))
        return 2

    errors = validate_cases(cases)
    if args.validate:
        payload = {
            "ok": not errors,
            "case_count": len(cases),
            "works": dict(Counter(case["work"] for case in cases)),
            "stages": dict(Counter(case["stage"] for case in cases)),
            "errors": errors,
        }
        print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
        return 0 if not errors else 1
    if errors:
        print(json.dumps({"ok": False, "errors": errors}, ensure_ascii=False, indent=2))
        return 2

    by_id = {case["id"]: case for case in cases}
    if args.paired_for:
        source = by_id.get(args.paired_for)
        if not source:
            print(json.dumps({"ok": False, "error": f"unknown case id: {args.paired_for}"}, ensure_ascii=False, indent=2))
            return 2
        results = [public_case(by_id[pair]) for pair in source["paired_with"]]
        print(json.dumps({"ok": True, "paired_for": args.paired_for, "results": results}, ensure_ascii=False, indent=2))
        return 0

    exact_ids = csv_set(args.ids)
    if exact_ids:
        missing = sorted(exact_ids - set(by_id))
        results = [public_case(case) for case_id, case in by_id.items() if case_id in exact_ids]
        print(json.dumps({"ok": not missing, "missing": missing, "results": results}, ensure_ascii=False, indent=2))
        return 0 if not missing else 1

    archetypes = csv_set(args.archetypes)
    constraints = csv_set(args.constraints)
    keywords = csv_set(args.keywords)
    excluded = csv_set(args.exclude)
    top_k = max(1, min(args.top_k, 50))

    unknown_archetypes = sorted(archetypes - ARCHETYPES)
    unknown_constraints = sorted(constraints - CONSTRAINTS)
    if unknown_archetypes or unknown_constraints:
        print(
            json.dumps(
                {
                    "ok": False,
                    "error": "unknown ontology tags",
                    "unknown_archetypes": unknown_archetypes,
                    "unknown_constraints": unknown_constraints,
                    "allowed_archetypes": sorted(ARCHETYPES),
                    "allowed_constraints": sorted(CONSTRAINTS),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 2

    ranked: list[tuple[float, str, dict[str, Any], dict[str, Any]]] = []
    for case in cases:
        if case["id"] in excluded:
            continue
        if args.work != "both" and case["work"] != args.work:
            continue
        score, matched = score_case(case, args.stage, archetypes, constraints, keywords)
        ranked.append((score, case["id"], case, matched))
    ranked.sort(key=lambda item: (-item[0], item[1]))

    results = [public_case(case, score, matched) for score, _, case, matched in ranked[:top_k]]
    payload = {
        "ok": True,
        "notice": "候选索引，不是史料核验；正式采用前必须打开原文页。",
        "request": {
            "stage": args.stage,
            "archetypes": sorted(archetypes),
            "constraints": sorted(constraints),
            "keywords": sorted(keywords),
            "work": args.work,
            "top_k": top_k,
        },
        "results": results,
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
