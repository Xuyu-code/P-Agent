"""Offline evaluation: intent routing / QA answerability / slot extraction.

Runs against the REAL LLM and the REAL local vector store (no image-generation
calls; generation is treated as an external service).
Numbers are only meaningful if actually run; the report records model, date
and every per-case detail so failures stay visible.

Usage (env vars as in start_agent.bat):
    python -m eval.run_eval [--only routing|qa|slots] [--sleep 8] [--heldout]

Free-tier Kimi keys are capped at 3 RPM: llm.chat retries on 429 and the
runner sleeps between cases. Expect ~15-20 minutes for the full set.

--heldout runs the same evaluators against eval/datasets/heldout_*.jsonl —
the frozen generalization set. Held-out results are reported as-is: cases
there must NOT be used to tune prompts/thresholds (that would re-create the
train-on-test problem the set exists to avoid).
"""

from __future__ import annotations

import argparse
import datetime
import json
import time
from pathlib import Path

from langchain_core.messages import AIMessage, HumanMessage

from agent_project import config, llm
from agent_project.graph import nodes

EVAL_ROOT = Path(__file__).resolve().parent
DATASETS = EVAL_ROOT / "datasets"
RESULTS = EVAL_ROOT / "results"

# Boundary phrasings observed in real answers (answer + report both keep the
# full text, so a keyword miss is auditable rather than hidden).
REFUSAL_MARKERS = [
    "没有支持", "无法回答", "不能凭空", "无法确认", "没有证据", "未提供",
    "建议补充", "资料中没", "未提及", "没有相关信息", "无法证实", "不建议",
]


def load_jsonl(name: str) -> list[dict]:
    return [json.loads(line) for line in (DATASETS / name).read_text(encoding="utf-8").splitlines() if line.strip()]


def is_refusal(answer: str) -> bool:
    return any(m in answer for m in REFUSAL_MARKERS)


def _tokens_since(snapshot: dict) -> dict:
    """Token delta between a usage_summary() snapshot and now (per case)."""
    now = llm.usage_summary()
    return {k: (now.get(k) or 0) - (snapshot.get(k) or 0)
            for k in ("calls", "prompt_tokens", "completion_tokens", "total_tokens")}


# ---------------------------------------------------------------------------
# Evaluators
# ---------------------------------------------------------------------------

def eval_routing(sleep: float, dataset: str = "intent_routing.jsonl") -> list[dict]:
    records = []
    for case in load_jsonl(dataset):
        messages = []
        for ctx in case.get("context", []):
            role, _, content = ctx.partition(":")
            messages.append(HumanMessage(content=content) if role == "user" else AIMessage(content=content))
        messages.append(HumanMessage(content=case["text"]))
        usage_before = llm.usage_summary()
        started = time.perf_counter()
        try:
            out = nodes.route_intent({"messages": messages})
        except Exception as exc:  # 网络/限流等基础设施故障：单列，不算模型判错
            records.append({"text": case["text"], "expect": case["expect"], "got": None,
                            "ok": False, "infra_error": repr(exc)[:200],
                            "latency_s": round(time.perf_counter() - started, 1),
                            "tokens": _tokens_since(usage_before)})
            print(f"  [!] {case['text'][:30]} 基础设施错误：{exc!r}"[:120])
            time.sleep(sleep)
            continue
        latency = time.perf_counter() - started
        got = out["intent"]
        records.append({"text": case["text"], "expect": case["expect"], "got": got,
                        "ok": got == case["expect"], "latency_s": round(latency, 1),
                        "tokens": _tokens_since(usage_before)})
        print(f"  [{'✓' if got == case['expect'] else '✗'}] {case['text'][:30]} -> {got}（期望 {case['expect']}）")
        time.sleep(sleep)
    return records


def eval_qa(sleep: float, dataset: str = "qa_answerability.jsonl") -> list[dict]:
    records = []
    for case in load_jsonl(dataset):
        usage_before = llm.usage_summary()
        started = time.perf_counter()
        try:
            out = nodes.qa_node({"messages": [HumanMessage(content=case["question"])]})
        except Exception as exc:  # 同上：基础设施故障单列
            records.append({"question": case["question"], "expect": case["expect"],
                            "ok": False, "infra_error": repr(exc)[:200],
                            "latency_s": round(time.perf_counter() - started, 1),
                            "tokens": _tokens_since(usage_before)})
            print(f"  [!] {case['question'][:30]} 基础设施错误：{exc!r}"[:120])
            time.sleep(sleep)
            continue
        latency = time.perf_counter() - started
        answer = out.get("answer", "")
        cited_titles = [c.get("title", "") for c in out.get("citations", [])]
        refused = is_refusal(answer)
        if case["expect"] == "refuse":
            ok = refused
        else:
            # 应答题以"引用到指定来源且给出实质回答"为准；限定语（"据…报道"）
            # 或部分边界说明不算拒答——evidence-bounded 的合法回答常含 hedging。
            # must_cite_any：多卡可独立支撑的问题，引用任一指定来源即合规。
            has_citation = bool(out.get("citations")) and "【" in answer
            accepted = case.get("must_cite_any") or [case.get("must_cite", "")]
            cited_ok = any(any(acc in t for acc in accepted) for t in cited_titles)
            ok = has_citation and cited_ok
        records.append({"question": case["question"], "expect": case["expect"], "refused": refused,
                        "hedged": refused and case["expect"] == "answer",
                        "cited_titles": cited_titles, "ok": ok, "latency_s": round(latency, 1),
                        "tokens": _tokens_since(usage_before),
                        "answer": answer})
        print(f"  [{'✓' if ok else '✗'}] {case['question'][:30]}（期望 {case['expect']}，实际 {'拒答' if refused else '回答'}）")
        time.sleep(sleep)
    return records


def eval_slots(sleep: float, dataset: str = "slot_extraction.jsonl") -> list[dict]:
    records = []
    for case in load_jsonl(dataset):
        usage_before = llm.usage_summary()
        started = time.perf_counter()
        try:
            slots = nodes._extract_slots(case["text"])
        except Exception as exc:  # 同上：基础设施故障单列
            records.append({"text": case["text"], "fields": {}, "ok": False,
                            "infra_error": repr(exc)[:200],
                            "latency_s": round(time.perf_counter() - started, 1),
                            "tokens": _tokens_since(usage_before)})
            print(f"  [!] {case['text'][:30]} 基础设施错误：{exc!r}"[:120])
            time.sleep(sleep)
            continue
        latency = time.perf_counter() - started
        field_results = {}
        for field, expected_sub in case["expect_slots"].items():
            got = getattr(slots, field, None)
            field_results[field] = {"expect": expected_sub, "got": got,
                                    "ok": bool(got) and expected_sub in str(got)}
        ok = all(f["ok"] for f in field_results.values())
        records.append({"text": case["text"], "fields": field_results, "ok": ok,
                        "latency_s": round(latency, 1), "tokens": _tokens_since(usage_before)})
        print(f"  [{'✓' if ok else '✗'}] {case['text'][:30]} -> { {k: v['got'] for k, v in field_results.items()} }")
        time.sleep(sleep)
    return records


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def summarize(name: str, records: list[dict]) -> dict:
    total = len(records)
    passed = sum(1 for r in records if r["ok"])
    infra = sum(1 for r in records if r.get("infra_error"))
    scored = total - infra  # 基础设施故障不计入模型准确率（报告单列）
    latencies = [r["latency_s"] for r in records]
    summary = {"name": name, "total": total, "passed": passed, "infra_errors": infra,
               "accuracy": round(passed / scored, 4) if scored else None,
               "avg_latency_s": round(sum(latencies) / len(latencies), 1) if latencies else None}
    token_lists = [r["tokens"] for r in records if r.get("tokens")]
    if token_lists:
        prompt = sum(t["prompt_tokens"] for t in token_lists)
        completion = sum(t["completion_tokens"] for t in token_lists)
        summary["tokens"] = {
            "prompt_total": prompt,
            "completion_total": completion,
            "avg_prompt_per_case": round(prompt / len(token_lists)),
            "avg_completion_per_case": round(completion / len(token_lists)),
            "avg_total_per_case": round((prompt + completion) / len(token_lists)),
        }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", choices=["routing", "qa", "slots"], default=None)
    parser.add_argument("--sleep", type=float, default=8.0)
    parser.add_argument("--heldout", action="store_true",
                        help="run against the frozen heldout_*.jsonl generalization sets")
    args = parser.parse_args()

    RESULTS.mkdir(exist_ok=True)
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    prefix = "heldout_" if args.heldout else ""
    suites = {
        "routing": (eval_routing, f"{prefix}intent_routing.jsonl"),
        "qa": (eval_qa, f"{prefix}qa_answerability.jsonl"),
        "slots": (eval_slots, f"{prefix}slot_extraction.jsonl"),
    }
    if args.only:
        suites = {args.only: suites[args.only]}

    llm.reset_usage()
    all_records = {}
    for name, (fn, dataset) in suites.items():
        print(f"\n=== {name}（{dataset}）===")
        all_records[name] = fn(args.sleep, dataset)

    summaries = [summarize(name, recs) for name, recs in all_records.items()]
    usage = llm.usage_summary()
    report = {
        "date": stamp,
        "heldout": args.heldout,
        "llm_model": config.LLM_MODEL,
        "llm_base_url": config.LLM_BASE_URL,
        "embedding": f"{config.EMBEDDING_PROVIDER}/{config.LOCAL_EMBEDDING_MODEL if config.EMBEDDING_PROVIDER == 'local' else config.EMBEDDING_MODEL}",
        "note": "真实运行结果；temperature=1（kimi-k2.6 限制），重跑数字可能小幅波动。",
        "summaries": summaries,
        "usage_total": usage,
        "records": all_records,
    }
    json_path = RESULTS / f"{prefix}eval_{stamp}.json"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [f"# Agent {'held-out 泛化' if args.heldout else '离线'}评测报告 {stamp}", "",
             f"- LLM：{config.LLM_MODEL}（{config.LLM_BASE_URL}）",
             f"- 嵌入：{report['embedding']}", f"- 说明：{report['note']}",
             f"- 全程 LLM 调用 {usage['calls']} 次，token 用量：prompt {usage['prompt_tokens']} / completion {usage['completion_tokens']} / total {usage['total_tokens']}", ""]
    for s in summaries:
        tok = s.get("tokens")
        tok_txt = (f"，token/例：prompt {tok['avg_prompt_per_case']} + completion {tok['avg_completion_per_case']}"
                   f" = {tok['avg_total_per_case']}") if tok else ""
        acc = f"{s['accuracy']:.0%}" if s["accuracy"] is not None else "n/a"
        infra_txt = f"（另有 {s['infra_errors']} 例基础设施错误，未计入准确率）" if s.get("infra_errors") else ""
        lines.append(f"## {s['name']}: {s['passed']}/{s['total']}（{acc}），平均延迟 {s['avg_latency_s']}s{tok_txt}{infra_txt}")
        for r in all_records[s["name"]]:
            if not r["ok"]:
                lines.append(f"- ✗ {json.dumps(r, ensure_ascii=False)}")
        lines.append("")
    md_path = RESULTS / f"{prefix}eval_{stamp}.md"
    md_path.write_text("\n".join(lines), encoding="utf-8")

    print("\n=== 汇总 ===")
    for s in summaries:
        acc = f"{s['accuracy']:.0%}" if s["accuracy"] is not None else "n/a"
        print(f"{s['name']}: {s['passed']}/{s['total']} = {acc}"
              + (f"（{s['infra_errors']} 例基础设施错误）" if s.get("infra_errors") else ""))
    print(f"token 总量：{usage['total_tokens']}（prompt {usage['prompt_tokens']} + completion {usage['completion_tokens']}）")
    print(f"\n报告：{md_path}")


if __name__ == "__main__":
    main()
