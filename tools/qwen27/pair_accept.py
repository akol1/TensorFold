#!/usr/bin/env python3
"""Accept a paired TensorFold serve against the checked-in single-host receipts.

run   send every oracle case twice; both replies must be identical, and either the oracle's token_sha or the
      case's recorded pair_known one
warm  one chat turn, then that conversation's next turn: the reply must report cached prompt tokens

The oracle (qwen38_27b_oracle.json, next to this file) is the single-Mac, undrafted, temperature-0 record.
A pair passes when the same token ids come out, so neither Mac's rank count nor the seam can show up in the text.
Where a case records pair_known, the pair is accepted at that one value and a third value fails: the pair's
seam sums can decide between two candidates closer together than a bf16 logit step.
"""
import argparse
import json
import sys
import urllib.error
import urllib.request

DEFAULT_MODEL = "qwen3.8-27b"


def post(base, path, body, timeout):
    request = urllib.request.Request(
        base.rstrip("/") + path,
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read())


def reply_text(reply, endpoint):
    choice = reply["choices"][0]
    if endpoint.endswith("/chat/completions"):
        return choice.get("message", {}).get("content") or ""
    return choice.get("text", "")


def token_sha(reply):
    return reply.get("tensorfold", {}).get("token_sha")


def request_body(case, model):
    body = {
        "model": model,
        "max_tokens": case["max_tokens"],
        "temperature": 0.0,
        "seed": 1234,
    }
    if case["endpoint"].endswith("/chat/completions"):
        body["messages"] = [{"role": "user", "content": case["prompt"]}]
        body["chat_template_kwargs"] = {"enable_thinking": bool(case.get("thinking", False))}
    else:
        body["prompt"] = case["prompt"]
    return body


def load_oracle(path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def run(base, oracle, model, repeat, timeout, only):
    cases = [c for c in oracle["cases"] if only is None or c["name"] in only]
    if not cases:
        print("pair_accept: no case matches --only", file=sys.stderr)
        return 2
    failures = 0
    recorded = 0
    for case in cases:
        body = request_body(case, model)
        shas = []
        texts = []
        for _ in range(repeat):
            reply = post(base, case["endpoint"], body, timeout)
            shas.append(token_sha(reply))
            texts.append(reply_text(reply, case["endpoint"]))
        same = len(set(shas)) == 1 and len(set(texts)) == 1
        known = (case.get("pair_known") or {}).get("token_sha")
        if same and shas[0] == case["token_sha"]:
            print("PASS {} token_sha={} ({} identical runs)".format(case["name"], shas[0], repeat))
        elif same and known is not None and shas[0] == known:
            recorded += 1
            print("KNOWN {} token_sha={} is the pair value recorded for this case, not the oracle's {}".format(case["name"], shas[0], case["token_sha"]))
        else:
            failures += 1
            if not same:
                why = "the {} runs differ".format(repeat)
            elif known is None:
                why = "token_sha {} != oracle {}".format(shas[0], case["token_sha"])
            else:
                why = "token_sha {} is neither the oracle's {} nor the recorded pair value {}".format(shas[0], case["token_sha"], known)
            print("FAIL {} {} shas={}".format(case["name"], why, ",".join(str(s) for s in shas)))
    total = len(cases)
    print("pair_accept: {}/{} cases match the oracle, {} recorded as different, {} unexpected on {}".format(total - failures - recorded, total, recorded, failures, base))
    return 0 if failures == 0 else 1


def cached_tokens(reply):
    usage = reply.get("usage") or {}
    details = usage.get("prompt_tokens_details") or {}
    value = details.get("cached_tokens")
    return value if value is not None else usage.get("cached_tokens")


def warm(base, oracle, model, timeout):
    case = next((c for c in oracle["cases"] if c["name"].startswith("chat")), None)
    if case is None:
        print("pair_accept: the oracle has no chat case to build a conversation from", file=sys.stderr)
        return 2
    first = post(base, case["endpoint"], request_body(case, model), timeout)
    used = cached_tokens(first)
    text = reply_text(first, case["endpoint"])
    if not text:
        print("FAIL warm: the first turn returned no text", file=sys.stderr)
        return 1
    second_body = {
        "model": model,
        "max_tokens": 32,
        "temperature": 0.0,
        "seed": 1234,
        "messages": [
            {"role": "user", "content": case["prompt"]},
            {"role": "assistant", "content": text},
            {"role": "user", "content": "Now give the same example for a 3 by 3 matrix."},
        ],
        "chat_template_kwargs": {"enable_thinking": False},
    }
    second = post(base, case["endpoint"], second_body, timeout)
    cached = cached_tokens(second)
    print("warm: turn 1 prefill marked {}; turn 2 cached={}".format(used, cached))
    if cached is None or cached <= 0:
        print("FAIL warm: the next turn reused no prompt state (cached={})".format(cached))
        return 1
    print("PASS warm: the next turn resumed {} cached prompt tokens".format(cached))
    return 0


def main():
    parser = argparse.ArgumentParser(description="Accept a paired TensorFold serve against single-host receipts")
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("run", "warm"):
        p = sub.add_parser(name)
        p.add_argument("--base-url", default="http://127.0.0.1:4100")
        p.add_argument("--oracle", default="tools/qwen27/qwen38_27b_oracle.json")
        p.add_argument("--model", default=DEFAULT_MODEL)
        p.add_argument("--timeout", type=float, default=600.0)
        if name == "run":
            p.add_argument("--repeat", type=int, default=2)
            p.add_argument("--only", nargs="*", default=None)
    args = parser.parse_args()
    oracle = load_oracle(args.oracle)
    try:
        if args.command == "run":
            return run(args.base_url, oracle, args.model, args.repeat, args.timeout, args.only)
        return warm(args.base_url, oracle, args.model, args.timeout)
    except urllib.error.URLError as err:
        print("pair_accept: {} did not answer: {}".format(args.base_url, err), file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
