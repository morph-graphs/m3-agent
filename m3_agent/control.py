# Copyright (2025) Bytedance Ltd. and/or its affiliates

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# EPIC-Morph port. Same loop as upstream (prompt, instruction, up to total_round rounds, top-k clips per search,
# threshold 0.5, "character id" searches over 20 memories, sampling 0.6 / 0.95 / 20, 1024 new tokens), with:
# - paths and vLLM settings as arguments (--model, --tp, ...);
# - multiple choice: a question with "options" lists them after the question and asks for the letter; the
#   chosen letter is compared with "answer" (no judge). Upstream's GPT-4o judge for open answers is --judge;
# - every request seeded (reproducible sampling); searches run in this process (the embedding model is
#   loaded once) on a cached copy of the graph truncated at the question's before_clip;
# - each round's search and the clips it returned are kept (evidence-retrieval scoring), with the memory-graph walk
#   (retrieve.walk_trace: entry entity nodes, ranked text nodes, their entity edges); results are written
#   after every batch and a rerun skips the question ids already in --output;
# - --blind: the same prompt with no memory, one round that must answer (the no-memory floor);
# - a question still without [Answer] after the last round (the model often searches again in the round that
#   must answer) gets one more round with no search possible (--no_fallback_answer: upstream's protocol, where it
#   stays unanswered); "answered_in_loop" keeps upstream's outcome.
import re
import os
import sys
import copy
import json
import time
import random
import zlib
import argparse
import mmagent.videograph
from mmagent.retrieve import search, walk_trace
from mmagent.utils.general import load_video_graph

sys.modules["videograph"] = mmagent.videograph
processing_config = json.load(open("configs/processing_config.json"))

system_prompt = "You are given a question and some relevant knowledge. Your task is to reason about whether the provided knowledge is sufficient to answer the question. If it is sufficient, output [Answer] followed by the answer. If it is not sufficient, output [Search] and generate a query that will be encoded into embeddings for a vector similarity search. The query will help retrieve additional information from a memory bank.\n\nQuestion: {question}"
instruction = f"""

Output the answer in the format:
Action: [Answer] or [Search]
Content: {{content}}

If the answer cannot be derived yet, the {{content}} should be a single search query that would help retrieve the missing information. The search {{content}} needs to be different from the previous.
You can get the mapping relationship between character ID and name by using search query such as: "What is the name of <character_{{i}}>" or "What is the character id of {{name}}".
After obtaining the mapping, it is best to use character ID instead of name for searching.
If the answer can be derived from the provided knowledge, the {{content}} is the specific answer to the question. Only name can appear in the answer, not character ID like <character_{{i}}>."""
force_answer = "\n(The Action of this round must be [Answer]. If there is insufficient information, you can make reasonable guesses.)"
fallback_note = "\n(No more searches are possible. Output Action: [Answer] with the letter of the option you find most likely.)"
pattern = r"Action: \[([^\]]*)\].*Content: (.*)"   # upstream's had a greedy (.*) inside the brackets
LETTERS = "ABCDEFGH"


def mcq_question(qa):
    """The question as the control model sees it: with the options, when there are any."""
    if not qa.get("options"):
        return qa["question"]
    opts = "\n".join(f"({LETTERS[i]}) {o}" for i, o in enumerate(qa["options"]))
    return f"{qa['question']}\nOptions:\n{opts}\nAnswer with the letter of the correct option."


def chosen_letter(response, options):
    """The option the answer names: a leading letter ("B", "(B)", "[B]", "B. butter"), else the one option whose text
    the answer contains (longest match). None when neither."""
    if response is None:
        return None, "none"
    r = response.strip().strip("*").strip()
    n = len(options)
    # a letter alone, in brackets ("(B)", "[B]"), or followed by ".", ")", ":" or ","; not "A knife" (an article)
    m = re.match(r"^(?:option\s*)?(?:[\(\[]([A-H])[\)\]]|([A-H])(?:[\.\):,]|$))", r, re.IGNORECASE)
    if m:
        m = m.group(1) or m.group(2)
    if m and LETTERS.index(m.upper()) < n:
        return m.upper(), "letter"
    low = r.lower()
    hits = [(len(o), i) for i, o in enumerate(options) if str(o).lower() in low]
    if hits:
        return LETTERS[max(hits)[1]], "text"
    return None, "unparsed"


_graphs = {}


def truncated_graph(mem_path, before_clip):
    """The memory as it was after clip before_clip (all of it when None), cached per (memory, clip)."""
    key = (mem_path, before_clip)
    if key not in _graphs:
        if (mem_path, None) not in _graphs:
            _graphs[(mem_path, None)] = load_video_graph(mem_path)
        g = copy.deepcopy(_graphs[(mem_path, None)])
        random.seed(0)   # refresh_equivalences breaks ties at random
        if before_clip is not None:
            g.truncate_memory_by_clip(before_clip, False)
        g.refresh_equivalences()
        _graphs[key] = g
    return _graphs[key]


def consumer(data, args, allow_search=True):
    if data["finish"]:
        return data
    before_clip = data.get("before_clip", None)
    response = data["conversations"][-1]["content"]
    # the text after the reasoning; none when the reasoning was cut off (upstream then searched inside it)
    after = response.split("</think>", 1)[1] if "</think>" in response else ("" if "<think>" in response else response)
    match_result = re.search(pattern, after, re.DOTALL)
    if match_result:
        action = match_result.group(1)
        content = match_result.group(2)
    else:
        action = "Search"
        content = None
    if action == "Answer":
        data["response"] = content
        data["finish"] = True
        data["rounds"].append({"action": "answer", "content": content})
        return data
    new_memories = {}
    step = {"action": "search", "query": content, "clips": []}
    if not allow_search:   # the round that must answer, or the fallback round: no search, the question stays unanswered
        data["rounds"].append({"action": "search_refused", "query": content, "clips": []})
        return data
    if content and not args.blind:
        mem_node = truncated_graph(data["mem_path"], before_clip)
        if "character id" in content:
            memories, _, _ = search(mem_node, content, [], mem_wise=True, topk=20, before_clip=before_clip)
            new_memories.update(memories)
            step["mem_wise"] = True
        else:
            memories, currenr_clips, _ = search(mem_node, content, data["currenr_clips"], threshold=args.threshold, topk=args.topk, before_clip=before_clip)
            data["currenr_clips"] = currenr_clips
            new_memories.update(memories)
        step["clips"] = [int(k.split("_")[1]) for k in new_memories]
        try:   # logging only: never changes the answer
            step["trace"] = walk_trace(mem_node, new_memories)
        except Exception as e:
            step["trace"] = {"error": f"{type(e).__name__}: {e}"[:200]}
    data["rounds"].append(step)
    search_result = "Searched knowledge: " + json.dumps(new_memories, ensure_ascii=False).encode("utf-8", "ignore").decode("utf-8")
    if len(new_memories) == 0:
        search_result += "\n(The search result is empty. Please try searching from another perspective.)"
    data["conversations"].append({"role": "user", "content": search_result})
    return data


def judge(question, predict, ground_truth):
    """Upstream's GPT-4o judge for open-ended answers (--judge; needs configs/api_config.json)."""
    from mmagent.utils.chat_api import generate_messages, get_response_with_retry
    from mmagent.prompts import prompt_agent_verify_answer_referencing
    if not predict:
        return False
    messages = generate_messages([{"type": "text", "content": prompt_agent_verify_answer_referencing.format(
        question=question, ground_truth_answer=ground_truth, agent_answer=predict)}])
    return "yes" in get_response_with_retry("gpt-4o-2024-11-20", messages)[0].lower()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_file", type=str, default="data/annotations/robot.json")
    parser.add_argument("--output", type=str, default=None, help="jsonl, one line per question (default data/results/<name>.jsonl)")
    parser.add_argument("--model", type=str, default=os.environ.get("M3AGENT_CONTROL_CKPT", "models/M3-Agent-Control"))
    parser.add_argument("--tp", type=int, default=2, help="tensor parallel size (upstream: 2)")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9)
    parser.add_argument("--max_model_len", type=int, default=None)
    parser.add_argument("--total_round", type=int, default=processing_config["total_round"])
    parser.add_argument("--topk", type=int, default=processing_config["topk"])
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--blind", action="store_true", help="no memory: one round that must answer")
    parser.add_argument("--judge", action="store_true", help="grade open answers with GPT-4o (upstream)")
    parser.add_argument("--no_fallback_answer", dest="fallback_answer", action="store_false",
                        help="upstream's protocol: a question with no [Answer] after the last round stays unanswered")
    args = parser.parse_args()
    if args.blind:
        args.total_round = 1
    dataset_name = args.data_file.split("/")[-1].split(".")[0]
    output_path = args.output or os.path.join("data/results", f"{dataset_name}.jsonl")
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    done = set()
    if os.path.exists(output_path):
        # a row from before the walk trace ("traced") is answered again, unless blind (no searches to trace)
        done = {r["id"] for r in map(json.loads, filter(str.strip, open(output_path))) if args.blind or r.get("traced")}

    data = []
    datas = json.load(open(args.data_file))
    for _, v in datas.items():
        for qa in v["qa_list"]:
            if qa["question_id"] in done:
                continue
            d = {"id": qa["question_id"], "mem_path": v["mem_path"], "question": qa["question"], "answer": qa["answer"]}
            for k in ("options", "key", "before_clip"):
                if k in qa:
                    d[k] = qa[k]
            data.append(d)
    print(f"{len(data)} questions to answer ({len(done)} already in {output_path})", flush=True)
    if not data:
        return
    batch_size = processing_config["batch_size"]
    batched_datas = [data[i:i + batch_size] for i in range(0, len(data), batch_size)]

    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    kw = {"max_model_len": args.max_model_len} if args.max_model_len else {}
    model = LLM(model=args.model, tensor_parallel_size=args.tp, gpu_memory_utilization=args.gpu_memory_utilization, seed=args.seed, **kw)

    for batched_data in batched_datas:
        t = time.time()
        for d in batched_data:
            d["conversations"] = [{"role": "system", "content": system_prompt.format(question=mcq_question(d))}, {"role": "user", "content": "Searched knowledge: {}"}]
            d["finish"] = False
            d["currenr_clips"] = []
            d["rounds"] = []

        def step(idx, extra, allow_search=True):   # one round for every unfinished question
            vllm_inputs, params = [], []
            for d in batched_data:
                if d["finish"]:
                    continue
                d["conversations"][-1]["content"] += instruction + extra
                text = tokenizer.apply_chat_template(
                    d["conversations"],
                    tokenize=True,
                    add_generation_prompt=True,
                    enable_thinking=True
                )
                vllm_inputs.append({"prompt_token_ids": text})
                params.append(SamplingParams(temperature=0.6, top_p=0.95, top_k=20, max_tokens=1024,
                                             seed=zlib.crc32(f"{args.seed}/{d['id']}/{idx}".encode())))
            if not vllm_inputs:
                return batched_data
            outputs = model.generate(prompts=vllm_inputs, sampling_params=params, use_tqdm=False)
            i = 0
            for d in batched_data:
                if d["finish"]:
                    continue
                d["conversations"].append({"role": "assistant", "content": outputs[i].outputs[0].text})
                i += 1
            assert i == len(vllm_inputs)
            return [consumer(d, args, allow_search) for d in batched_data]

        for idx in range(args.total_round):
            last = idx == args.total_round - 1   # the round that must answer: a search there would not be read
            batched_data = step(idx, force_answer if last else "", allow_search=not last)
        for d in batched_data:
            d["answered_in_loop"] = d["finish"]   # upstream counts a question with no [Answer] by now as wrong
        if args.fallback_answer:
            # the control model often searches again in the round that must answer (always with no memory);
            # one more round, with no search possible, gives such a question an answer (marked in "rounds")
            batched_data = step(args.total_round, force_answer + fallback_note, allow_search=False)

        with open(output_path, "a") as f:
            for d in batched_data:
                d.pop("currenr_clips", None)
                d["traced"] = not args.blind   # each search round carries its "trace"
                if d.get("options"):
                    d["chosen"], d["parse"] = chosen_letter(d.get("response"), d["options"])
                    d["correct"] = d["chosen"] == d["answer"]
                elif args.judge:
                    d["gpt_eval"] = judge(d["question"], d.get("response"), d["answer"])
                f.write(json.dumps(d, ensure_ascii=False) + "\n")
        n = sum(bool(d.get("correct")) for d in batched_data)
        print(f"batch of {len(batched_data)}: {n} right, {time.time() - t:.0f} s", flush=True)


if __name__ == "__main__":
    main()
