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
import os
import json
import logging
import argparse
import glob
import pickle
import random
import time

import numpy as np

from mmagent.videograph import VideoGraph
from mmagent.utils.video_processing import process_video_clip
from mmagent.face_processing import process_faces
from mmagent.voice_processing import process_voices
from mmagent.memory_processing_qwen import process_memories, generate_memories

logger = logging.getLogger(__name__)
processing_config = json.load(open("configs/processing_config.json"))
memory_config = json.load(open("configs/memory_config.json"))

preprocessing = []

def process_segment(
    video_graph,
    base64_video,
    base64_frames,
    base64_audio,
    clip_id,
    sample,
    clip_path
):
    save_path = sample["intermediate_outputs"]
    
    id2voices = process_voices(
        video_graph,
        base64_audio,
        base64_video,
        save_path=os.path.join(save_path, f"clip_{clip_id}_voices.json"),
        preprocessing=[],
    )

    id2faces = process_faces(
        video_graph,
        base64_frames,
        save_path=os.path.join(save_path, f"clip_{clip_id}_faces.json"),
        preprocessing=[],
    )

    episodic_memories, semantic_memories = generate_memories(
        base64_frames,
        id2faces,
        id2voices,
        clip_path,
    )

    process_memories(video_graph, episodic_memories, clip_id, type="episodic")
    process_memories(video_graph, semantic_memories, clip_id, type="semantic")

def streaming_process_video(video_graph, sample):
    """Process the clips of one video in order and save the memory graph to sample["mem_path"].

    EPIC-Morph port: clips are taken in clip-id order (upstream's unsorted glob made node ids, and so truncation
    by clip, depend on directory order); after each clip the graph and the RNG state are saved to
    sample["partial_path"] (default <mem_path>.partial), and a rerun resumes from there (preemption); a JSON dump of the text memories is written
    next to the graph (sample["json_path"], when given)."""
    clips = sorted(glob.glob(sample["clip_path"] + "/*.mp4"), key=lambda p: int(os.path.basename(p).split(".")[0]))
    partial = sample.get("partial_path", sample["mem_path"] + ".partial")
    done = -1
    random.seed(0)
    np.random.seed(0)
    if os.path.exists(partial):
        with open(partial, "rb") as f:
            state = pickle.load(f)
        video_graph, done = state["graph"], state["clip_id"]
        random.setstate(state["random"])
        np.random.set_state(state["np_random"])
        logger.warning(f"resuming {sample['id']} after clip {done}")
    for clip_path in clips:
        clip_id = int(clip_path.split("/")[-1].split(".")[0])
        if clip_id <= done:
            continue
        t = time.time()
        base64_video, base64_frames, base64_audio = process_video_clip(clip_path)

        # Process frames for this interval
        if base64_frames:
            process_segment(
                video_graph,
                base64_video,
                base64_frames,
                base64_audio,
                clip_id,
                sample,
                clip_path
            )
        with open(partial + ".tmp", "wb") as f:
            pickle.dump({"graph": video_graph, "clip_id": clip_id, "random": random.getstate(),
                         "np_random": np.random.get_state()}, f)
        os.replace(partial + ".tmp", partial)
        n_text = len(video_graph.text_nodes_by_clip.get(clip_id, []))
        print(f"clip {clip_id + 1}/{len(clips)}: {n_text} memories, {len(video_graph.nodes)} nodes, {time.time() - t:.0f} s", flush=True)

    video_graph.refresh_equivalences()
    with open(sample["mem_path"] + ".tmp", "wb") as f:
        pickle.dump(video_graph, f)
    if sample.get("json_path"):
        dump_memories(video_graph, sample["json_path"])
    os.replace(sample["mem_path"] + ".tmp", sample["mem_path"])
    os.remove(partial)


def dump_memories(video_graph, path):
    """The graph's text memories per clip, readable (EPIC-Morph port)."""
    clips = {}
    for clip_id in sorted(video_graph.text_nodes_by_clip):
        nodes = [video_graph.nodes[n] for n in video_graph.text_nodes_by_clip[clip_id]]
        clips[f"CLIP_{clip_id}"] = {t: [n.metadata["contents"][0] for n in nodes if n.type == t] for t in ("episodic", "semantic")}
    out = {"n_nodes": len(video_graph.nodes),
           "n_faces": sum(n.type == "img" for n in video_graph.nodes.values()),
           "n_voices": sum(n.type == "voice" for n in video_graph.nodes.values()),
           "characters": video_graph.character_mappings, "clips": clips}
    with open(path, "w") as f:
        json.dump(out, f, indent=1, ensure_ascii=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_file", type=str, default="data/data.jsonl")
    args = parser.parse_args()
    video_inputs = []
    
    with open(args.data_file, "r") as f:
        for line in f:
            sample = json.loads(line)
            if not os.path.exists(sample["mem_path"]):
                video_graph = VideoGraph(**memory_config)
                streaming_process_video(video_graph, sample)
