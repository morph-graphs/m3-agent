# Local stand-ins for the closed APIs upstream calls (EPIC-Morph port; M3AGENT_BACKEND=local is the default,
# M3AGENT_BACKEND=api restores the Azure OpenAI / Gemini calls of configs/api_config.json).
#
# - embed(): Qwen3-Embedding-4B (M3AGENT_EMBED_MODEL) instead of OpenAI text-embedding-3-large, unit-normalised.
#   Search queries get Qwen3-Embedding's query instruction; memories are embedded as plain text. The
#   memorization environment pins a transformers commit without Qwen3, so there the model runs in a helper
#   process with another Python (M3AGENT_EMBED_PY, the control environment's): `python local_backends.py --serve`.
#   M3AGENT_EMBED_DEVICE (default cuda when there is one) puts it elsewhere, e.g. cpu next to vLLM on 24 GB cards.
# - transcribe(): faster-whisper (M3AGENT_ASR_MODEL) instead of Gemini-1.5-Pro's ASR + "diarization". Upstream
#   asks Gemini only for speech segments (start, end, text) split at speaker turns, with no speaker labels; the
#   speaker identity comes from ERes2NetV2 voice embeddings in both versions. Whisper splits at pauses instead
#   of speaker turns.
import base64
import json
import math
import os
import subprocess
import sys
import threading

import numpy as np

EMBED_MODEL = os.environ.get("M3AGENT_EMBED_MODEL", "Qwen/Qwen3-Embedding-4B")
ASR_MODEL = os.environ.get("M3AGENT_ASR_MODEL", "large-v3-turbo")
QUERY_PROMPT = "Instruct: Given a search query about a video, retrieve the memories of the video that answer it\nQuery: "

_st = None        # in-process sentence-transformers model
_helper = None    # helper process (memorization environment)
_whisper = None
_lock = threading.Lock()


def _load_st():
    import torch
    from sentence_transformers import SentenceTransformer
    dev = os.environ.get("M3AGENT_EMBED_DEVICE") or ("cuda" if torch.cuda.is_available() else "cpu")
    if dev == "cpu":   # vLLM sets the process to one torch thread; the embedding model wants all the job's cores
        torch.set_num_threads(int(os.environ.get("SLURM_CPUS_PER_TASK") or os.cpu_count()))
    return SentenceTransformer(EMBED_MODEL, device=dev, model_kwargs={"torch_dtype": torch.bfloat16 if dev == "cuda" else torch.float32})


def _encode(model, texts, query):
    emb = model.encode(list(texts), batch_size=16, normalize_embeddings=True, prompt=QUERY_PROMPT if query else None,
                       convert_to_numpy=True, show_progress_bar=False)
    return emb.astype(np.float32)


def _in_process():
    try:
        from transformers import Qwen3Config  # noqa: F401  (absent from the memorization pin)
        import sentence_transformers  # noqa: F401
        return True
    except ImportError:
        return False


def embed(texts, query=False):
    """List of float32 vectors (unit norm), one per text."""
    global _st, _helper
    if len(texts) == 0:
        return []
    if _st is None and _helper is None:
        if _in_process():
            _st = _load_st()
        else:
            py = os.environ.get("M3AGENT_EMBED_PY")
            if not py:
                raise RuntimeError("this environment cannot load Qwen3-Embedding: set M3AGENT_EMBED_PY to the control environment's python")
            _helper = subprocess.Popen([py, os.path.abspath(__file__), "--serve"], stdin=subprocess.PIPE,
                                       stdout=subprocess.PIPE, text=True, bufsize=1)
    if _st is not None:
        return list(_encode(_st, texts, query))
    with _lock:   # one request at a time on the pipe
        _helper.stdin.write(json.dumps({"texts": list(texts), "query": bool(query)}) + "\n")
        _helper.stdin.flush()
        line = _helper.stdout.readline()
    if not line:
        raise RuntimeError("embedding helper process died")
    r = json.loads(line)
    arr = np.frombuffer(base64.b64decode(r["emb"]), dtype=np.float32).reshape(r["shape"])
    return list(arr.copy())


def _serve():
    # the protocol gets its own copy of fd 1; fd 1 itself (Python and C-level prints) goes to stderr
    out = os.fdopen(os.dup(1), "w", buffering=1)
    os.dup2(2, 1)
    sys.stdout = sys.stderr
    model = _load_st()
    for line in sys.stdin:
        req = json.loads(line)
        emb = _encode(model, req["texts"], req["query"])
        out.write(json.dumps({"shape": list(emb.shape), "emb": base64.b64encode(emb.tobytes()).decode()}) + "\n")
        out.flush()


def _mmss(sec):
    sec = int(sec)
    return f"{sec // 60:02d}:{sec % 60:02d}"


def transcribe(wav_path, duration):
    """Speech segments of a clip's audio as upstream's Gemini prompt returns them:
    [{"start_time": "MM:SS", "end_time": "MM:SS", "asr": text}], times within the clip."""
    global _whisper
    if _whisper is None:
        import ctypes
        import glob
        import torch
        from faster_whisper import WhisperModel
        cuda = torch.cuda.is_available()
        if cuda:   # CTranslate2 dlopens cuBLAS / cuDNN by soname: load torch's pip copies first (not on LD_LIBRARY_PATH)
            import nvidia.cublas, nvidia.cudnn
            libs = [l for d in (nvidia.cublas.__path__[0], nvidia.cudnn.__path__[0])
                    for l in sorted(glob.glob(os.path.join(d, "lib", "lib*.so.*")))]
            for attempt in (0, 1):   # a second pass for libraries whose dependencies came later in the list
                for lib in libs:
                    try:
                        ctypes.CDLL(lib, mode=ctypes.RTLD_GLOBAL)
                    except OSError:
                        if attempt:
                            raise
        _whisper = WhisperModel(ASR_MODEL, device="cuda" if cuda else "cpu", compute_type="float16" if cuda else "int8")
    segs, _ = _whisper.transcribe(wav_path, beam_size=5, vad_filter=True, condition_on_previous_text=False)
    out = []
    for s in segs:
        text = s.text.strip()
        if not text or s.no_speech_prob > 0.6:
            continue
        end = min(math.ceil(s.end), int(duration))   # upstream drops segments that end past the audio
        out.append({"start_time": _mmss(math.floor(s.start)), "end_time": _mmss(end), "asr": text})
    return out


if __name__ == "__main__" and "--serve" in sys.argv:
    _serve()
