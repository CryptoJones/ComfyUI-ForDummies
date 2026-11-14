# AGENTS.md — Image Studio frontend (webui.py)

Guide for AI agents deploying or modifying this frontend on an installed ComfyUI.
README.md is the human-facing overview; this file is the operating manual.

## What this is

`webui.py` is a self-contained aiohttp app: HTML/JS frontend + JSON API + image
proxy in one file. It talks to a running ComfyUI over HTTP (default
`http://127.0.0.1:8188`) and serves everything itself on port 8191. It is **not**
a ComfyUI custom node — never put it in `custom_nodes/`.

## Deploy on an installed ComfyUI

1. Verify ComfyUI is up: `curl -s http://127.0.0.1:8188/queue` must return JSON.
2. Copy `webui.py` to the ComfyUI root (next to `main.py`). `LORA_DIR` and
   `OUTPUT_DIR` resolve relative to the file, so placement matters.
3. Dependencies: `aiohttp` and `pillow` in the ComfyUI venv
   (`.venv/bin/pip install aiohttp pillow`). Nothing else.
4. Set the constants at the top of `webui.py` (below) for the models actually staged.
5. Run `.venv/bin/python webui.py` (binds 0.0.0.0:8191), or install the sample
   systemd user units from `deploy/`.
6. Verify: `GET /` returns 200 and contains `id="engine"`; `GET /api/queue`
   returns JSON with `running_count`.

## Configuration (constants, not files)

webui.py reads no config file and no environment variables at runtime.
Everything is a module-level constant:

| Constant | Meaning |
|---|---|
| `COMFY` | ComfyUI base URL (default `http://127.0.0.1:8188`) |
| `QWEN21_REMOTE_ENC` | Optional remote Qwen encoder service URL; `""` = local CPU encode |
| `*_UNET` / `*_CLIP` / `*_VAE` / `*_CKPT` | Model filenames per engine — generic placeholders, set your own |
| `DEFAULT_PROMPT` / `DEFAULT_NEG` | Prompt-box prefill (shipped empty on purpose) |

Model filenames ship as generic placeholders (`sdxl_checkpoint.safetensors`,
`flux1-finetune-1_Q4_K_S.gguf`, …) so the repo fingerprints nobody's library.
Set them to the files actually staged before first run. The FLUX finetune slots
use engine keys `fluxft1` / `fluxft2` / `fluxft3` — frontend option values and
the backend `FLUX_ENGINES` dict must stay in sync if you rename them.

Sample service configs live in `deploy/` (`*.example`): copy, drop the suffix,
adjust paths, then `systemctl --user daemon-reload && systemctl --user enable --now <unit>`.
`flux-webui.env.example` is for an optional auth layer in front of the UI — the
plain webui.py has no auth.

## Models and where they go

Filenames are the constants in webui.py; folders are ComfyUI's standard ones:

| Folder | Files |
|---|---|
| `models/unet` | FLUX.1 GGUFs (dev, schnell, finetunes), MiniMax-H3 GGUF — via ComfyUI-GGUF |
| `models/diffusion_models` | flux-2-klein-4b, flux3-image-klein-bf16, qwen_image_2.1_bf16 |
| `models/text_encoders` | t5xxl_fp8, clip_l, qwen_3_4b, qwen3vl_8b_int8_convrot, qwen3vl_4b_bf16, qwen3vl_32b_minimax_h3_int8_convrot |
| `models/vae` | ae_flux, flux2-vae, qwen_image_2.1_vae_bf16, minimax_h3_image_vae_fp16, vae-ft-mse-840000-ema-pruned |
| `models/checkpoints` | your SDXL checkpoint, v1-5-pruned-emaonly, Pony V6 XL, RealVisXL V5, Illustrious-XL v2 |
| `models/loras` | any `.safetensors` LoRA — auto-listed and auto-filtered by architecture |
| `models/pulid` | pulid_flux_v0.9.1 |
| `models/controlnet` | flux-depth-controlnet, instantid-controlnet |
| `models/instantid` | ip-adapter.bin |

Custom nodes per feature: **ComfyUI-GGUF** (GGUF engines), **ComfyUI-PuLID-Flux**
(FLUX face-lock), **ComfyUI_InstantID** (SDXL face-lock), **comfyui_controlnet_aux**
(two-image pose lock; also needs the DepthAnythingV2 checkpoint). The qwen21 /
flux3 / minimax graphs use native nodes that only exist in recent ComfyUI
releases — update ComfyUI before debugging "unknown node" errors.
`RemoteQwenEncode` is optional and only used when `QWEN21_REMOTE_ENC` is set.

## API contract (for driving it programmatically)

`POST /api/generate` — multipart form fields:
`prompt` (required), `negative`, `seed` (blank = random), `steps`, `cfg`,
`denoise`, `shape` (`"WxH"`, e.g. `"1280x960"`), `engine`, `lora`,
`lora_strength`, `hires` (`"1"`), `keepface` (`"1"`), `body_change`,
`image` (file), `image2` (file).
Returns `{"image": "/img?filename=...&subfolder=...&type=output"}` or
`{"error": "..."}` with HTTP 500.

Other endpoints: `GET /api/queue`, `GET /api/recent?page=&per_page=`,
`POST /api/delete` (JSON `{"images": [filenames]}`), `POST /api/interrupt`,
`GET /img?type=output&filename=...`, `GET /shot?filename=...` (per-image shot
page: prompt + negative with copy buttons, reference head/body filenames linked
to `input/`, full settings), `GET /gallery`.

## Behavior you must not break

- **Engine switch evicts VRAM**: changing engines calls ComfyUI `/free`
  (`unload_models` + `free_memory`) — the big bf16 engines don't co-reside on one
  GPU. Same-engine runs keep the cache warm.
- **qwen21 is square-only**: `res = max(32, (min(w,h)//32)*32)`; CFG is clamped
  ≤ 4 (higher pins the VAE blue channel at 0 → yellow image).
- **Distilled engines** (flux2 / flux3 / schnell / minimax): CFG forced to 1,
  default steps 4 / 4 / 4 / 6; an explicit lower steps value is honored.
- **Two images** (`image` + `image2`): output size derives from image 2's aspect
  (~1 MP, multiples of 16); the `shape` field is ignored. Face from image 1
  (PuLID), pose from image 2 (depth ControlNet). FLUX engines only.
- **Custom shape**: the frontend rounds to ×8 with a 256 floor; the backend
  parses any `"WxH"`.
- **img2img** sources are scaled to ~1 MP (`ImageScaleToTotalPixels`) before encoding.
- **`run_graph` polls `/history` for up to ~30 min** — MiniMax can take ~9 min
  per render (32B CPU encoder). Do not shorten the timeout.
- **Saved PNGs embed the full graph** in metadata; `/shot` and gallery captions
  read it back. Keep a `SaveImage` node in every graph. `/shot` extracts the
  prompt from every engine's conditioning node (including the native qwen21 /
  minimax ones) and labels two-image references as head (node 10) / body
  (node 50); the first KSampler wins so hires passes don't overwrite base settings.
- **Path confinement**: image serving and deletion are confined to `OUTPUT_DIR`
  with basename checks. Keep it that way.

## House rules

- Single file. No new dependencies beyond aiohttp + pillow. No frameworks.
- Match the existing compact style: long lines, `$("id")` helpers, terse comments.
- No telemetry, analytics, or outbound internet calls — the only HTTP client
  requests go to `COMFY` (and the optional encoder service).
- Never commit personal prompts, hostnames, LAN IPs, or credentials. The shipped
  defaults are empty on purpose.
- After any change: `python -m py_compile webui.py`, serve the page, and run
  `node --check` on the extracted `<script>` blocks. Exercise frontend logic
  (stub the DOM or drive a browser) before calling it done.
