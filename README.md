# ComfyUI-ForDummies

ComfyUI for meat-brain monkeys. Let the clankers do the hard work so you can click the buttons.

This repo backs up **Image Studio** — a single-file, mobile-friendly web frontend
(`webui.py`) that sits in front of a normal ComfyUI install. One port (8191) serves
the page, the API, and the generated images.

## What you get

- One big Generate button. Engines: FLUX.1-dev / schnell / finetunes (GGUF),
  FLUX.2-Klein, FLUX.3-klein, Qwen-Image-2.1, MiniMax-H3-Image, SDXL family
  (Lustify, RealVisXL, Illustrious, Pony), Stable Diffusion 1.5.
- txt2img + img2img. Optional face-lock (PuLID on FLUX, InstantID on SDXL).
- Two-image mode: face from image 1 onto the body/pose of image 2 (depth ControlNet).
- LoRA dropdown auto-detected from each file's own tensor layout and filtered to
  the current engine's architecture.
- Preset sizes up to 4K, plus custom width×height with a live MP / aspect readout.
- Queue panel, kill button, recovery gallery (last 25 / all), per-image prompt +
  settings page (read from the PNG metadata), bulk delete.
- Ships with **empty default prompts** — bring your own.

## Quick start (ComfyUI already installed)

1. Copy `webui.py` into your ComfyUI root (next to `main.py`). It finds `models/`
   and `output/` relative to itself.
2. Install its two extra dependencies into the ComfyUI venv:
   `.venv/bin/pip install aiohttp pillow`
3. Make sure ComfyUI is running on `127.0.0.1:8188`, then:
   `.venv/bin/python webui.py`
4. Open `http://localhost:8191`.

Models: the constants at the top of `webui.py` name the exact files each engine
expects; AGENTS.md maps them to the ComfyUI subfolders they live in.

## Configuration

There is no runtime config file — everything is constants at the top of
`webui.py` (`COMFY`, `QWEN21_REMOTE_ENC`, model filenames, `DEFAULT_PROMPT`,
`DEFAULT_NEG`). Sample systemd user units and an env file live in `deploy/`:

- `deploy/comfyui.service.example` — the ComfyUI backend
- `deploy/comfyui-webui.service.example` — this frontend
- `deploy/flux-webui.env.example` — optional auth-layer credentials

## Safety

`webui.py` has **no authentication** and can delete images from `output/`.
Run it on localhost or a trusted LAN, or put it behind an auth proxy.

## For agents

[AGENTS.md](AGENTS.md) is the deployment / operations guide for AI agents
working with this frontend on an installed ComfyUI.
