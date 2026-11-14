#!/usr/bin/env python3
"""
Simple mobile-friendly web wrapper around ComfyUI.

Open http://localhost:8191 -> upload an image (optional) + type a prompt ->
Generate. Does img2img when an image is supplied, txt2img otherwise. Everything
(page + result image) is served through this one port so only 8191 needs to be open.
"""
import json, random, asyncio, os, struct, time
from aiohttp import web, ClientSession, ClientTimeout, FormData

COMFY = "http://127.0.0.1:8188"
LORA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "loras")
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")

def _list_loras():
    """Sorted .safetensors LoRA filenames available to ComfyUI (FLUX.1-dev)."""
    try:
        return sorted(f for f in os.listdir(LORA_DIR) if f.endswith(".safetensors"))
    except OSError:
        return []


def _lora_family(filename):
    """Peek a LoRA's safetensors header (no tensor data read) to guess which
    engine family it's built for: 'flux1' (FLUX.1-dev-architecture engines),
    'flux2' (FLUX.2-Klein), 'sdxl', or 'unknown' (shown for every engine).
    Filename is used only to disambiguate flux1 vs flux2 (both can be
    3072-hidden-dim and share the double_blocks/single_blocks key layout)."""
    try:
        path = os.path.join(LORA_DIR, filename)
        with open(path, "rb") as f:
            header_len = struct.unpack("<Q", f.read(8))[0]
            header = json.loads(f.read(header_len))
    except (OSError, struct.error, json.JSONDecodeError):
        return "unknown"
    keys = [k for k in header if k != "__metadata__"]
    if any("double_blocks" in k or "single_blocks" in k for k in keys):
        lname = filename.lower()
        return "flux2" if ("klein" in lname or "flux2" in lname or "f2k" in lname) else "flux1"
    if any("transformer_blocks" in k for k in keys):
        return "qwen21"
    if any(k.startswith("lora_te") or "down_blocks" in k or "up_blocks" in k for k in keys):
        return "sdxl"
    return "unknown"


def _lora_options_html():
    return "".join(
        '<option value="%s" data-family="%s">%s</option>' % (n, _lora_family(n), n)
        for n in _list_loras()
    )
CKPT = "sdxl_checkpoint.safetensors"
SD15_CKPT = "v1-5-pruned-emaonly.safetensors"
SD15_VAE = "vae-ft-mse-840000-ema-pruned.safetensors"
PONY_CKPT = "ponyDiffusionV6XL_v6StartWithThisOne.safetensors"
REALVIS_CKPT = "RealVisXL_V5.0_fp16.safetensors"
ILLUSTRIOUS_CKPT = "Illustrious-XL-v2.0.safetensors"
FLUX_UNET = "flux1-dev-Q5_K_S.gguf"
FLUX_SCHNELL_UNET = "flux1-schnell-Q5_K_S.gguf"
FLUXFT1_UNET = "flux1-finetune-1_Q4_K_S.gguf"
FLUXFT2_UNET = "flux1-finetune-2_Q6_K.gguf"
FLUXFT3_UNET = "flux1-finetune-3_fp8pruned.gguf"
FLUX_T5 = "t5xxl_fp8_e4m3fn.safetensors"
FLUX_CLIP_L = "clip_l.safetensors"
FLUX_VAE = "ae_flux.safetensors"
FLUX_GUIDANCE = 3.5

# FLUX.1-dev-architecture engines that only swap the GGUF UNET file (same CLIP/VAE/sampler).
FLUX_ENGINES = {
    "flux": FLUX_UNET,
    "schnell": FLUX_SCHNELL_UNET,
    "fluxft1": FLUXFT1_UNET,
    "fluxft2": FLUXFT2_UNET,
    "fluxft3": FLUXFT3_UNET,
}
# Plain-SDXL-checkpoint engines (CheckpointLoaderSimple), sharing the sdxl code path.
SDXL_ENGINES = {
    "sdxl": CKPT,
    "pony": PONY_CKPT,
    "realvis": REALVIS_CKPT,
    "illustrious": ILLUSTRIOUS_CKPT,
}
PULID_FLUX = "pulid_flux_v0.9.1.safetensors"
# FLUX.2-Klein (distilled, 4-step). No face-lock model staged -> txt2img + img2img only.
FLUX2_UNET = "flux-2-klein-4b.safetensors"
FLUX2_CLIP = "qwen_3_4b.safetensors"
FLUX2_VAE = "flux2-vae.safetensors"
# Qwen-Image-2.1 + FLUX.3-klein. Both bf16: a 32 GB card fits one engine at a time.
# The qwen text encoder loads on CPU (encoder runs once per generation; the bf16 UNet
# + int8 Qwen3-VL-8B don't both fit the card) and its VAE decodes tiled.
QWEN21_UNET = "qwen_image_2.1_bf16.safetensors"
QWEN21_CLIP = "qwen3vl_8b_int8_convrot.safetensors"
QWEN21_VAE = "qwen_image_2.1_vae_bf16.safetensors"
# Encoder offload: optional remote Qwen text-encoder service (RemoteQwenEncode node).
# Empty string = local CPU encode.
QWEN21_REMOTE_ENC = ""
FLUX3_UNET = "flux3-image-klein-bf16.safetensors"
FLUX3_CLIP = "qwen3vl_4b_bf16.safetensors"
FLUX3_VAE = "flux2-vae.safetensors"
# MiniMax-H3-Image (distilled 6-step, CFG 1). DiT Q4_K_M GGUF on the GPU, the 32B
# Qwen3-VL encoder int8_convrot on CPU (26 GB; won't co-reside with the DiT), image VAE
# fp16. Native VAEDecode is fine at 2048. No LoRAs staged.
MINIMAX_UNET = "MiniMax-H3-Image-Pruned-Q4_K_M.gguf"
MINIMAX_CLIP = "qwen3vl_32b_minimax_h3_int8_convrot.safetensors"
MINIMAX_VAE = "minimax_h3_image_vae_fp16.safetensors"
DEFAULT_NEG = "lowres, blurry, jpeg artifacts, watermark, text, deformed, bad anatomy, extra limbs"
DEFAULT_PROMPT = ""

PAGE = """<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Image Studio</title>
<style>
 :root{color-scheme:dark}
 *{box-sizing:border-box}
 body{font-family:-apple-system,system-ui,Segoe UI,Roboto,sans-serif;margin:0;background:#15171c;color:#e8eaed}
 .wrap{max-width:680px;margin:0 auto;padding:18px}
 h1{font-size:20px;font-weight:600;margin:6px 0 16px}
 label{display:block;font-size:13px;color:#9aa0a6;margin:14px 0 6px}
 textarea,input[type=text],input[type=number],select{width:100%;padding:11px;border-radius:10px;border:1px solid #2c2f36;background:#1e2127;color:#e8eaed;font-size:15px}
 textarea{min-height:84px;resize:vertical}
 .row{display:flex;gap:12px}.row>div{flex:1}
 .drop{border:2px dashed #3a3f48;border-radius:12px;padding:18px;text-align:center;color:#9aa0a6;cursor:pointer}
 .drop.has{border-color:#5b8cff;color:#cfe0ff}
 #preview,#preview2{max-width:100%;border-radius:10px;margin-top:10px;display:none}
 button{width:100%;margin-top:18px;padding:14px;border:0;border-radius:12px;background:#5b8cff;color:#fff;font-size:16px;font-weight:600}
 button:disabled{background:#3a3f48;color:#7a7f88}
 .deleteControl{display:block;width:100%;padding:14px;border-radius:12px;background:#b3261e;color:#fff;text-align:center;text-decoration:none;font-size:16px;font-weight:600;cursor:pointer;user-select:none}
 #kill{display:none;background:#b3261e;margin-top:10px}
 .removeImage{display:none;margin-top:10px;padding:9px;background:#2c2f36;color:#c8ccd4;font-size:13px;font-weight:500}
 .slider{display:none}
 .out{margin-top:18px;text-align:center}
 #result{max-width:100%;border-radius:12px}
 .muted{font-size:12px;color:#727784;margin-top:6px}
 .spin{display:none;margin-top:18px;text-align:center;color:#9aa0a6}
 details{margin-top:8px}summary{cursor:pointer;color:#9aa0a6;font-size:13px}
</style></head><body><div class="wrap">
 <h1>🎨 Image Studio</h1>

 <label>Engine</label>
 <select id="engine">
   <option value="flux3" selected>FLUX.3-klein — newest FLUX, distilled (4 steps)</option>
   <option value="qwen21">Qwen-Image-2.1 — photoreal, no LoRA unless selected (slow)</option>
   <option value="minimax">MiniMax-H3-Image — 6-step, perfect text rendering (slow encode)</option>
   <option value="sdxl">SDXL — fast (~20s), face-lock</option>
   <option value="realvis">RealVisXL V5 — SDXL photorealism, face-lock</option>
   <option value="illustrious">Illustrious-XL — SDXL anime/illustration, face-lock</option>
   <option value="sd15">Stable Diffusion 1.5 — lightweight</option>
   <option value="pony">Pony Diffusion V6 XL — stylized</option>
 </select>
 <div class="muted" id="engineNote">FLUX = highest quality (slower). Face-lock works in both (FLUX uses PuLID, SDXL uses InstantID).</div>

 <label>LoRA (FLUX engines)</label>
 <select id="lora">
   <option value="None">None</option>
   <!--LORA_OPTIONS-->
 </select>
 <label>LoRA strength <span id="loraStrengthVal">0.90</span></label>
 <input type="range" id="loraStrength" min="0" max="1.5" step="0.05" value="0.9"
        oninput="document.getElementById('loraStrengthVal').textContent=(+this.value).toFixed(2)">
 <div class="muted" id="loraNote">Applies to the selected FLUX engine (SDXL ignores it). The dropdown is auto-filtered to LoRAs whose base architecture matches the current engine (detected from the file's own tensor layout, not its name) — mismatched LoRAs silently no-op instead of erroring, so keeping them out of the list avoids that trap. Drop more <code>.safetensors</code> into <code>ComfyUI/models/loras</code> and refresh to pick them.</div>

 <label>Size / Aspect</label>
 <select id="shape">
   <option value="1344x768">Landscape — 1344×768</option>
   <option value="1024x1024">Square — 1024×1024</option>
   <option value="768x1344">Portrait — 768×1344</option>
   <option value="1536x640">Wide — 1536×640</option>
   <option value="1920x1080">1080p — 1920×1080 (landscape)</option>
   <option value="1080x1920">1080p portrait — 1080×1920</option>
   <option value="3840x2160">4K UHD — 3840×2160 (landscape)</option>
   <option value="2160x3840">4K portrait — 2160×3840</option>
   <option value="4096x4096">4K square — 4096×4096</option>
   <option value="custom">Custom…</option>
 </select>
 <div id="customSize" style="display:none;margin-top:10px">
   <div style="display:flex;gap:8px;align-items:center">
     <input type="number" id="cw" min="256" step="8" value="1280" style="flex:1;min-width:0">
     <span class="muted">×</span>
     <input type="number" id="ch" min="256" step="8" value="960" style="flex:1;min-width:0">
     <button id="swapSize" title="Swap width / height" style="width:auto;margin:0;padding:11px 14px">⇄</button>
   </div>
   <div class="muted" id="sizeReadout" style="margin-top:6px"></div>
 </div>
 <div class="muted">1080p/4K are tall: qwen21 will OOM beyond ~1024×1024 on a 32 GB card, and 4K needs the tiled decoder. Use FLUX.1-dev for 1080p. (Two-image auto-sizes from the body image.)</div>

 <label>Image 1 — reference / face source (optional)</label>
 <div class="drop" id="drop">Tap to choose a photo / drawing</div>
 <input type="file" id="file" accept="image/*" style="display:none">
 <img id="preview">
 <button type="button" class="removeImage" id="removeImage">Remove image 1</button>

 <div class="slider" id="strengthBox">
   <label style="display:flex;align-items:center;gap:10px;margin-bottom:6px;color:#e8eaed">
     <input type="checkbox" id="keepface" checked style="width:auto;transform:scale(1.3)">
     🧑 Keep face from photo (InstantID)
   </label>
   <label><span id="strLabel">Face match strength: </span><span id="dv">0.85</span>
     <span class="muted" id="strHint">(higher = closer to the original face)</span></label>
   <input type="range" id="denoise" min="0.2" max="0.95" step="0.05" value="0.85" style="width:100%">
 </div>

 <label>Image 2 — body / pose (optional)</label>
 <div class="drop" id="drop2">Tap to choose the BODY image</div>
 <input type="file" id="file2" accept="image/*" style="display:none">
 <img id="preview2">
 <button type="button" class="removeImage" id="removeImage2">Remove image 2</button>
 <div class="slider" id="bodyBox">
   <label>Pose lock strength: <span id="bcv">0.60</span>
     <span class="muted">(higher = stick to Image-2 body/pose; your prompt still controls outfit/scene)</span></label>
   <input type="range" id="bodychange" min="0.2" max="1.0" step="0.05" value="0.6" style="width:100%">
 </div>
 <div class="muted" id="faceonbodyNote" style="display:none;color:#cfe0ff">🔀 Two images: face from Image 1 (PuLID) + body/pose from Image 2 (depth ControlNet). Your prompt drives the outfit/scene. FLUX, ~2 min.</div>

 <label>Prompt</label>
 <textarea id="prompt">""" + DEFAULT_PROMPT + """</textarea>
 <div class="muted" id="promptCount"></div>

 <details><summary>Advanced</summary>
   <label>Negative prompt</label>
   <textarea id="neg">""" + DEFAULT_NEG + """</textarea>
   <label>Seed <span class="muted">(blank = random)</span></label>
   <input type="text" id="seed" placeholder="random" inputmode="numeric">
   <label>Steps</label><input type="text" id="steps" value="28">
   <div class="muted">Size moved up top (Size / Aspect). Steps auto-set per engine (FLUX.2 = 4).</div>
   <label>Prompt strength (CFG): <span id="cfgv">6.5</span>
     <span class="muted">(low = looser/softer, high = follows prompt harder)</span></label>
   <input type="range" id="cfg" min="2" max="12" step="0.5" value="6.5" style="width:100%">
 </details>

 <label style="display:flex;align-items:center;gap:10px;margin-top:18px;font-size:15px;color:#e8eaed">
   <input type="checkbox" id="hires" style="width:auto;transform:scale(1.3)">
   ✨ Higher quality (hi-res pass · ~2× slower)
 </label>

 <button id="go">Generate</button>
 <button id="kill" type="button">Kill current job</button>
 <button id="clear" style="background:#2c2f36;margin-top:10px">↺ Clear / New</button>
 <div class="spin" id="spin">⏳ Generating… (~15–25s)</div>
 <div class="out"><img id="result" style="display:none"></div>
 <div style="margin-top:14px;padding:12px;border:1px solid #2c2f36;border-radius:10px">
   <div style="display:flex;justify-content:space-between;align-items:center">
     <strong style="font-size:14px">Queue</strong>
     <button id="queueRefresh" type="button" style="width:auto;margin:0;padding:6px 12px;font-size:13px;background:#2c2f36">🔄 Refresh</button>
   </div>
   <div id="queueStatus" class="muted" style="margin-top:8px">Tap refresh to check.</div>
 </div>
 <button id="recentBtn" style="background:#2c2f36;margin-top:10px">🖼 Last 25 images</button>
 <div class="muted">Recovers anything that generated but didn't display (e.g. the tab slept mid-render). Tap a thumbnail to open it full-size.</div>
 <div id="gallery" style="margin-top:8px"></div>
 <a id="moreImages" href="gallery?page=2" style="display:none;margin-top:12px;padding:12px;border-radius:10px;background:#2c2f36;color:#fff;text-align:center;text-decoration:none;font-weight:600">Show more</a>
 <a href="#" role="button" tabindex="0" class="deleteControl" id="deleteImages" style="display:none;margin-top:12px">Delete selected</a>
</div>
<script>
const $=id=>document.getElementById(id);
let fileData=null;
let activeJobs=0;
$("drop").onclick=()=>$("file").click();
$("file").onchange=e=>{const f=e.target.files[0];if(!f)return;fileData=f;
  $("drop").classList.add("has");$("drop").textContent=f.name;
  const r=new FileReader();r.onload=ev=>{$("preview").src=ev.target.result;$("preview").style.display="block";$("removeImage").style.display="block";$("strengthBox").style.display="block";};r.readAsDataURL(f);};
$("removeImage").onclick=()=>{fileData=null;$("file").value="";$("drop").classList.remove("has");$("drop").textContent="Tap to choose a photo / drawing";$("preview").style.display="none";$("preview").src="";$("removeImage").style.display="none";$("strengthBox").style.display="none";};
let fileData2=null;
$("drop2").onclick=()=>$("file2").click();
$("file2").onchange=e=>{const f=e.target.files[0];if(!f)return;fileData2=f;
  $("drop2").classList.add("has");$("drop2").textContent=f.name;
  const r=new FileReader();r.onload=ev=>{$("preview2").src=ev.target.result;$("preview2").style.display="block";$("removeImage2").style.display="block";$("bodyBox").style.display="block";$("faceonbodyNote").style.display="block";};r.readAsDataURL(f);};
$("removeImage2").onclick=()=>{fileData2=null;$("file2").value="";$("drop2").classList.remove("has");$("drop2").textContent="Tap to choose the BODY image";$("preview2").style.display="none";$("preview2").src="";$("removeImage2").style.display="none";$("bodyBox").style.display="none";$("faceonbodyNote").style.display="none";};
$("bodychange").oninput=e=>$("bcv").textContent=parseFloat(e.target.value).toFixed(2);
$("denoise").oninput=e=>$("dv").textContent=parseFloat(e.target.value).toFixed(2);
$("cfg").oninput=e=>$("cfgv").textContent=parseFloat(e.target.value).toFixed(1);
function updatePromptCount(){
  const text=$("prompt").value;
  const words=text.trim()?text.trim().split(/\\s+/).length:0;
  const estTokens=Math.round(words*1.3);
  const el=$("promptCount");
  el.textContent=text.length+" chars / ~"+estTokens+" tokens (CLIP-L chunks in 77-token windows past this point — SDXL/Pony/RealVisXL/Illustrious/SD1.5 and FLUX's pooled CLIP; T5/Qwen encoders on FLUX-family engines have no practical limit)";
  el.style.color=estTokens>77?"#ff9b9b":"#727784";
}
$("prompt").addEventListener("input",updatePromptCount);
updatePromptCount();
const FLUX1_ENGINES=["flux","schnell","fluxft1","fluxft2","fluxft3"];
function filterLoraOptions(engine){
  const family=engine==="flux2"?"flux2":(engine==="qwen21"?"qwen21":(FLUX1_ENGINES.includes(engine)?"flux1":null));
  const sel=$("lora"),prevValue=sel.value;
  let sawSelected=false;
  Array.from(sel.options).forEach(opt=>{
    if(opt.value==="None"){opt.hidden=false;return;}
    const fam=opt.dataset.family;
    const show=family===null?false:(fam===family||fam==="unknown");
    opt.hidden=!show;
    if(show&&opt.value===prevValue)sawSelected=true;
  });
  if(!sawSelected)sel.value="None";
  sel.disabled=family===null;
  $("loraNote").style.opacity=family===null?"0.5":"1";
}
$("engine").onchange=()=>{
  const e=$("engine").value;
  filterLoraOptions(e);
  if(e!=="qwen21") $("cfg").max="12";
  if(e==="flux2"){
    $("steps").value="4";$("cfg").value="1";$("cfgv").textContent="1.0";
    $("engineNote").textContent="FLUX.2-Klein = distilled & very fast (4 steps, CFG 1, no negative). txt2img + img2img; no face-lock.";
  }else if(e==="flux3"){
    $("steps").value="4";$("cfg").value="1";$("cfgv").textContent="1.0";
    $("engineNote").textContent="FLUX.3-klein = newest FLUX, distilled (4 steps, CFG 1, no negative). txt2img + img2img; no LoRAs staged; no face-lock.";
  }else if(e==="qwen21"){
    $("steps").value="28";$("cfg").max="4";$("cfg").value="4";$("cfgv").textContent="4.0";
    $("engineNote").textContent="Qwen-Image-2.1 = photoreal, slow (28 steps, CFG 4, negative used). CFG above 4 washes the picture yellow. CPU text encoder + tiled decode; square output only; no face-lock.";
  }else if(e==="minimax"){
    $("steps").value="6";$("cfg").max="4";$("cfg").value="1";$("cfgv").textContent="1.0";
    $("engineNote").textContent="MiniMax-H3-Image = 6-step distilled, CFG 1, renders product text perfectly. 32B CPU encoder: ~9 min/render. No LoRAs or face-lock staged.";
  }else if(e==="schnell"){
    $("steps").value="4";$("cfg").value="1";$("cfgv").textContent="1.0";
    $("engineNote").textContent="FLUX.1-schnell = distilled & very fast (4 steps, CFG 1). Face-lock (PuLID) available.";
  }else if(e==="flux"){
    $("steps").value="28";
    $("engineNote").textContent="FLUX = highest quality (slower). Face-lock works in both (FLUX uses PuLID, SDXL uses InstantID).";
  }else if(e==="fluxft1"){
    $("steps").value="28";
    $("engineNote").textContent="FLUX.1 finetune 1 (GGUF). Face-lock via PuLID.";
  }else if(e==="fluxft2"){
    $("steps").value="28";
    $("engineNote").textContent="FLUX.1 finetune 2 (GGUF). Face-lock via PuLID.";
  }else if(e==="fluxft3"){
    $("steps").value="28";
    $("engineNote").textContent="FLUX.1 finetune 3 (GGUF). Face-lock via PuLID.";
  }else if(e==="sdxl"){
    $("steps").value="28";
    $("engineNote").textContent="SDXL = fast (~20s). Face-lock via InstantID.";
  }else if(e==="realvis"){
    $("steps").value="28";
    $("engineNote").textContent="RealVisXL V5 = SDXL photorealism. Face-lock via InstantID.";
  }else if(e==="illustrious"){
    $("steps").value="28";
    $("engineNote").textContent="Illustrious-XL = SDXL anime/illustration finetune. Face-lock via InstantID.";
  }else if(e==="sd15"){
    $("steps").value="28";
    $("engineNote").textContent="Stable Diffusion 1.5 = lightweight, broad checkpoint/LoRA ecosystem. Face-lock unavailable.";
  }else{
    $("steps").value="28";
    $("engineNote").textContent="Pony Diffusion V6 XL = stylized SDXL finetune. Use score_9, score_8_up, score_7_up quality tags; face-lock via InstantID.";
  }
};
filterLoraOptions($("engine").value);
$("engine").dispatchEvent(new Event("change"));
$("keepface").onchange=()=>{
  const kf=$("keepface").checked;
  $("strLabel").textContent=kf?"Face match strength: ":"Transformation strength: ";
  $("strHint").textContent=kf?"(higher = closer to the original face)":"(low = keep original, high = more change)";
  const v=kf?0.85:0.6;$("denoise").value=v;$("dv").textContent=v.toFixed(2);
};
function resetInputs(clearPrompt){
  fileData=null;$("file").value="";
  $("drop").classList.remove("has");$("drop").textContent="Tap to choose a photo / drawing";
  $("preview").style.display="none";$("preview").src="";
  $("removeImage").style.display="none";
  $("strengthBox").style.display="none";
  fileData2=null;$("file2").value="";
  $("drop2").classList.remove("has");$("drop2").textContent="Tap to choose the BODY image";
  $("preview2").style.display="none";$("preview2").src="";
  $("removeImage2").style.display="none";
  $("bodyBox").style.display="none";$("faceonbodyNote").style.display="none";
  if(clearPrompt){$('prompt').value='""" + DEFAULT_PROMPT + """';}
}
$("clear").onclick=()=>{resetInputs(true);$("result").style.display="none";$("result").src="";};
function updateActiveUi(){
  $("kill").style.display=activeJobs>0?"block":"none";
  $("spin").style.display=activeJobs>0?"block":"none";
  $("spin").textContent=activeJobs>1?"⏳ "+activeJobs+" jobs in progress… (queued on the GPU, will finish in order)":$("spin").dataset.lastMsg||"⏳ Generating…";
}
function customDims(){
  const r=v=>Math.max(256,Math.round((parseInt(v)||1024)/8)*8);
  return [r($("cw").value),r($("ch").value)];
}
function updateSizeReadout(){
  let [w,h]=customDims();
  const q=$("engine").value==="qwen21";
  if(q)w=h=Math.max(32,Math.floor(Math.min(w,h)/32)*32);
  const g=(a,b)=>b?g(b,a%b):a,d=g(w,h);
  const ar=(w/d<=32&&h/d<=32)?(w/d)+":"+(h/d):(w/h).toFixed(2)+":1";
  let s=w+"×"+h+" · "+(w*h/1e6).toFixed(2)+" MP · "+ar;
  if(q)s+=" — qwen21 renders square"+(w*h>1048576?"; OOM risk past ~1 MP, use FLUX.1-dev":"");
  $("sizeReadout").textContent=s;
}
function syncCustomSize(){
  const on=$("shape").value==="custom";
  $("customSize").style.display=on?"block":"none";
  if(on)updateSizeReadout();
}
$("shape").onchange=syncCustomSize;
$("cw").oninput=$("ch").oninput=updateSizeReadout;
$("swapSize").onclick=()=>{const w=$("cw").value;$("cw").value=$("ch").value;$("ch").value=w;updateSizeReadout();};
$("engine").addEventListener("change",updateSizeReadout);
syncCustomSize();
$("go").onclick=async()=>{
  const p=$("prompt").value.trim();
  if(!p){alert("Please enter a prompt");return;}
  activeJobs++;$("kill").style.display="block";$("kill").disabled=false;$("kill").textContent="Kill running job";$("result").style.display="none";
  const fd=new FormData();
  fd.append("prompt",p);fd.append("negative",$("neg").value);
  fd.append("seed",$("seed").value);
  fd.append("denoise",$("denoise").value);fd.append("steps",$("steps").value);
  fd.append("shape",$("shape").value==="custom"?customDims().join("x"):$("shape").value);fd.append("cfg",$("cfg").value);
  fd.append("hires",$("hires").checked?"1":"");
  fd.append("keepface",$("keepface").checked?"1":"");
  fd.append("engine",$("engine").value);
  fd.append("lora",$("lora").value);
  fd.append("lora_strength",$("loraStrength").value);
  fd.append("body_change",$("bodychange").value);
  if(fileData)fd.append("image",fileData);
  if(fileData2)fd.append("image2",fileData2);
  const twoImg=fileData&&fileData2;
  const eng=$("engine").value;const flux=eng==="flux";
  const msg=twoImg?"⏳ Putting face on body (FLUX+PuLID)… ~90s":(eng==="flux2"?"⏳ Generating with FLUX.2-Klein… (~10–20s)":(eng==="flux3"?"⏳ Generating with FLUX.3-klein… (~30–60s)":(eng==="qwen21"?"⏳ Generating with Qwen-Image-2.1… (~2–5 min, please wait)":(flux?"⏳ Generating with FLUX… (~1–2 min, please wait)":(eng==="sd15"?"⏳ Generating with Stable Diffusion 1.5… (~20–40s)":(eng==="pony"?"⏳ Generating with Pony Diffusion… (~30–60s)":($("hires").checked?"⏳ Generating (hi-res)… ~30–50s":"⏳ Generating… (~15–25s)")))))));
  $("spin").dataset.lastMsg=msg;
  updateActiveUi();
  try{
    const r=await fetch("/api/generate",{method:"POST",body:fd});
    const j=await r.json();
    if(j.image){$("result").src=j.image+"&t="+Date.now();$("result").style.display="inline";}
      // image + prompt deliberately kept so you can tweak/re-run without re-uploading;
      // each run is already independent server-side (fresh seed + unique upload name).
    else alert("Error: "+(j.error||"unknown"));
  }catch(e){alert("Request failed: "+e);}
  activeJobs=Math.max(0,activeJobs-1);updateActiveUi();
};
$("kill").onclick=async()=>{
  if(activeJobs<=0)return;
  $("kill").disabled=true;$("kill").textContent="Killing running job…";
  try{
    const r=await fetch("/api/interrupt",{method:"POST"});
    const j=await r.json();
    if(!r.ok)throw new Error(j.error||"interrupt failed");
  }catch(e){alert("Could not kill the running job: "+e);}
  $("kill").disabled=false;$("kill").textContent="Kill running job";
};
// Base path the page is served under, so URLs work both behind a reverse proxy
// and when hitting :8191 directly. (ends with "/")
function apiBase(){return location.pathname.endsWith("/")?location.pathname:location.pathname.replace(/[^/]*$/,"");}
$("queueRefresh").onclick=async()=>{
  const el=$("queueStatus");el.textContent="checking…";
  try{
    const r=await fetch(apiBase()+"api/queue");
    const j=await r.json();
    if(j.error){el.textContent="Error: "+j.error;return;}
    const fmt=list=>list.map(x=>x.engine).join(", ")||"none";
    el.textContent="Running ("+j.running_count+"): "+fmt(j.running)+"  |  Pending ("+j.pending_count+"): "+fmt(j.pending);
  }catch(e){el.textContent="Failed to check queue: "+e;}
};
function fmtWhen(mtime){
  const d=new Date(mtime*1000);
  return d.toLocaleString(undefined,{month:"short",day:"numeric",hour:"numeric",minute:"2-digit"});
}
$("recentBtn").onclick=async()=>{
  const g=$("gallery");g.textContent="loading…";
  try{
    const r=await fetch(apiBase()+"api/recent?page=1&per_page=25");
    const j=await r.json();
    g.innerHTML="";
    const items=j.items&&j.items.length?j.items:(j.images||[]).map(fn=>({filename:fn,mtime:null,engine:null}));
    if(!items.length){g.textContent="No images yet.";return;}
    items.forEach(it=>{
      const fn=it.filename;
      const url=apiBase()+"img?type=output&filename="+encodeURIComponent(fn);
      const box=document.createElement("div");
      box.style.cssText="display:inline-block;width:31%;margin:1%;position:relative;vertical-align:top";
      const caption=(it.mtime?fmtWhen(it.mtime):"")+(it.engine?"  ·  "+it.engine:"")+(it.model?"  ·  "+it.model:"")+(it.lora?"  ·  LoRA: "+it.lora:"")+"  ·  "+fn;
      const im=document.createElement("img");
      im.src=url;im.title=caption;im.alt=caption;
      im.style.cssText="width:100%;border-radius:6px;cursor:pointer;display:block";
      im.onclick=()=>window.open(apiBase()+"shot?filename="+encodeURIComponent(fn),"_blank","noopener");;
      const cb=document.createElement("input");cb.type="checkbox";cb.value=fn;cb.className="imageSelect";
      cb.style.cssText="position:absolute;z-index:3;top:7px;left:7px;width:22px;height:22px;accent-color:#b3261e;cursor:pointer;background:#15171ccc";
      cb.onclick=e=>e.stopPropagation();cb.onchange=updateDeleteButton;
      box.append(im,cb);g.appendChild(box);
    });
    const more=$("moreImages");
    more.href=apiBase()+"gallery?page=2";
    more.style.display=j.has_more?"block":"none";
  }catch(e){g.textContent="Failed to load recent: "+e;}
};
function updateDeleteButton(){
  const n=document.querySelectorAll(".imageSelect:checked").length;
  const b=$("deleteImages");b.style.display=n?"block":"none";b.textContent="Delete selected ("+n+")";
}
$("deleteImages").onclick=async(event)=>{
  event.preventDefault();
  const btn=$("deleteImages");
  const images=Array.from(document.querySelectorAll(".imageSelect:checked"),x=>x.value);
  if(!images.length||!confirm("Permanently delete "+images.length+" selected image"+(images.length===1?"":"s")+"? This cannot be undone."))return;
  btn.disabled=true;
  try{
    const r=await fetch(apiBase()+"api/delete",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({images})});
    const j=await r.json();
    if(!r.ok){alert("Delete failed: "+(j.error||r.status));return;}
    alert("Deleted "+j.deleted.length+" image"+(j.deleted.length===1?"":"s")+".");
    $("recentBtn").click();updateDeleteButton();
  }catch(e){alert("Delete request failed: "+e);}
  finally{btn.disabled=false;}
};
</script></body></html>"""

GALLERY_PAGE = r"""<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Image Studio Gallery</title>
<style>
 :root{color-scheme:dark}*{box-sizing:border-box}
 body{font-family:-apple-system,system-ui,Segoe UI,Roboto,sans-serif;margin:0;background:#15171c;color:#e8eaed}
 .wrap{max-width:900px;margin:0 auto;padding:18px}h1{font-size:20px;margin:6px 0 4px}.muted{font-size:13px;color:#9aa0a6;margin-bottom:14px}
 .grid{display:grid;grid-template-columns:repeat(3,1fr);gap:8px}.tile{position:relative}.tile img{width:100%;aspect-ratio:1;object-fit:cover;border-radius:8px;cursor:pointer;display:block}.tile input{position:absolute;z-index:3;top:7px;left:7px;width:24px;height:24px;accent-color:#b3261e;cursor:pointer;background:#15171ccc}
 .tilecap{font-size:11px;line-height:1.35;color:#9aa0a6;padding:3px 2px 0;word-break:break-all}.tilecap b{color:#c8ccd2;font-weight:600}
 .nav{display:flex;gap:10px;margin:18px 0}.nav a{flex:1;padding:12px;border-radius:10px;background:#2c2f36;color:#fff;text-align:center;text-decoration:none;font-weight:600}.nav a.primary{background:#5b8cff}.nav a.hidden{display:none}
 .deleteControl{display:none;width:100%;margin-top:18px;padding:14px;border-radius:12px;background:#b3261e;color:#fff;font-size:16px;font-weight:600;text-align:center;text-decoration:none;cursor:pointer;user-select:none}
 #full{display:none;width:100%;max-height:85vh;object-fit:contain;border-radius:12px;margin-bottom:14px}
</style></head><body><div class="wrap">
 <h1>🖼 Image archive</h1><div class="muted" id="status">Loading…</div>
 <img id="full"><div class="grid" id="gallery"></div>
 <a href="#" role="button" tabindex="0" class="deleteControl" id="deleteImages">Delete selected</a>
 <div class="nav"><a id="prev">Previous 25</a><a class="primary" id="next">Show more</a></div>
 <div class="nav"><a class="primary" href="gallery?all=1">Show ALL</a></div>
 <div class="nav"><a href="./">Back to Image Generator</a></div>
</div><script>
const params=new URLSearchParams(location.search);const showAll=params.get("all")==="1";const page=Math.max(2,parseInt(params.get("page")||"2",10)||2);
const base=location.pathname.replace(/gallery\/?$/,""),gallery=document.getElementById("gallery");
function fmtWhen(mtime){
  const d=new Date(mtime*1000);
  return d.toLocaleString(undefined,{month:"short",day:"numeric",hour:"numeric",minute:"2-digit"});
}
async function loadGallery(){
 const items=[];let total=0,current=showAll?1:page,hasMore=true;
 while(hasMore){
   const j=await fetch(base+"api/recent?page="+current+"&per_page=25").then(r=>r.json());
   const pageItems=j.items&&j.items.length?j.items:(j.images||[]).map(fn=>({filename:fn,mtime:null,engine:null}));
   items.push(...pageItems);total=j.total||0;hasMore=j.has_more;if(showAll)current++;
   if(!showAll)break;
 }
 document.getElementById("status").textContent=showAll?"All images ("+items.length+")":(items.length?"Images "+((page-1)*25+1)+"–"+((page-1)*25+items.length)+" of "+total:"No more images.");
 items.forEach(it=>{const fn=it.filename,url=base+"img?type=output&filename="+encodeURIComponent(fn),tile=document.createElement("div"),im=document.createElement("img"),cb=document.createElement("input");tile.className="tile";const caption=(it.mtime?fmtWhen(it.mtime):"")+(it.engine?"  ·  "+it.engine:"")+(it.model?"  ·  "+it.model:"")+(it.lora?"  ·  LoRA: "+it.lora:"")+"  ·  "+fn;im.src=url;im.title=caption;im.alt=caption;im.onclick=e=>{e.preventDefault();e.stopPropagation();window.open(base+"shot?filename="+encodeURIComponent(fn),"_blank","noopener");};cb.type="checkbox";cb.value=fn;cb.className="imageSelect";cb.onclick=e=>e.stopPropagation();cb.onchange=updateDeleteButton;const cap=document.createElement("div");cap.className="tilecap";const bits=[];if(it.engine&&it.engine!=="?")bits.push("<b>"+it.engine+"</b>");if(it.model)bits.push(it.model.replace(/\.(safetensors|gguf|ckpt)$/i,""));if(it.lora)bits.push("LoRA: "+it.lora.replace(/\.(safetensors|gguf|ckpt)$/i,""));cap.innerHTML=bits.length?bits.join(" · "):"&nbsp;";tile.append(im,cb,cap);gallery.appendChild(tile);});
 const prev=document.getElementById("prev"),next=document.getElementById("next");
 if(showAll){prev.classList.add("hidden");next.classList.add("hidden");}
 else {prev.href=page===2?"./":"gallery?page="+(page-1);next.href="gallery?page="+(page+1);if(!hasMore)next.classList.add("hidden");}
}
loadGallery().catch(e=>document.getElementById("status").textContent="Failed to load images: "+e);
function updateDeleteButton(){const n=document.querySelectorAll(".imageSelect:checked").length,b=document.getElementById("deleteImages");b.style.display=n?"block":"none";b.textContent="Delete selected ("+n+")";}
document.getElementById("deleteImages").onclick=async(event)=>{
  event.preventDefault();
  const btn=document.getElementById("deleteImages");
  const images=Array.from(document.querySelectorAll(".imageSelect:checked"),x=>x.value);
  if(!images.length||!confirm("Permanently delete "+images.length+" selected image"+(images.length===1?"":"s")+"? This cannot be undone."))return;
  btn.disabled=true;
  try{
    const r=await fetch(base+"api/delete",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({images})});
    const j=await r.json();
    if(!r.ok){alert("Delete failed: "+(j.error||r.status));btn.disabled=false;return;}
    alert("Deleted "+j.deleted.length+" image"+(j.deleted.length===1?"":"s")+".");
    location.reload();
  }catch(e){alert("Delete request failed: "+e);btn.disabled=false;}
};
document.getElementById("deleteImages").onkeydown=event=>{
  if(event.key==="Enter"||event.key===" "){event.preventDefault();event.currentTarget.click();}
};
</script></body></html>"""


def _ksampler(seed, steps, cfg, denoise, latent, model=["4",0], pos=["6",0], neg=["7",0]):
    return {"class_type":"KSampler","inputs":{"seed":seed,"steps":steps,"cfg":cfg,"sampler_name":"dpmpp_2m","scheduler":"karras","denoise":denoise,"model":model,"positive":pos,"negative":neg,"latent_image":latent}}


def _finish(g, base_latent, seed, cfg, hires, model=["4",0], pos=["6",0], neg=["7",0], vae=["4",2]):
    """Append optional hi-res pass, then VAEDecode + SaveImage. Returns g."""
    last = base_latent
    if hires:
        g["13"] = {"class_type":"LatentUpscaleBy","inputs":{"samples":base_latent,"upscale_method":"bislerp","scale_by":1.5}}
        g["14"] = _ksampler(seed + 1, 20, cfg, 0.45, ["13",0], model, pos, neg)   # refine pass adds detail
        last = ["14",0]
    g["8"] = {"class_type":"VAEDecode","inputs":{"samples":last,"vae":vae}}
    g["9"] = {"class_type":"SaveImage","inputs":{"filename_prefix":"web","images":["8",0]}}
    return g


def _sd_condition(g, p, n, clip_skip=0):
    clip = ["4", 1]
    if clip_skip:
        g["24"] = {"class_type":"CLIPSetLastLayer","inputs":{"clip":["4",1],"stop_at_clip_layer":clip_skip}}
        clip = ["24", 0]
    g["6"] = {"class_type":"CLIPTextEncode","inputs":{"text":p,"clip":clip}}
    g["7"] = {"class_type":"CLIPTextEncode","inputs":{"text":n,"clip":clip}}
    return g


def txt2img_graph(p, n, seed, steps, w, h, cfg, hires, ckpt=CKPT, clip_skip=0):
    g = {
        "4":{"class_type":"CheckpointLoaderSimple","inputs":{"ckpt_name":ckpt}},
        "5":{"class_type":"EmptyLatentImage","inputs":{"width":w,"height":h,"batch_size":1}},
        "3":_ksampler(seed, steps, cfg, 1.0, ["5",0]),
    }
    if ckpt == SD15_CKPT:
        g["25"] = {"class_type":"VAELoader","inputs":{"vae_name":SD15_VAE}}
    _sd_condition(g, p, n, clip_skip)
    vae = ["25",0] if ckpt == SD15_CKPT else ["4",2]
    return _finish(g, ["3",0], seed, cfg, hires, vae=vae)


def img2img_graph(p, n, seed, steps, denoise, image_name, cfg, hires, ckpt=CKPT, clip_skip=0):
    g = {
        "10":{"class_type":"LoadImage","inputs":{"image":image_name}},
        "11":{"class_type":"ImageScaleToTotalPixels","inputs":{"image":["10",0],"upscale_method":"lanczos","megapixels":1.0,"resolution_steps":64}},
        "4":{"class_type":"CheckpointLoaderSimple","inputs":{"ckpt_name":ckpt}},
        "12":{"class_type":"VAEEncode","inputs":{"pixels":["11",0],"vae":["4",2]}},
        "3":_ksampler(seed, steps, cfg, denoise, ["12",0]),
    }
    if ckpt == SD15_CKPT:
        g["25"] = {"class_type":"VAELoader","inputs":{"vae_name":SD15_VAE}}
    _sd_condition(g, p, n, clip_skip)
    vae = ["25",0] if ckpt == SD15_CKPT else ["4",2]
    if ckpt == SD15_CKPT:
        g["12"]["inputs"]["vae"] = vae
    return _finish(g, ["3",0], seed, cfg, hires, vae=vae)


def instantid_graph(p, n, seed, steps, w, h, weight, image_name, cfg, hires, ckpt=CKPT):
    """Lock the face/identity from the uploaded image; generate a fresh image
    (pose/clothing/scene from the prompt) that keeps that face."""
    M, P, N = ["23",0], ["23",1], ["23",2]   # ApplyInstantID outputs (modified model + conditioning)
    g = {
        "4":{"class_type":"CheckpointLoaderSimple","inputs":{"ckpt_name":ckpt}},
        "10":{"class_type":"LoadImage","inputs":{"image":image_name}},
        "20":{"class_type":"InstantIDModelLoader","inputs":{"instantid_file":"ip-adapter.bin"}},
        "21":{"class_type":"InstantIDFaceAnalysis","inputs":{"provider":"CPU"}},
        "22":{"class_type":"ControlNetLoader","inputs":{"control_net_name":"instantid-controlnet.safetensors"}},
        "6":{"class_type":"CLIPTextEncode","inputs":{"text":p,"clip":["4",1]}},
        "7":{"class_type":"CLIPTextEncode","inputs":{"text":n,"clip":["4",1]}},
        "23":{"class_type":"ApplyInstantID","inputs":{
            "instantid":["20",0],"insightface":["21",0],"control_net":["22",0],"image":["10",0],
            "model":["4",0],"positive":["6",0],"negative":["7",0],
            "weight":weight,"start_at":0.0,"end_at":1.0}},
        "5":{"class_type":"EmptyLatentImage","inputs":{"width":w,"height":h,"batch_size":1}},
        "3":_ksampler(seed, steps, cfg, 1.0, ["5",0], M, P, N),
    }
    return _finish(g, ["3",0], seed, cfg, hires, M, P, N)


# ---------- FLUX.1-dev (GGUF) ----------
def _flux_loaders(lora=None, lora_strength=0.9, unet_name=FLUX_UNET):
    g = {
        "30":{"class_type":"UnetLoaderGGUF","inputs":{"unet_name":unet_name}},
        "31":{"class_type":"DualCLIPLoader","inputs":{"clip_name1":FLUX_T5,"clip_name2":FLUX_CLIP_L,"type":"flux"}},
        "32":{"class_type":"VAELoader","inputs":{"vae_name":FLUX_VAE}},
        "6":None,  # filled by caller
    }
    if lora and lora != "None":
        # Patch the GGUF UNET with a model-only LoRA (FLUX keeps CLIP separate).
        g["35"]={"class_type":"LoraLoaderModelOnly","inputs":{
            "model":["30",0],"lora_name":lora,"strength_model":lora_strength}}
    return g


def _flux_model(g):
    """Model ref for samplers: the LoRA-patched model when present, else the UNET."""
    return ["35",0] if "35" in g else ["30",0]


def _flux_sampler(seed, steps, denoise, latent, model=["30",0], pos=["33",0], neg=["7",0]):
    return {"class_type":"KSampler","inputs":{"seed":seed,"steps":steps,"cfg":1.0,"sampler_name":"euler","scheduler":"simple","denoise":denoise,"model":model,"positive":pos,"negative":neg,"latent_image":latent}}


def _flux_finish(g, base_latent, seed, hires, model=["30",0], pos=["33",0], neg=["7",0]):
    last = base_latent
    if hires:
        g["13"]={"class_type":"LatentUpscaleBy","inputs":{"samples":base_latent,"upscale_method":"bislerp","scale_by":1.5}}
        g["14"]=_flux_sampler(seed+1, 18, 0.45, ["13",0], model, pos, neg)
        last=["14",0]
    g["8"]={"class_type":"VAEDecode","inputs":{"samples":last,"vae":["32",0]}}
    g["9"]={"class_type":"SaveImage","inputs":{"filename_prefix":"web","images":["8",0]}}
    return g


def flux_pulid_graph(p, n, seed, steps, w, h, weight, image_name, cfg, hires, unet_name=FLUX_UNET):
    """FLUX with PuLID face-lock: keep the face from image_name, generate fresh."""
    g = _flux_loaders(unet_name=unet_name)
    g["40"]={"class_type":"PulidFluxModelLoader","inputs":{"pulid_file":PULID_FLUX}}
    g["41"]={"class_type":"PulidFluxEvaClipLoader","inputs":{}}
    g["42"]={"class_type":"PulidFluxInsightFaceLoader","inputs":{"provider":"CPU"}}
    g["10"]={"class_type":"LoadImage","inputs":{"image":image_name}}
    g["43"]={"class_type":"ApplyPulidFlux","inputs":{
        "model":["30",0],"pulid_flux":["40",0],"eva_clip":["41",0],"face_analysis":["42",0],
        "image":["10",0],"weight":weight,"start_at":0.0,"end_at":1.0}}
    g["6"]={"class_type":"CLIPTextEncode","inputs":{"text":p,"clip":["31",0]}}
    g["33"]={"class_type":"FluxGuidance","inputs":{"conditioning":["6",0],"guidance":FLUX_GUIDANCE}}
    g["7"]={"class_type":"CLIPTextEncode","inputs":{"text":"","clip":["31",0]}}
    g["5"]={"class_type":"EmptySD3LatentImage","inputs":{"width":w,"height":h,"batch_size":1}}
    g["3"]=_flux_sampler(seed, steps, 1.0, ["5",0], ["43",0])
    return _flux_finish(g, ["3",0], seed, hires, ["43",0])


def flux_pulid_body_graph(p, n, seed, steps, weight, denoise, face_name, body_name, cfg, hires, unet_name=FLUX_UNET):
    """Put the face from face_name (PuLID) onto the body/composition of body_name (img2img base)."""
    g = _flux_loaders(unet_name=unet_name)
    g["40"]={"class_type":"PulidFluxModelLoader","inputs":{"pulid_file":PULID_FLUX}}
    g["41"]={"class_type":"PulidFluxEvaClipLoader","inputs":{}}
    g["42"]={"class_type":"PulidFluxInsightFaceLoader","inputs":{"provider":"CPU"}}
    g["10"]={"class_type":"LoadImage","inputs":{"image":face_name}}
    g["43"]={"class_type":"ApplyPulidFlux","inputs":{
        "model":["30",0],"pulid_flux":["40",0],"eva_clip":["41",0],"face_analysis":["42",0],
        "image":["10",0],"weight":weight,"start_at":0.0,"end_at":1.0}}
    g["50"]={"class_type":"LoadImage","inputs":{"image":body_name}}
    g["51"]={"class_type":"ImageScaleToTotalPixels","inputs":{"image":["50",0],"upscale_method":"lanczos","megapixels":1.0,"resolution_steps":64}}
    g["52"]={"class_type":"VAEEncode","inputs":{"pixels":["51",0],"vae":["32",0]}}
    g["6"]={"class_type":"CLIPTextEncode","inputs":{"text":p,"clip":["31",0]}}
    g["33"]={"class_type":"FluxGuidance","inputs":{"conditioning":["6",0],"guidance":FLUX_GUIDANCE}}
    g["7"]={"class_type":"CLIPTextEncode","inputs":{"text":"","clip":["31",0]}}
    g["3"]=_flux_sampler(seed, steps, denoise, ["52",0], ["43",0])
    return _flux_finish(g, ["3",0], seed, hires, ["43",0])


def flux_pulid_control_graph(p, n, seed, steps, w, h, weight, cn_strength, face_name, body_name, cfg, hires, unet_name=FLUX_UNET):
    """Face from face_name (PuLID) + body/pose locked from body_name via depth ControlNet.
    Generates from empty latent at denoise 1.0 so the prompt fully drives outfit/scene."""
    g = _flux_loaders(unet_name=unet_name)
    g["40"]={"class_type":"PulidFluxModelLoader","inputs":{"pulid_file":PULID_FLUX}}
    g["41"]={"class_type":"PulidFluxEvaClipLoader","inputs":{}}
    g["42"]={"class_type":"PulidFluxInsightFaceLoader","inputs":{"provider":"CPU"}}
    g["10"]={"class_type":"LoadImage","inputs":{"image":face_name}}
    g["43"]={"class_type":"ApplyPulidFlux","inputs":{
        "model":["30",0],"pulid_flux":["40",0],"eva_clip":["41",0],"face_analysis":["42",0],
        "image":["10",0],"weight":weight,"start_at":0.0,"end_at":1.0}}
    g["50"]={"class_type":"LoadImage","inputs":{"image":body_name}}
    g["51"]={"class_type":"ImageScale","inputs":{"image":["50",0],"upscale_method":"lanczos","width":w,"height":h,"crop":"center"}}
    g["53"]={"class_type":"DepthAnythingV2Preprocessor","inputs":{"image":["51",0],"ckpt_name":"depth_anything_v2_vitl.pth","resolution":1024}}
    g["54"]={"class_type":"ControlNetLoader","inputs":{"control_net_name":"flux-depth-controlnet.safetensors"}}
    g["6"]={"class_type":"CLIPTextEncode","inputs":{"text":p,"clip":["31",0]}}
    g["33"]={"class_type":"FluxGuidance","inputs":{"conditioning":["6",0],"guidance":FLUX_GUIDANCE}}
    g["7"]={"class_type":"CLIPTextEncode","inputs":{"text":"","clip":["31",0]}}
    g["55"]={"class_type":"ControlNetApplyAdvanced","inputs":{
        "positive":["33",0],"negative":["7",0],"control_net":["54",0],"image":["53",0],
        "strength":cn_strength,"start_percent":0.0,"end_percent":0.8,"vae":["32",0]}}
    g["5"]={"class_type":"EmptySD3LatentImage","inputs":{"width":w,"height":h,"batch_size":1}}
    g["3"]=_flux_sampler(seed, steps, 1.0, ["5",0], ["43",0], ["55",0], ["55",1])
    return _flux_finish(g, ["3",0], seed, hires, ["43",0], ["55",0], ["55",1])


def flux_txt2img_graph(p, n, seed, steps, w, h, cfg, hires, lora=None, lora_strength=0.9, unet_name=FLUX_UNET):
    g = _flux_loaders(lora, lora_strength, unet_name)
    m = _flux_model(g)
    g["6"]={"class_type":"CLIPTextEncode","inputs":{"text":p,"clip":["31",0]}}
    g["33"]={"class_type":"FluxGuidance","inputs":{"conditioning":["6",0],"guidance":FLUX_GUIDANCE}}
    g["7"]={"class_type":"CLIPTextEncode","inputs":{"text":"","clip":["31",0]}}
    g["5"]={"class_type":"EmptySD3LatentImage","inputs":{"width":w,"height":h,"batch_size":1}}
    g["3"]=_flux_sampler(seed, steps, 1.0, ["5",0], m)
    return _flux_finish(g, ["3",0], seed, hires, m)


def flux_img2img_graph(p, n, seed, steps, denoise, image_name, cfg, hires, lora=None, lora_strength=0.9, unet_name=FLUX_UNET):
    g = _flux_loaders(lora, lora_strength, unet_name)
    m = _flux_model(g)
    g["10"]={"class_type":"LoadImage","inputs":{"image":image_name}}
    g["11"]={"class_type":"ImageScaleToTotalPixels","inputs":{"image":["10",0],"upscale_method":"lanczos","megapixels":1.0,"resolution_steps":64}}
    g["12"]={"class_type":"VAEEncode","inputs":{"pixels":["11",0],"vae":["32",0]}}
    g["6"]={"class_type":"CLIPTextEncode","inputs":{"text":p,"clip":["31",0]}}
    g["33"]={"class_type":"FluxGuidance","inputs":{"conditioning":["6",0],"guidance":FLUX_GUIDANCE}}
    g["7"]={"class_type":"CLIPTextEncode","inputs":{"text":"","clip":["31",0]}}
    g["3"]=_flux_sampler(seed, steps, denoise, ["12",0], m)
    return _flux_finish(g, ["3",0], seed, hires, m)


# ---------- FLUX.2-Klein (distilled, CFGGuider + SamplerCustomAdvanced) ----------
def _flux2_loaders(lora=None, lora_strength=0.9):
    g = {
        "60":{"class_type":"UNETLoader","inputs":{"unet_name":FLUX2_UNET,"weight_dtype":"default"}},
        "61":{"class_type":"CLIPLoader","inputs":{"clip_name":FLUX2_CLIP,"type":"flux2","device":"default"}},
        "62":{"class_type":"VAELoader","inputs":{"vae_name":FLUX2_VAE}},
    }
    if lora and lora != "None":
        # Patch the FLUX.2 UNET with a model-only LoRA (the LoRA must be a flux2_klein LoRA).
        g["80"]={"class_type":"LoraLoaderModelOnly","inputs":{
            "model":["60",0],"lora_name":lora,"strength_model":lora_strength}}
    return g


def _flux2_model(g):
    """Model ref for the guider: the LoRA-patched model when present, else the UNET."""
    return ["80",0] if "80" in g else ["60",0]


def _flux2_cond_guider(p, cfg, model_ref=["60",0]):
    # positive = prompt; negative = ConditioningZeroOut(positive). Distilled (cfg ~1.0),
    # so there is no separate negative-text path (matches the known-good Klein recipe).
    return {
        "63":{"class_type":"CLIPTextEncode","inputs":{"text":p,"clip":["61",0]}},
        "64":{"class_type":"ConditioningZeroOut","inputs":{"conditioning":["63",0]}},
        "65":{"class_type":"CFGGuider","inputs":{"model":model_ref,"positive":["63",0],"negative":["64",0],"cfg":cfg}},
        "66":{"class_type":"KSamplerSelect","inputs":{"sampler_name":"euler"}},
    }


def _flux2_finish(g, latent_ref):
    g["71"]={"class_type":"VAEDecode","inputs":{"samples":latent_ref,"vae":["62",0]}}
    g["72"]={"class_type":"SaveImage","inputs":{"filename_prefix":"web","images":["71",0]}}
    return g


def flux2_txt2img_graph(p, n, seed, steps, w, h, cfg, hires, lora=None, lora_strength=0.9):
    g = _flux2_loaders(lora, lora_strength)
    g.update(_flux2_cond_guider(p, cfg, _flux2_model(g)))
    g["67"]={"class_type":"Flux2Scheduler","inputs":{"steps":steps,"width":w,"height":h}}
    g["68"]={"class_type":"EmptyFlux2LatentImage","inputs":{"width":w,"height":h,"batch_size":1}}
    g["69"]={"class_type":"RandomNoise","inputs":{"noise_seed":seed}}
    g["70"]={"class_type":"SamplerCustomAdvanced","inputs":{
        "noise":["69",0],"guider":["65",0],"sampler":["66",0],"sigmas":["67",0],"latent_image":["68",0]}}
    return _flux2_finish(g, ["70",0])


def flux2_img2img_graph(p, n, seed, steps, denoise, image_name, w, h, cfg, hires, lora=None, lora_strength=0.9):
    g = _flux2_loaders(lora, lora_strength)
    g.update(_flux2_cond_guider(p, cfg, _flux2_model(g)))
    g["73"]={"class_type":"LoadImage","inputs":{"image":image_name}}
    g["74"]={"class_type":"ImageScaleToTotalPixels","inputs":{"image":["73",0],"upscale_method":"lanczos","megapixels":1.0,"resolution_steps":64}}
    g["75"]={"class_type":"VAEEncode","inputs":{"pixels":["74",0],"vae":["62",0]}}
    g["67"]={"class_type":"Flux2Scheduler","inputs":{"steps":steps,"width":w,"height":h}}
    # split the sigma schedule so only the last `denoise` fraction runs (img2img)
    g["76"]={"class_type":"SplitSigmasDenoise","inputs":{"sigmas":["67",0],"denoise":denoise}}
    g["69"]={"class_type":"RandomNoise","inputs":{"noise_seed":seed}}
    g["70"]={"class_type":"SamplerCustomAdvanced","inputs":{
        "noise":["69",0],"guider":["65",0],"sampler":["66",0],"sigmas":["76",1],"latent_image":["75",0]}}
    return _flux2_finish(g, ["70",0])


# ---- Qwen-Image-2.1 ----
# Text encoder on CPU: the bf16 UNet (14 GB) + int8 Qwen3-VL-8B (8.9 GB) + VAE don't all fit
# one GPU at once under --gpu-only; the encoder runs once per generation, so CPU costs little.
# VAE decode is tiled (full-frame decode OOMs at 1 MP). Output is square: the conditioning
# node's latent output is resolution x resolution.
def _qwen21_loaders(lora=None, lora_strength=0.9):
    g = {
        "60":{"class_type":"UNETLoader","inputs":{"unet_name":QWEN21_UNET,"weight_dtype":"default"}},
        "62":{"class_type":"VAELoader","inputs":{"vae_name":QWEN21_VAE}},
    }
    if not QWEN21_REMOTE_ENC:  # local fallback: CPU-resident encoder
        g["61"]={"class_type":"CLIPLoader","inputs":{"clip_name":QWEN21_CLIP,"type":"qwen_image","device":"cpu"}}
    if lora and lora != "None":
        g["80"]={"class_type":"LoraLoaderModelOnly","inputs":{
            "model":["60",0],"lora_name":lora,"strength_model":lora_strength}}
    return g


def _qwen21_model(g):
    return ["80",0] if "80" in g else ["60",0]


def _qwen21_finish(g, latent_ref):
    g["71"]={"class_type":"VAEDecodeTiled","inputs":{
        "samples":latent_ref,"vae":["62",0],"tile_size":256,"overlap":64,"temporal_size":64,"temporal_overlap":8}}
    g["72"]={"class_type":"SaveImage","inputs":{"filename_prefix":"web","images":["71",0]}}
    return g


def _qwen21_encode_node(p, n, res):
    if QWEN21_REMOTE_ENC:
        return {"class_type":"RemoteQwenEncode","inputs":{
            "prompt":p,"negative_prompt":n,"resolution":res,"server":QWEN21_REMOTE_ENC}}
    return {"class_type":"TextEncodeQwenImage21","inputs":{
        "clip":["61",0],"prompt":p,"negative_prompt":n,"resolution":res,"images":{}}}


def qwen21_txt2img_graph(p, n, seed, steps, w, h, cfg, hires, lora=None, lora_strength=0.9):
    g = _qwen21_loaders(lora, lora_strength)
    model = _qwen21_model(g)
    res = max(32, (min(w, h) // 32) * 32)
    g["63"]=_qwen21_encode_node(p, n, res)
    g["65"]={"class_type":"CFGGuider","inputs":{"model":model,"positive":["63",0],"negative":["63",1],"cfg":cfg}}
    g["66"]={"class_type":"KSamplerSelect","inputs":{"sampler_name":"euler"}}
    g["67"]={"class_type":"BasicScheduler","inputs":{"model":model,"scheduler":"simple","steps":steps,"denoise":1.0}}
    g["69"]={"class_type":"RandomNoise","inputs":{"noise_seed":seed}}
    g["70"]={"class_type":"SamplerCustomAdvanced","inputs":{
        "noise":["69",0],"guider":["65",0],"sampler":["66",0],"sigmas":["67",0],"latent_image":["63",2]}}
    return _qwen21_finish(g, ["70",0])


def qwen21_img2img_graph(p, n, seed, steps, denoise, image_name, w, h, cfg, hires, lora=None, lora_strength=0.9):
    g = _qwen21_loaders(lora, lora_strength)
    model = _qwen21_model(g)
    res = max(32, (min(w, h) // 32) * 32)
    g["63"]=_qwen21_encode_node(p, n, res)
    g["65"]={"class_type":"CFGGuider","inputs":{"model":model,"positive":["63",0],"negative":["63",1],"cfg":cfg}}
    g["66"]={"class_type":"KSamplerSelect","inputs":{"sampler_name":"euler"}}
    g["73"]={"class_type":"LoadImage","inputs":{"image":image_name}}
    g["74"]={"class_type":"ImageScaleToTotalPixels","inputs":{"image":["73",0],"upscale_method":"lanczos","megapixels":1.0,"resolution_steps":64}}
    g["75"]={"class_type":"VAEEncode","inputs":{"pixels":["74",0],"vae":["62",0]}}
    # BasicScheduler trims the sigma schedule by denoise natively (model's own shift settings).
    g["67"]={"class_type":"BasicScheduler","inputs":{"model":model,"scheduler":"simple","steps":steps,"denoise":denoise}}
    g["69"]={"class_type":"RandomNoise","inputs":{"noise_seed":seed}}
    g["70"]={"class_type":"SamplerCustomAdvanced","inputs":{
        "noise":["69",0],"guider":["65",0],"sampler":["66",0],"sigmas":["67",0],"latent_image":["75",0]}}
    return _qwen21_finish(g, ["70",0])


# ---- FLUX.3-klein ----
# Distilled like FLUX.2 (4 steps, CFG 1, ZeroOut negative); reuses the flux2 guider/finish
# helpers. BasicScheduler (not Flux2Scheduler) so the model's own timestep shift applies.
def _flux3_loaders(lora=None, lora_strength=0.9):
    g = {
        "60":{"class_type":"UNETLoader","inputs":{"unet_name":FLUX3_UNET,"weight_dtype":"default"}},
        "61":{"class_type":"CLIPLoader","inputs":{"clip_name":FLUX3_CLIP,"type":"flux3","device":"default"}},
        "62":{"class_type":"VAELoader","inputs":{"vae_name":FLUX3_VAE}},
    }
    if lora and lora != "None":
        g["80"]={"class_type":"LoraLoaderModelOnly","inputs":{
            "model":["60",0],"lora_name":lora,"strength_model":lora_strength}}
    return g


def _flux3_model(g):
    return ["80",0] if "80" in g else ["60",0]


def _flux3_finish(g, latent_ref):
    # Tiled decode: full-frame VAEDecode needs ~9 GB activations on top of the 19.6 GB of
    # bf16 models and OOMs the card; tiles keep it under ~2 GB.
    g["71"]={"class_type":"VAEDecodeTiled","inputs":{
        "samples":latent_ref,"vae":["62",0],"tile_size":512,"overlap":64,"temporal_size":64,"temporal_overlap":8}}
    g["72"]={"class_type":"SaveImage","inputs":{"filename_prefix":"web","images":["71",0]}}
    return g


def _minimax_loaders():
    return {
        "60": {"class_type":"UnetLoaderGGUF","inputs":{"unet_name":MINIMAX_UNET}},
        "61": {"class_type":"CLIPLoader","inputs":{"clip_name":MINIMAX_CLIP,"type":"minimax","device":"cpu"}},
        "62": {"class_type":"VAELoader","inputs":{"vae_name":MINIMAX_VAE}},
    }


def _minimax_finish(g, latent_ref):
    g["71"]={"class_type":"VAEDecode","inputs":{"samples":latent_ref,"vae":["62",0]}}
    g["72"]={"class_type":"SaveImage","inputs":{"filename_prefix":"web","images":["71",0]}}
    return g


def _minimax_sample(g, model_ref, cond_ref, latent_ref, seed, steps):
    # 6-step PDD-distilled: CFG 1 (negative ignored), euler; the sigma schedule ships
    # in the GGUF metadata.
    g["65"]={"class_type":"CFGGuider","inputs":{"model":model_ref,"positive":cond_ref,"negative":cond_ref,"cfg":1.0}}
    g["66"]={"class_type":"KSamplerSelect","inputs":{"sampler_name":"euler"}}
    g["67"]={"class_type":"BasicScheduler","inputs":{"model":model_ref,"scheduler":"simple","steps":steps,"denoise":1.0}}
    g["69"]={"class_type":"RandomNoise","inputs":{"noise_seed":seed}}
    g["70"]={"class_type":"SamplerCustomAdvanced","inputs":{
        "noise":["69",0],"guider":["65",0],"sampler":["66",0],"sigmas":["67",0],"latent_image":latent_ref}}


def minimax_txt2img_graph(p, n, seed, steps, w, h, cfg, hires, lora=None, lora_strength=0.9):
    g = _minimax_loaders()
    g["63"]={"class_type":"MiniMaxH3ImageConditioning","inputs":{
        "clip":["61",0],"vae":["62",0],"prompt":p,"width":w,"height":h}}
    _minimax_sample(g, ["60",0], ["63",0], ["63",1], seed, steps)
    return _minimax_finish(g, ["70",0])


def minimax_img2img_graph(p, n, seed, steps, denoise, image_name, w, h, cfg, hires, lora=None, lora_strength=0.9):
    # The conditioning node treats image_1 as the picture to edit; the denoise slider
    # is ignored (6-step distilled model, full schedule).
    g = _minimax_loaders()
    g["73"]={"class_type":"LoadImage","inputs":{"image":image_name}}
    g["63"]={"class_type":"MiniMaxH3ImageConditioning","inputs":{
        "clip":["61",0],"vae":["62",0],"prompt":p,"width":0,"height":0,
        "images":{"image_1":["73",0]}}}
    _minimax_sample(g, ["60",0], ["63",0], ["63",1], seed, steps)
    return _minimax_finish(g, ["70",0])


def flux3_txt2img_graph(p, n, seed, steps, w, h, cfg, hires, lora=None, lora_strength=0.9):
    g = _flux3_loaders(lora, lora_strength)
    model = _flux3_model(g)
    g.update(_flux2_cond_guider(p, cfg, model))
    g["67"]={"class_type":"BasicScheduler","inputs":{"model":model,"scheduler":"simple","steps":steps,"denoise":1.0}}
    g["68"]={"class_type":"EmptyFlux2LatentImage","inputs":{"width":w,"height":h,"batch_size":1}}
    g["69"]={"class_type":"RandomNoise","inputs":{"noise_seed":seed}}
    g["70"]={"class_type":"SamplerCustomAdvanced","inputs":{
        "noise":["69",0],"guider":["65",0],"sampler":["66",0],"sigmas":["67",0],"latent_image":["68",0]}}
    return _flux3_finish(g, ["70",0])


def flux3_img2img_graph(p, n, seed, steps, denoise, image_name, w, h, cfg, hires, lora=None, lora_strength=0.9):
    g = _flux3_loaders(lora, lora_strength)
    model = _flux3_model(g)
    g.update(_flux2_cond_guider(p, cfg, model))
    g["73"]={"class_type":"LoadImage","inputs":{"image":image_name}}
    g["74"]={"class_type":"ImageScaleToTotalPixels","inputs":{"image":["73",0],"upscale_method":"lanczos","megapixels":1.0,"resolution_steps":64}}
    g["75"]={"class_type":"VAEEncode","inputs":{"pixels":["74",0],"vae":["62",0]}}
    g["67"]={"class_type":"BasicScheduler","inputs":{"model":model,"scheduler":"simple","steps":steps,"denoise":denoise}}
    g["69"]={"class_type":"RandomNoise","inputs":{"noise_seed":seed}}
    g["70"]={"class_type":"SamplerCustomAdvanced","inputs":{
        "noise":["69",0],"guider":["65",0],"sampler":["66",0],"sigmas":["67",0],"latent_image":["75",0]}}
    # _flux3_finish (tiled decode), NOT _flux2_finish: the full-frame VAEDecode wants ~9 GB
    # activations on top of ~21 GB resident (bf16 UNet + GPU-resident 4B encoder + VAE) and
    # OOMs the card. flux3 img2img used the flux2 finish since wiring and died at the decode.
    return _flux3_finish(g, ["70",0])


_last_engine = None

async def _free_vram_on_switch(engine):
    """The bf16 engines (qwen21/flux3/flux2/flux1) don't co-reside on one GPU: loading one
    while another is cached OOMs or trips a HIP illegal-memory-access. Evict the previous
    engine's models when the selection changes; same-engine runs keep the cache warm."""
    global _last_engine
    if _last_engine != engine:
        # Also fires on the first request after a webui restart (_last_engine is None):
        # ComfyUI may still hold the previous engine's models even though we lost track.
        timeout = ClientTimeout(total=60)
        async with ClientSession(timeout=timeout) as s:
            async with s.post(f"{COMFY}/free", json={"unload_models": True, "free_memory": True}) as r:
                await r.read()
    _last_engine = engine


async def run_graph(graph):
    timeout = ClientTimeout(total=600)
    async with ClientSession(timeout=timeout) as s:
        async with s.post(f"{COMFY}/prompt", json={"prompt": graph}) as r:
            resp = await r.json()
        if "prompt_id" not in resp:
            details = []
            for info in (resp.get("node_errors") or {}).values():
                for e in info.get("errors", []):
                    details.append(f"{info.get('class_type','?')}: {e.get('message','')} {e.get('details','')}".strip())
            msg = (resp.get("error", {}) or {}).get("message", "submit failed")
            raise RuntimeError("; ".join([msg] + details) if details else msg)
        pid = resp["prompt_id"]
        for _ in range(1800):   # up to ~30 min: covers FLUX + PuLID + queue waits
            async with s.get(f"{COMFY}/history/{pid}") as r:
                h = await r.json()
            if pid in h:
                for node in h[pid].get("outputs", {}).values():
                    for img in node.get("images", []):
                        return img
                raise RuntimeError("no image produced")
            await asyncio.sleep(1)
    raise RuntimeError("timed out")


async def page(_):
    # Inject the current LoRA list each load, so dropping a new LoRA in just needs a refresh.
    html = PAGE.replace("<!--LORA_OPTIONS-->", _lora_options_html())
    return web.Response(text=html, content_type="text/html")


async def _upload_image(image_bytes, fn):
    async with ClientSession() as s:
        fd = FormData()
        uniq = f"up_{random.randint(10**8, 10**9)}_{fn or 'upload.png'}"
        fd.add_field("image", image_bytes, filename=uniq, content_type="application/octet-stream")
        fd.add_field("overwrite", "true")
        async with s.post(f"{COMFY}/upload/image", data=fd) as r:
            up = await r.json()
    name = up.get("name") or up.get("filename")
    if up.get("subfolder"):
        name = f"{up['subfolder']}/{name}"
    return name


async def api_generate(request):
    reader = await request.multipart()
    f = {}; image_bytes = None; image_fn = None; image2_bytes = None; image2_fn = None
    async for part in reader:
        if part.name == "image" and part.filename:
            image_bytes = await part.read(decode=False); image_fn = part.filename
        elif part.name == "image2" and part.filename:
            image2_bytes = await part.read(decode=False); image2_fn = part.filename
        else:
            f[part.name] = (await part.read(decode=True)).decode("utf-8", "ignore")
    prompt = (f.get("prompt") or "").strip()
    neg = (f.get("negative") or DEFAULT_NEG).strip()
    try: steps = max(8, min(60, int(float(f.get("steps", 28)))))
    except: steps = 28
    try: denoise = max(0.2, min(0.95, float(f.get("denoise", 0.6))))
    except: denoise = 0.6
    try: body_change = max(0.2, min(1.0, float(f.get("body_change", 0.6))))
    except: body_change = 0.6
    try: cfg = max(1.0, min(14.0, float(f.get("cfg", 6.5))))
    except: cfg = 6.5
    hires = f.get("hires") in ("1", "true", "on", "yes")
    keepface = f.get("keepface") in ("1", "true", "on", "yes")
    engine = (f.get("engine") or "flux").lower()
    if engine == "qwen21":
        # CFG 12 pins the Qwen VAE's blue channel at 0, so the saved picture is yellow.
        cfg = min(cfg, 4.0)
    await _free_vram_on_switch(engine)
    lora = f.get("lora") or "None"
    try: lora_strength = max(0.0, min(2.0, float(f.get("lora_strength", 0.9))))
    except: lora_strength = 0.9
    shape = f.get("shape", "1024x1024")
    try: w, h = (int(x) for x in shape.split("x"))
    except: w, h = 1024, 1024
    seed_raw = (f.get("seed") or "").strip()
    try:
        seed = int(seed_raw) & (2**31 - 1) if seed_raw else random.randint(0, 2**31 - 1)
    except ValueError:
        seed = random.randint(0, 2**31 - 1)
    if engine in ("flux2", "flux3", "schnell"):
        # distilled models: default 4 steps / cfg 1.0, but honour an explicit lower steps value
        # (the global steps clamp floors at 8, which is wrong for a 4-step model).
        try: steps = max(1, min(50, int(float(f.get("steps", 4)))))
        except: steps = 4
        cfg = 1.0
    try:
        # Two images: face from image 1 (PuLID) onto body/composition of image 2 (FLUX, always).
        if image_bytes and image2_bytes:
            # derive output size from Image 2's aspect (~1MP, mult of 16) so the body isn't stretched
            try:
                from PIL import Image as _PI; import io as _io, math as _m
                bw, bh = _PI.open(_io.BytesIO(image2_bytes)).size
                ar = bw / bh
                w = max(256, int(round((1024 * _m.sqrt(ar)) / 16)) * 16)
                h = max(256, int(round((1024 / _m.sqrt(ar)) / 16)) * 16)
            except Exception:
                w, h = 1024, 1024
            face_name = await _upload_image(image_bytes, image_fn)
            body_name = await _upload_image(image2_bytes, image2_fn)
            # denoise slider = PuLID face weight; body_change slider = ControlNet (pose-lock) strength
            unet_name = FLUX_ENGINES.get(engine, FLUX_UNET)
            img = await run_graph(flux_pulid_control_graph(prompt, neg, seed, steps, w, h, denoise, body_change, face_name, body_name, cfg, hires, unet_name))
        elif image_bytes or image2_bytes:
            # Either uploaded image can be a standalone img2img source. Image 1
            # retains optional identity locking; Image 2 is prompt-driven only.
            source_bytes = image_bytes if image_bytes else image2_bytes
            source_fn = image_fn if image_bytes else image2_fn
            name = await _upload_image(source_bytes, source_fn)
            if engine == "flux2":
                # FLUX.2-Klein img2img (no face-lock model staged; keepface ignored)
                img = await run_graph(flux2_img2img_graph(prompt, neg, seed, steps, denoise, name, w, h, cfg, hires, lora, lora_strength))
            elif engine == "flux3":
                img = await run_graph(flux3_img2img_graph(prompt, neg, seed, steps, denoise, name, w, h, cfg, hires, lora, lora_strength))
            elif engine == "qwen21":
                img = await run_graph(qwen21_img2img_graph(prompt, neg, seed, steps, denoise, name, w, h, cfg, hires, lora, lora_strength))
            elif engine == "minimax":
                img = await run_graph(minimax_img2img_graph(prompt, neg, seed, steps, denoise, name, w, h, cfg, hires, lora, lora_strength))
            elif engine in FLUX_ENGINES:
                unet_name = FLUX_ENGINES[engine]
                if keepface and image_bytes:
                    # denoise slider doubles as PuLID identity weight here
                    img = await run_graph(flux_pulid_graph(prompt, neg, seed, steps, w, h, denoise, name, cfg, hires, unet_name))
                else:
                    img = await run_graph(flux_img2img_graph(prompt, neg, seed, steps, denoise, name, cfg, hires, lora, lora_strength, unet_name))
            elif keepface and image_bytes and engine in SDXL_ENGINES:
                # denoise slider doubles as InstantID identity weight here
                ckpt = SDXL_ENGINES[engine]
                img = await run_graph(instantid_graph(prompt, neg, seed, steps, w, h, denoise, name, cfg, hires, ckpt))
            else:
                clip_skip = -2 if engine == "pony" else 0
                ckpt = SDXL_ENGINES.get(engine, SD15_CKPT if engine == "sd15" else CKPT)
                img = await run_graph(img2img_graph(prompt, neg, seed, steps, denoise, name, cfg, hires, ckpt, clip_skip))
        else:
            if engine == "flux2":
                img = await run_graph(flux2_txt2img_graph(prompt, neg, seed, steps, w, h, cfg, hires, lora, lora_strength))
            elif engine == "flux3":
                img = await run_graph(flux3_txt2img_graph(prompt, neg, seed, steps, w, h, cfg, hires, lora, lora_strength))
            elif engine == "qwen21":
                img = await run_graph(qwen21_txt2img_graph(prompt, neg, seed, steps, w, h, cfg, hires, lora, lora_strength))
            elif engine == "minimax":
                img = await run_graph(minimax_txt2img_graph(prompt, neg, seed, steps, w, h, cfg, hires, lora, lora_strength))
            elif engine in FLUX_ENGINES:
                img = await run_graph(flux_txt2img_graph(prompt, neg, seed, steps, w, h, cfg, hires, lora, lora_strength, FLUX_ENGINES[engine]))
            elif engine in SDXL_ENGINES:
                clip_skip = -2 if engine == "pony" else 0
                img = await run_graph(txt2img_graph(prompt, neg, seed, steps, w, h, cfg, hires, SDXL_ENGINES[engine], clip_skip))
            elif engine == "sd15":
                img = await run_graph(txt2img_graph(prompt, neg, seed, steps, w, h, cfg, hires, SD15_CKPT))
            else:
                img = await run_graph(txt2img_graph(prompt, neg, seed, steps, w, h, cfg, hires))
        from urllib.parse import urlencode
        q = urlencode({"filename": img["filename"], "subfolder": img.get("subfolder",""), "type": img.get("type","output")})
        return web.json_response({"image": f"/img?{q}"})
    except Exception as e:
        return web.json_response({"error": str(e)}, status=500)


async def api_interrupt(_request):
    """Interrupt the currently executing ComfyUI prompt."""
    try:
        async with ClientSession(timeout=ClientTimeout(total=10)) as s:
            async with s.post(f"{COMFY}/interrupt") as r:
                if r.status >= 400:
                    return web.json_response({"error": await r.text()}, status=502)
        return web.json_response({"ok": True})
    except Exception as e:
        return web.json_response({"error": str(e)}, status=502)


def _engine_label(graph):
    """Best-effort guess at which engine a queued/running graph is using, for display only."""
    # The bf16 UNETLoader engines (flux2/qwen21/flux3) all share node 60 (UNETLoader) and
    # node 65 (CFGGuider, via the shared _flux2_cond_guider helper), so the node-ID checks
    # below can't tell them apart — a qwen21 or flux3 graph used to fall into the `if "65"`
    # branch and get mislabeled "flux2". Identify by the actual UNET filename first.
    unet60 = (graph.get("60", {}).get("inputs") or {}).get("unet_name", "")
    if unet60 == QWEN21_UNET:
        return "qwen21"
    if unet60 == MINIMAX_UNET:
        return "minimax"
    if unet60 == FLUX3_UNET:
        return "flux3"
    if unet60 == FLUX2_UNET:
        return "flux2"
    if "65" in graph:
        return "flux2"
    if "30" in graph:
        unet = (graph.get("30", {}).get("inputs") or {}).get("unet_name", "")
        for label, fname in FLUX_ENGINES.items():
            if fname == unet:
                return label
        return "flux"
    if "4" in graph and graph["4"].get("class_type") == "CheckpointLoaderSimple":
        ckpt = (graph.get("4", {}).get("inputs") or {}).get("ckpt_name", "")
        for label, fname in SDXL_ENGINES.items():
            if fname == ckpt:
                return label
        if ckpt == SD15_CKPT:
            return "sd15"
        return "sdxl"
    return "?"


def _model_lora(graph):
    """(model, lora) filenames used by a graph, for gallery captions. Scans every node
    rather than fixed IDs because the engines use different node numbers; multiple LoRAs
    (rare) join with '+'. Returns (None, None) when the graph predates metadata embedding."""
    model = None
    loras = []
    for node in (graph or {}).values():
        if not isinstance(node, dict):
            continue
        ct, inputs = node.get("class_type", ""), (node.get("inputs") or {})
        if ct in ("UNETLoader", "CheckpointLoaderSimple"):
            model = inputs.get("unet_name") or inputs.get("ckpt_name") or model
        elif ct in ("LoraLoaderModelOnly", "LoraLoader") and inputs.get("lora_name"):
            loras.append(inputs["lora_name"])
    return model, ("+".join(loras) if loras else None)


async def api_queue(_request):
    """Lightweight summary of the ComfyUI queue for the status panel (no polling loop server-side)."""
    try:
        async with ClientSession(timeout=ClientTimeout(total=10)) as s:
            async with s.get(f"{COMFY}/queue") as r:
                if r.status >= 400:
                    return web.json_response({"error": await r.text()}, status=502)
                q = await r.json()
    except Exception as e:
        return web.json_response({"error": str(e)}, status=502)

    def summarize(items):
        out = []
        for item in items:
            graph = item[2] if len(item) > 2 else {}
            out.append({"id": item[1], "engine": _engine_label(graph)})
        return out

    running = summarize(q.get("queue_running", []))
    pending = summarize(q.get("queue_pending", []))
    return web.json_response({"running": running, "pending": pending,
                              "running_count": len(running), "pending_count": len(pending)})


async def img(request):
    from urllib.parse import urlencode
    q = urlencode({k: request.query.get(k, "") for k in ("filename","subfolder","type")})
    async with ClientSession() as s:
        async with s.get(f"{COMFY}/view?{q}") as r:
            data = await r.read()
            ct = r.headers.get("Content-Type", "image/png")
    return web.Response(body=data, content_type=ct.split(";")[0])


def _esc(value):
    return (str(value).replace("&", "&amp;").replace("<", "&lt;")
            .replace(">", "&gt;").replace('"', "&quot;"))


def _shot_from_graph(graph):
    """Pull the prompt and the knobs that made an image out of a saved ComfyUI graph."""
    prompt = negative = None
    steps = cfg = seed = denoise = sampler = scheduler = guidance = None
    width = height = megapixels = resolution = None
    model = vae = None
    loras = []
    refs = {}
    for node_id, node in (graph or {}).items():
        if not isinstance(node, dict):
            continue
        ct = node.get("class_type", "")
        inputs = node.get("inputs") or {}
        if ct in ("RemoteQwenEncode", "TextEncodeQwenImage21"):
            prompt = inputs.get("prompt") or prompt
            negative = inputs.get("negative_prompt") or negative
            resolution = inputs.get("resolution") or resolution
        elif ct == "MiniMaxH3ImageConditioning":
            prompt = inputs.get("prompt") or prompt
            width = inputs.get("width") or width
            height = inputs.get("height") or height
        elif ct == "CLIPTextEncode":
            text = inputs.get("text")
            if text and prompt is None:
                prompt = text
            elif text and negative is None and text != prompt:
                negative = text
        elif ct in ("UNETLoader", "UnetLoaderGGUF", "CheckpointLoaderSimple"):
            model = inputs.get("unet_name") or inputs.get("ckpt_name") or model
        elif ct == "VAELoader":
            vae = inputs.get("vae_name") or vae
        elif ct in ("LoraLoaderModelOnly", "LoraLoader") and inputs.get("lora_name"):
            strength = inputs.get("strength_model")
            loras.append(inputs["lora_name"] + (f"  ({strength})" if strength not in (None, "") else ""))
        elif ct == "LoadImage":
            if inputs.get("image"):
                refs[node_id] = inputs["image"]
        elif ct in ("KSampler", "KSamplerAdvanced") and steps is None:
            # First sampler wins: the hires second pass (18 steps, denoise 0.45)
            # must not overwrite the base pass settings.
            steps = inputs.get("steps", steps)
            cfg = inputs.get("cfg", cfg)
            seed = inputs.get("seed", seed)
            denoise = inputs.get("denoise", denoise)
            sampler = inputs.get("sampler_name", sampler)
            scheduler = inputs.get("scheduler", scheduler)
        elif ct == "FluxGuidance":
            guidance = inputs.get("guidance", guidance)
        elif ct == "CFGGuider":
            cfg = inputs.get("cfg", cfg)
        elif ct == "BasicScheduler":
            steps = inputs.get("steps", steps)
            denoise = inputs.get("denoise", denoise)
            scheduler = inputs.get("scheduler", scheduler)
        elif ct == "KSamplerSelect":
            sampler = inputs.get("sampler_name", sampler)
        elif ct == "RandomNoise":
            seed = inputs.get("noise_seed", seed)
        elif ct in ("EmptyLatentImage", "EmptySD3LatentImage"):
            width = inputs.get("width", width)
            height = inputs.get("height", height)
        elif ct == "ImageScaleToTotalPixels":
            megapixels = inputs.get("megapixels", megapixels)
    size = None
    if width and height:
        size = f"{width} × {height}"
    elif megapixels:
        size = f"{megapixels} megapixels"
    elif resolution:
        size = f"{resolution} × {resolution}"
    rows = [
        ("Engine", _engine_label(graph) if graph else None),
        ("Model", model),
        ("LoRA", " + ".join(loras) if loras else None),
        ("VAE", vae),
        ("Size", size),
        ("Steps", steps),
        ("CFG", cfg),
        ("Guidance", guidance),
        ("Sampler", sampler),
        ("Scheduler", scheduler),
        ("Seed", seed),
        ("Denoise", denoise),
    ]
    # Two-image graphs load the face at node 10 and the body at node 50; single-image
    # graphs load one source (node 10 or 73). Name them so the shot can be replicated.
    if "50" in refs:
        rows.append(("Reference head", refs.get("10")))
        rows.append(("Reference body", refs["50"]))
    elif refs:
        rows.append(("Started from", next(iter(refs.values()))))
    return prompt, negative, [(k, v) for k, v in rows if v not in (None, "", "?")]


def _output_image_path(filename):
    if (not isinstance(filename, str) or filename != os.path.basename(filename)
            or not filename.lower().endswith((".png", ".jpg", ".jpeg", ".webp"))):
        return None
    path = os.path.join(OUTPUT_DIR, filename)
    return path if os.path.isfile(path) else None


async def shot(request):
    """One image, the prompt, and the settings, in their own tab."""
    filename = request.query.get("filename", "")
    path = _output_image_path(filename)
    if path is None:
        return web.Response(status=404, text="Image not found", content_type="text/plain")
    prompt = negative = None
    rows = []
    note = None
    if filename.lower().endswith(".png"):
        try:
            from PIL import Image as _PILImage
            with _PILImage.open(path) as im:
                raw = im.text.get("prompt") if hasattr(im, "text") else None
            if raw:
                prompt, negative, rows = _shot_from_graph(json.loads(raw))
            else:
                note = "This image has no saved prompt."
        except Exception:
            note = "The saved prompt could not be read."
    else:
        note = "This image has no saved prompt."
    def block(title, body, ident):
        if not body:
            return ""
        return (f"<h2>{title} <button type='button' class='copy' data-copy='{ident}'>copy</button></h2>"
                f"<pre id='{ident}'>{_esc(body)}</pre>")
    from urllib.parse import urlencode
    ref_keys = ("Reference head", "Reference body", "Started from")
    def setting_row(k, v):
        if k in ref_keys:
            q = urlencode({"filename": v, "subfolder": "", "type": "input"})
            return f"<div class='row'><span>{_esc(k)}</span><b><a href='img?{q}' target='_blank' rel='noopener'>{_esc(v)}</a></b></div>"
        return f"<div class='row'><span>{_esc(k)}</span><b>{_esc(v)}</b></div>"
    settings = "".join(setting_row(k, v) for k, v in rows)
    ref_note = ("<p class='note'>Reference images live in ComfyUI's input/ folder — click a filename to view it.</p>"
                if any(k in ref_keys for k, _ in rows) else "")
    html = f"""<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{_esc(filename)}</title>
<style>
 :root{{color-scheme:dark}}*{{box-sizing:border-box}}
 body{{font-family:-apple-system,system-ui,Segoe UI,Roboto,sans-serif;margin:0;background:#15171c;color:#e8eaed}}
 .wrap{{max-width:860px;margin:0 auto;padding:18px}}
 img{{width:100%;border-radius:12px;display:block;background:#0e1014}}
 h1{{font-size:16px;font-weight:600;margin:14px 0 8px;word-break:break-all}}
 h2{{font-size:13px;letter-spacing:.04em;text-transform:uppercase;color:#9aa0a6;margin:18px 0 6px}}
 pre{{white-space:pre-wrap;word-break:break-word;font:15px/1.45 inherit;margin:0;background:#1d2026;padding:12px;border-radius:10px}}
 .row{{display:flex;justify-content:space-between;gap:16px;padding:8px 0;border-bottom:1px solid #2a2e36;font-size:14px}}
 .row span{{color:#9aa0a6}}.row b{{font-weight:600;text-align:right}}
 .note{{color:#9aa0a6;margin-top:14px}}
 a{{color:#9ec1ff}}
 .copy{{border:1px solid #2c2f36;background:#1e2127;color:#9ec1ff;border-radius:8px;padding:2px 10px;font-size:12px;cursor:pointer;margin-left:8px;vertical-align:middle}}
 .copy:hover{{background:#2c2f36}}
</style></head><body><div class="wrap">
 <p><a href="./">Back</a></p>
 <img src="img?type=output&filename={_esc(filename)}" alt="">
 <h1>{_esc(filename)}</h1>
 {block("Prompt", prompt, "promptText")}
 {block("Negative", negative, "negText")}
 <h2>Settings</h2>
 {settings or "<div class='note'>No settings stored with this image.</div>"}
 {ref_note}
 {f"<p class='note'>{_esc(note)}</p>" if note else ""}
</div>
<script>
document.querySelectorAll(".copy").forEach(function(b){{
  b.onclick=function(){{
    var el=document.getElementById(b.dataset.copy);
    var ok=function(){{b.textContent="copied ✓";setTimeout(function(){{b.textContent="copy";}},1500);}};
    var sel=function(){{var r=document.createRange();r.selectNodeContents(el);var s=getSelection();s.removeAllRanges();s.addRange(r);try{{if(document.execCommand("copy"))ok();else b.textContent="select + Ctrl-C";}}catch(e){{b.textContent="select + Ctrl-C";}}}};
    if(navigator.clipboard&&navigator.clipboard.writeText)navigator.clipboard.writeText(el.textContent).then(ok,sel);else sel();
  }};
}});
</script></body></html>"""
    return web.Response(text=html, content_type="text/html")


async def api_recent(request):
    # Images on disk, newest first, paginated for the recovery gallery.
    # finished server-side but its request dropped (service restart, phone tab
    # sleeping, network blip), so the image never displayed.
    try:
        per_page = max(1, min(25, int(request.query.get("per_page", request.query.get("n", 25)))))
        page_num = max(1, int(request.query.get("page", 1)))
    except ValueError:
        per_page, page_num = 25, 1
    try:
        files = [f for f in os.listdir(OUTPUT_DIR)
                 if f.lower().endswith((".png", ".jpg", ".jpeg", ".webp"))]
        files.sort(key=lambda f: os.path.getmtime(os.path.join(OUTPUT_DIR, f)), reverse=True)
        start = (page_num - 1) * per_page
        images = files[start:start + per_page]
        items = []
        for fn in images:
            path = os.path.join(OUTPUT_DIR, fn)
            mtime = os.path.getmtime(path)
            engine, model, lora = None, None, None
            if fn.lower().endswith(".png"):
                try:
                    from PIL import Image as _PILImage
                    with _PILImage.open(path) as im:
                        prompt_meta = im.text.get("prompt") if hasattr(im, "text") else None
                    if prompt_meta:
                        graph = json.loads(prompt_meta)
                        engine = _engine_label(graph)
                        model, lora = _model_lora(graph)
                except Exception:
                    engine = None
            items.append({"filename": fn, "mtime": mtime, "engine": engine or "?",
                          "model": model, "lora": lora})
        return web.json_response({"images": images, "items": items, "page": page_num,
                                  "per_page": per_page, "total": len(files),
                                  "has_more": start + per_page < len(files)})
    except OSError as e:
        return web.json_response({"images": [], "items": [], "error": str(e)}, status=500)


async def api_delete(request):
    """Permanently delete selected image files, confined to OUTPUT_DIR."""
    try:
        payload = await request.json()
        images = payload.get("images", [])
        if not isinstance(images, list) or not images or len(images) > 100:
            raise ValueError("Select between 1 and 100 images")
        deleted = []
        for filename in images:
            if (not isinstance(filename, str) or filename != os.path.basename(filename)
                    or not filename.lower().endswith((".png", ".jpg", ".jpeg", ".webp"))):
                raise ValueError("Invalid image filename")
            path = os.path.join(OUTPUT_DIR, filename)
            if os.path.isfile(path):
                os.unlink(path)
                deleted.append(filename)
        return web.json_response({"deleted": deleted})
    except (ValueError, json.JSONDecodeError) as e:
        return web.json_response({"error": str(e)}, status=400)
    except OSError as e:
        return web.json_response({"error": str(e)}, status=500)


app = web.Application(client_max_size=64 * 1024 * 1024)
app.router.add_get("/", page)
app.router.add_get("/gallery", lambda request: web.Response(text=GALLERY_PAGE, content_type="text/html"))
app.router.add_get("/shot", shot)
app.router.add_post("/api/generate", api_generate)
app.router.add_post("/api/interrupt", api_interrupt)
app.router.add_get("/api/recent", api_recent)
app.router.add_get("/api/queue", api_queue)
app.router.add_post("/api/delete", api_delete)
app.router.add_get("/img", img)

if __name__ == "__main__":
    web.run_app(app, host="0.0.0.0", port=8191)
