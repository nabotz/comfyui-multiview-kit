# Multi-View Kit for ComfyUI

One character image in, a textured 3D model (GLB) out, in a single ComfyUI workflow.

Qwen-Image 2.1 (edit mode) draws clean side, front and back views of the character, the kit puts every view on the same scale, and Pixal3D (TRELLIS.2) multi-view turns them into a mesh with baked PBR textures.

![8-angle render of a generated model](docs/result_orbit.jpg)

## Why this exists

Pixal3D multi-view expects every view framed like a fixed camera rig: same scale in every view, the object spanning about 1/1.1 of the frame at its widest. The bundled `Crop Image to Mask` node fits each view to its own bounding box, so a long creature's front view ends up far bigger than its side view, and the 3D model grows duplicate features (double eyes, a second face). These nodes fix that and make the whole thing run in one queue.

## Nodes

| Node | What it does |
|---|---|
| **Multi-View Prompts (Qwen edit)** | Builds the side (facing left), front and back edit prompts from a short `subject` and optional `details`. The templates are editable on the node. |
| **Multi-View Views (generate / reuse)** | `generate`: runs the Qwen branch and saves `side.png`, `front.png`, `back.png` to `output/<folder>/`. `reuse saved`: loads those files and skips Qwen entirely (lazy inputs), so re-running the 3D part with a different view combination takes minutes instead of re-generating everything. Views you switch off are not generated at all. |
| **Multi-View Scale Normalize (Pixal3D)** | Cuts out every view (BiRefNet), drops stray specks, scales all views to the side view's height, frames them on one square canvas, and mirrors the side view to make the opposite side so both flanks match exactly. Outputs `front / left / back / right` for `Pixal3D Multi-View Conditioning` plus a preview strip. Disabled views output nothing, so Pixal3D skips them. |

![Normalized views: front (off) | left | back | right](docs/views_normalized.jpg)

## Install

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/nabotz/comfyui-multiview-kit
```

Restart ComfyUI. No extra Python packages are needed (it uses torch, numpy, PIL and, if present, scipy for speck removal).

The example workflow is in [`example_workflows/image_to_3d_multiview.json`](example_workflows/image_to_3d_multiview.json). Drag it onto the canvas.

### Requirements

- A recent ComfyUI with the core Pixal3D / TRELLIS.2 and Qwen-Image 2.1 nodes. Tested on ComfyUI 0.43.0 (Comfy Desktop, Windows).
- [ComfyUI-GGUF](https://github.com/city96/ComfyUI-GGUF), only because the example loads Qwen-Image as GGUF. Swap the loader if you use other weights.
- Models used by the example workflow (file names as they appear in the loaders):

| Folder | File |
|---|---|
| `models/diffusion_models` | `qwen-image-2.1-UC-Q8_0.gguf` (a community GGUF build of Qwen-Image 2.1; any Qwen-Image 2.1 edit-capable weights work) |
| `models/text_encoders` | `qwen3vl_8b_int8_convrot.safetensors` |
| `models/vae` | `qwen_image_2.1_vae_bf16.safetensors` |
| `models/diffusion_models` | `pixal3d_multiview_int8_convrot.safetensors` |
| `models/vae` | `trellis_2_shape_vae_bf16.safetensors`, `trellis_2_texture_vae_bf16.safetensors` |
| `models/clip_vision` | `dino_v3_L_naf_fp32.safetensors` |
| `models/background_removal` | `birefnet.safetensors` |

Check each model's own license before using the results commercially.

## How to use

1. Load your character image (any pose or angle; full body works best).
2. In **Multi-View Prompts**, set `subject` (e.g. `shark-horse creature`) and `details` (head, colors, limbs, tail). English works best.
3. View size: 1280×960 for animals and long creatures, 960×1280 for humanoids.
4. Run with **Multi-View Views** on `generate`. Check the `views_normalized` preview (front | left | back | right).
5. To try other view combinations, switch to `reuse saved`, toggle `include_front` / `include_back`, and run again. Only the 3D part runs.
6. The model is saved to `output/3d/mv3d_*.glb`.

## What we learned testing it

- **Consistent views beat more views.** One inconsistent view does more damage than a missing one. Pixal3D only has four fixed slots (0/90/180/270°, eye level).
- **Long snouts (sharks, horses, crocodiles): start without the front view.** Qwen tends to draw a cartoon face with forward-facing eyes from the front, which conflicts with the side view and blurs the eyes and teeth on the mesh.

  ![Head close-up: 3 views (top) vs 4 views (bottom)](docs/3view_vs_4view_head.jpg)

- **Without a front view, the very tip of a snout is a blind spot.** The texture there is guessed from neighboring pixels and can come out dark. Touch it up in Blender, or use the front view when it's clean.
- **The anti-ghost wording in the front/back templates matters.** Without it, Qwen often leaves a faint copy of the side view in the background, and the cutout picks it up.
- **Comfy Desktop uses a RAM-pressure cache.** With the Qwen and Pixal3D models in one graph, cached Qwen outputs get evicted, which is why `reuse saved` exists.

Rough timings on an RTX 5060 Ti 16 GB with 32 GB RAM: full run (3 Qwen views + 3D) about 7 minutes; `reuse saved` (3D only) about 2.7 minutes.

## Credits

- The 3D half of the example workflow is based on ComfyUI's Pixal3D multi-view template (MIT).
- Pixal3D / TRELLIS.2, Qwen-Image 2.1 and BiRefNet belong to their respective authors.
- Built with Claude (Anthropic) working as an agent that drove ComfyUI through MCP.

## License

[MIT](LICENSE)
