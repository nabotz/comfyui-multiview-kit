"""Multi-View Kit: helpers for a one-image -> Qwen-Image edit views -> Pixal3D multi-view pipeline.

MultiViewPrompts       builds the side / front / back edit prompts from one subject description.
MultiViewScaleNormalize puts every view on the SAME scale (the bundled ImageCropToMask fits each view
                        to its own box, so a long creature's front view ends up far bigger than its side
                        view and Pixal3D grows duplicate features). It also mirrors the side view to make
                        the opposite side, so both flanks match exactly.
"""

import torch
import torch.nn.functional as F

VIEW_ORDER = ("front", "left", "back", "right")

DEFAULT_SIDE = (
    "Redraw this exact {subject} as a clean character reference image for 3D modeling. "
    "Strict side view in profile facing left, standing still in a neutral pose with the limbs straight "
    "and relaxed and the head held level. {details} "
    "Keep the same head shape, face, colors, materials, markings and body proportions as the reference. "
    "Full body completely inside the frame with generous margin on all sides, camera at body height, "
    "orthographic look, soft even studio lighting, no cast shadows, plain flat light grey background, no ground."
)
DEFAULT_FRONT = (
    "Turn the camera around this same {subject} so it is seen from the front: a strict front view, the {subject} "
    "faces the camera head-on, perfectly symmetric and centered in the image, with the face in the middle and both "
    "eyes visible. {details} "
    "Same neutral standing pose, same colors, same materials and same body proportions as the reference, and the "
    "same height in the frame. This is a brand-new image that contains only this single front view: nothing of the "
    "original side view remains, no faint silhouette, no ghost image, no second {subject}. "
    "Full body completely inside the frame with generous margin, camera at body height, orthographic look, "
    "soft even studio lighting, no cast shadows, perfectly clean uniform flat light grey background, no ground."
)
DEFAULT_BACK = (
    "Turn the camera around this same {subject} so it is seen from behind: a strict rear view, the camera is "
    "directly behind the {subject} looking straight at its back, perfectly symmetric and centered in the image, "
    "the face is not visible. {details} "
    "Same neutral standing pose, same colors, same materials and same body proportions as the reference, and the "
    "same height in the frame. This is a brand-new image that contains only this single rear view: nothing of the "
    "original side view remains, no faint silhouette, no ghost image, no second {subject}. "
    "Full body completely inside the frame with generous margin, camera at body height, orthographic look, "
    "soft even studio lighting, no cast shadows, perfectly clean uniform flat light grey background, no ground."
)


def _fill(template, subject, details):
    text = template.replace("{subject}", subject.strip()).replace("{details}", details.strip())
    while "  " in text:
        text = text.replace("  ", " ")
    return text.replace(" .", ".").strip()


class MultiViewPrompts:
    """Side / front / back edit prompts for Qwen-Image edit, from one subject description."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "subject": ("STRING", {"default": "creature", "multiline": False,
                                       "tooltip": "Short name of the character, e.g. 'shark-horse creature' or 'knight in plate armor'. Replaces {subject}."}),
                "details": ("STRING", {"default": "", "multiline": True,
                                       "tooltip": "Optional extra description (key features, colors, anatomy) added to every prompt. Replaces {details}."}),
                "side_template": ("STRING", {"default": DEFAULT_SIDE, "multiline": True}),
                "front_template": ("STRING", {"default": DEFAULT_FRONT, "multiline": True}),
                "back_template": ("STRING", {"default": DEFAULT_BACK, "multiline": True}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("side_prompt", "front_prompt", "back_prompt")
    FUNCTION = "build"
    CATEGORY = "multiview"
    DESCRIPTION = "Builds the side (facing left), front and back edit prompts used to turn one image into a turnaround."

    def build(self, subject, details, side_template, front_template, back_template):
        return (_fill(side_template, subject, details),
                _fill(front_template, subject, details),
                _fill(back_template, subject, details))


def _drop_specks(m, binary, min_fraction):
    """Zero out mask islands smaller than min_fraction of the largest one (stray flecks the cutout
    model keeps would otherwise stretch the bounding box and push the subject off-center)."""
    if min_fraction <= 0:
        return m, binary
    try:
        from scipy import ndimage
    except ImportError:
        return m, binary
    grown = ndimage.binary_dilation(binary.numpy(), iterations=2)
    labels, count = ndimage.label(grown)
    if count <= 1:
        return m, binary
    sizes = ndimage.sum(binary.numpy(), labels, index=range(1, count + 1))
    keep_ids = [k + 1 for k, s in enumerate(sizes) if s >= min_fraction * sizes.max()]
    keep = torch.from_numpy(__import__("numpy").isin(labels, keep_ids))
    return m * keep, binary & keep


def _prepare(image, mask, threshold, min_fraction=0.0):
    """image [B,H,W,C], mask [B,H,W] or [H,W] -> (rgb*mask cropped to bbox [h,w,3], box h, box w)."""
    img = image[0, ..., :3].float().cpu()
    m = mask
    while m.dim() > 3:
        m = m[:, 0] if m.shape[1] == 1 else m[..., 0]
    if m.dim() == 2:
        m = m[None]
    m = m[0].float().cpu()
    if m.shape != img.shape[:2]:
        m = F.interpolate(m[None, None], size=img.shape[:2], mode="bilinear", align_corners=False)[0, 0]
    m = m.clamp(0, 1)
    m, binary = _drop_specks(m, m > threshold, min_fraction)
    ys, xs = torch.nonzero(binary, as_tuple=True)
    if ys.numel() == 0:
        raise ValueError("MultiViewScaleNormalize: a mask is empty; check the background removal for that view")
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    rgb = img[y0:y1, x0:x1] * m[y0:y1, x0:x1, None]
    return rgb, y1 - y0, x1 - x0


def _resize(rgb, h, w):
    t = rgb.permute(2, 0, 1)[None]
    t = F.interpolate(t, size=(max(1, h), max(1, w)), mode="bicubic", align_corners=False, antialias=True)
    return t[0].permute(1, 2, 0).clamp(0, 1)


def _place(rgb, scale, size):
    h, w = rgb.shape[:2]
    nh, nw = int(round(h * scale)), int(round(w * scale))
    canvas = torch.zeros(size, size, 3)
    piece = _resize(rgb, nh, nw)
    nh, nw = min(nh, size), min(nw, size)
    top, left = (size - nh) // 2, (size - nw) // 2
    canvas[top:top + nh, left:left + nw] = piece[:nh, :nw]
    return canvas[None]


class MultiViewScaleNormalize:
    """Shared-scale framing of a side / front / back turnaround for Pixal3DMultiViewConditioning."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "side_image": ("IMAGE",),
                "side_faces": (["left", "right"], {"default": "left",
                                                   "tooltip": "Which way the character faces in the side image. Facing image-left goes to Pixal3D's 'left' slot."}),
                "mirror_other_side": ("BOOLEAN", {"default": True,
                                                  "tooltip": "Make the opposite side by mirroring the side view (both flanks match exactly). Off: use other_side_image if connected."}),
                "use_front": ("BOOLEAN", {"default": True,
                                          "tooltip": "Turn off when the generated front view is inconsistent (double eyes, extra face); Pixal3D then poses the mesh from the next view."}),
                "use_back": ("BOOLEAN", {"default": True}),
                "size": ("INT", {"default": 1024, "min": 256, "max": 2048, "step": 64}),
                "pad": ("FLOAT", {"default": 1.1, "min": 1.0, "max": 2.0, "step": 0.01,
                                  "tooltip": "Frame size / object size at its widest. Pixal3D's rig expects 1.1."}),
                "mask_threshold": ("FLOAT", {"default": 0.5, "min": 0.01, "max": 0.99, "step": 0.01}),
                "drop_specks": ("FLOAT", {"default": 0.02, "min": 0.0, "max": 0.5, "step": 0.01,
                                          "tooltip": "Remove cutout islands smaller than this fraction of the main subject (stray flecks shift the framing). 0 = keep everything."}),
            },
            "optional": {
                "bg_removal_model": ("BACKGROUND_REMOVAL", {"tooltip": "Cuts out every view that has no mask connected (Load Background Removal Model)."}),
                "side_mask": ("MASK",),
                "front_image": ("IMAGE",),
                "front_mask": ("MASK",),
                "back_image": ("IMAGE",),
                "back_mask": ("MASK",),
                "other_side_image": ("IMAGE",),
                "other_side_mask": ("MASK",),
            },
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "IMAGE", "IMAGE", "IMAGE")
    RETURN_NAMES = ("front", "left", "back", "right", "preview")
    FUNCTION = "normalize"
    CATEGORY = "multiview"
    DESCRIPTION = ("Scales every view to the side view's height, frames them on one square canvas size "
                   "(object = 1/pad of the frame at its widest, black background) and mirrors the side view "
                   "for the opposite side. Disabled or missing views output nothing, so Pixal3D skips them.")

    def normalize(self, side_image, side_faces, mirror_other_side, use_front, use_back, size, pad,
                  mask_threshold, drop_specks=0.02, bg_removal_model=None, side_mask=None, front_image=None,
                  front_mask=None, back_image=None, back_mask=None, other_side_image=None, other_side_mask=None):
        def need_mask(image, mask, name):
            if image is None or mask is not None:
                return mask
            if bg_removal_model is None:
                raise ValueError(f"MultiViewScaleNormalize: connect {name}_mask or a bg_removal_model")
            return bg_removal_model.encode_image(image)

        side_mask = need_mask(side_image, side_mask, "side")
        if use_front:
            front_mask = need_mask(front_image, front_mask, "front")
        if use_back:
            back_mask = need_mask(back_image, back_mask, "back")
        if not mirror_other_side:
            other_side_mask = need_mask(other_side_image, other_side_mask, "other_side")
        pieces = {}
        side_rgb, ref_h, _ = _prepare(side_image, side_mask, mask_threshold, drop_specks)
        side_slot = "left" if side_faces == "left" else "right"
        other_slot = "right" if side_slot == "left" else "left"
        pieces[side_slot] = side_rgb
        if mirror_other_side:
            pieces[other_slot] = torch.flip(side_rgb, dims=[1])
        elif other_side_image is not None and other_side_mask is not None:
            pieces[other_slot] = _prepare(other_side_image, other_side_mask, mask_threshold, drop_specks)[0]
        if use_front and front_image is not None and front_mask is not None:
            pieces["front"] = _prepare(front_image, front_mask, mask_threshold, drop_specks)[0]
        if use_back and back_image is not None and back_mask is not None:
            pieces["back"] = _prepare(back_image, back_mask, mask_threshold, drop_specks)[0]

        # every view to the side view's height, then one frame that fits the widest of them
        ks = {name: ref_h / rgb.shape[0] for name, rgb in pieces.items()}
        extent = max([ref_h] + [rgb.shape[1] * ks[name] for name, rgb in pieces.items()])
        frame = pad * extent
        out = {name: _place(rgb, ks[name] * size / frame, size) for name, rgb in pieces.items()}

        tiles = [out.get(name, torch.zeros(1, size, size, 3)) for name in VIEW_ORDER]
        preview = _resize(torch.cat(tiles, dim=2)[0], size // 2, size * 2)[None]
        return tuple(out.get(name) for name in VIEW_ORDER) + (preview,)


def _save_png(img, path):
    from PIL import Image
    import numpy as np
    arr = (img[0, ..., :3].clamp(0, 1).cpu().numpy() * 255.0 + 0.5).astype(np.uint8)
    Image.fromarray(arr).save(path, compress_level=4)


def _load_png(path):
    from PIL import Image
    import numpy as np
    arr = np.asarray(Image.open(path).convert("RGB")).astype(np.float32) / 255.0
    return torch.from_numpy(arr)[None]


class MultiViewViews:
    """Generate the turnaround views once, then reuse them: in 'reuse' mode the upstream Qwen branch never runs."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mode": (["generate", "reuse saved"], {"default": "generate",
                         "tooltip": "generate: run Qwen and save the views. reuse saved: load the last saved views and skip Qwen entirely (fast re-runs of the 3D part)."}),
                "folder": ("STRING", {"default": "mv3d/latest", "tooltip": "Subfolder of ComfyUI output where side.png / front.png / back.png are kept."}),
                "include_front": ("BOOLEAN", {"default": True, "tooltip": "Off: the front view is neither generated nor used (long snouts often get a wrong front face)."}),
                "include_back": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "side": ("IMAGE", {"lazy": True}),
                "front": ("IMAGE", {"lazy": True}),
                "back": ("IMAGE", {"lazy": True}),
            },
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "IMAGE")
    RETURN_NAMES = ("side", "front", "back")
    FUNCTION = "run"
    CATEGORY = "multiview"
    DESCRIPTION = ("Switch between generating the side / front / back views and reusing the saved ones. "
                   "Disabled views are not generated and come out empty, so the normalizer skips them.")

    @staticmethod
    def _wanted(include_front, include_back):
        return ["side"] + (["front"] if include_front else []) + (["back"] if include_back else [])

    def check_lazy_status(self, mode, folder, include_front, include_back, side=None, front=None, back=None):
        if mode != "generate":
            return []
        have = {"side": side, "front": front, "back": back}
        return [name for name in self._wanted(include_front, include_back) if have[name] is None]

    def run(self, mode, folder, include_front, include_back, side=None, front=None, back=None):
        import os
        import folder_paths
        root = os.path.abspath(folder_paths.get_output_directory())
        out_dir = os.path.abspath(os.path.join(root, folder.strip().strip("/\\") or "mv3d/latest"))
        if os.path.commonpath([root, out_dir]) != root:
            raise ValueError("MultiViewViews: folder must stay inside the ComfyUI output directory")
        wanted = self._wanted(include_front, include_back)
        given = {"side": side, "front": front, "back": back}
        result = {"side": None, "front": None, "back": None}
        if mode == "generate":
            os.makedirs(out_dir, exist_ok=True)
            for name in wanted:
                if given[name] is None:
                    raise ValueError(f"MultiViewViews: '{name}' input is not connected")
                _save_png(given[name], os.path.join(out_dir, f"{name}.png"))
                result[name] = given[name]
        else:
            for name in wanted:
                path = os.path.join(out_dir, f"{name}.png")
                if not os.path.isfile(path):
                    raise ValueError(f"MultiViewViews: {path} not found. Run once in 'generate' mode (with {name} included) first.")
                result[name] = _load_png(path)
        sub = os.path.relpath(out_dir, folder_paths.get_output_directory())
        ui = [{"filename": f"{n}.png", "subfolder": sub, "type": "output"} for n in wanted]
        return {"ui": {"images": ui}, "result": (result["side"], result["front"], result["back"])}


NODE_CLASS_MAPPINGS = {
    "MultiViewPrompts": MultiViewPrompts,
    "MultiViewScaleNormalize": MultiViewScaleNormalize,
    "MultiViewViews": MultiViewViews,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MultiViewPrompts": "Multi-View Prompts (Qwen edit)",
    "MultiViewScaleNormalize": "Multi-View Scale Normalize (Pixal3D)",
    "MultiViewViews": "Multi-View Views (generate / reuse)",
}
