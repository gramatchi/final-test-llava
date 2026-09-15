"""Build the per-photo best-method lookup from the raw search output of
find_best_methods_{jpeg,noise}_final.py.

Output: {image_path: {severity_value: method_name}} -- just the winning
method per photo/severity, no scores. Used by build_dataset_manifest_final.py.
"""
import json
import pathlib

_HERE = pathlib.Path(__file__).parent
DATASET_DIR = _HERE.parent / "dataset"

BATCH_SUBDIR = "extra_images/0004000"


def _relative_path_for(bare_name: str) -> str:
    return f"{BATCH_SUBDIR}/{bare_name}"


def _reduce(per_image_list, value_field):
    out = {}
    for entry in per_image_list:
        path = _relative_path_for(entry["image"])
        by_value = entry[f"by_{value_field}"]
        out[path] = {v: data["best_ssim"]["method"] for v, data in by_value.items()}
    return out


def build(dtype: str, value_field: str):
    raw = json.loads((DATASET_DIR / f"{dtype}_per_image" / "per_image.json").read_text())
    lookup = _reduce(raw, value_field)
    out_path = DATASET_DIR / f"method_lookup_{dtype}.json"
    out_path.write_text(json.dumps(lookup, indent=2))
    print(f"{dtype}: {len(lookup)} photos -> {out_path}")
    return lookup


if __name__ == "__main__":
    jpeg = build("jpeg", "quality")
    noise = build("noise", "sigma")
    assert set(jpeg) == set(noise), "jpeg/noise lookups cover different photo sets!"
    print(f"\nTotal unique photos covered by both lookups: {len(jpeg)}")
