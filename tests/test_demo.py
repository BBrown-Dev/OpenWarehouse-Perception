import sys

sys.path.insert(0, "scripts")

from demo import siglip_prompt, slug  # noqa: E402


def test_siglip_prompt_matches_frozen_photo_wording():
    assert siglip_prompt("box") == "a photo of a box"
    assert siglip_prompt("a damaged box") == "a photo of a damaged box"
    assert siglip_prompt("the shelf") == "a photo of the shelf"


def test_slug_is_filename_safe():
    assert slug('Damaged box / "fallen"') == "damaged-box-fallen"
