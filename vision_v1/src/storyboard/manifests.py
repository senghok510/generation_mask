from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Iterable

from storyboard.utils.common import read_json, write_json

DEFAULT_ID_CANDIDATES = (
    "id",
    "uid",
    "example_id",
    "image_id",
    "global_id",
    "story_id",
)
DEFAULT_CAPTION_CANDIDATES = ("text", "caption", "description", "prompt")
DEFAULT_IMAGE_CANDIDATES = ("image_path", "image_file", "image")


def derive_panel_id_from_followings(row: dict, followings_field: str = "followings") -> str | None:
    followings = row.get(followings_field)
    if not isinstance(followings, list) or not followings:
        return None
    first_following = str(followings[0])
    prefix, separator, suffix = first_following.rpartition("_")
    if not separator or not suffix.isdigit():
        return None
    panel_index = int(suffix) - 1
    if panel_index < 0:
        return None
    return f"{prefix}_{panel_index}"


def resolve_row_id(
    row: dict,
    *,
    id_field: str | None,
    followings_field: str = "followings",
) -> str | None:
    if id_field is not None and id_field in row:
        return str(row[id_field])
    return derive_panel_id_from_followings(row, followings_field)


def load_story_manifest(path: str | Path) -> list[dict]:
    payload = read_json(path)
    return payload["stories"]


def save_story_manifest(
    stories: list[dict],
    path: str | Path,
    *,
    dataset_name: str,
    source_split: str,
    extra: dict | None = None,
) -> None:
    payload = {
        "dataset_name": dataset_name,
        "source_split": source_split,
        "num_stories": len(stories),
        "stories": stories,
    }
    if extra:
        payload.update(extra)
    write_json(payload, path)


def infer_field_name(row: dict, candidates: Iterable[str]) -> str | None:
    for candidate in candidates:
        if candidate in row:
            return candidate
    return None


def validate_story(story: dict) -> None:
    required = ("story_id", "reference", "panels")
    missing = [field for field in required if field not in story]
    if missing:
        raise ValueError(f"Story is missing fields: {missing}")
    if not story["panels"]:
        raise ValueError(f"Story {story['story_id']} contains no target panels")


def collect_assets(stories: list[dict]) -> tuple[list[str], list[str]]:
    image_paths: list[str] = []
    captions: list[str] = []
    for story in stories:
        image_paths.append(story["reference"]["image_path"])
        if story["reference"].get("caption"):
            captions.append(story["reference"]["caption"])
        for panel in story["panels"]:
            image_paths.append(panel["image_path"])
            captions.append(panel["caption"])
    return sorted(set(image_paths)), sorted(set(captions))


def build_story_manifest_from_rows(
    rows: list[dict],
    *,
    dataset_name: str,
    source_split: str,
    num_future_panels: int = 4,
    id_field: str | None = None,
    caption_field: str | None = None,
    image_field: str | None = None,
    followings_field: str = "followings",
    story_id_field: str | None = None,
    panel_index_field: str | None = None,
) -> list[dict]:
    if not rows:
        raise ValueError("Cannot build a story manifest from an empty row set")

    sample = rows[0]
    caption_field = caption_field or infer_field_name(sample, DEFAULT_CAPTION_CANDIDATES)
    image_field = image_field or infer_field_name(sample, DEFAULT_IMAGE_CANDIDATES)

    if caption_field is None or image_field is None:
        raise ValueError("Could not infer caption/image fields from downloaded rows")

    if followings_field in sample:
        id_field = id_field or infer_field_name(sample, DEFAULT_ID_CANDIDATES)
        if id_field is None and derive_panel_id_from_followings(sample, followings_field) is None:
            raise ValueError(
                "The dataset exposes 'followings' but no resolvable id field. "
                "Pass --id-field explicitly."
            )
        rows_with_ids: list[tuple[str, dict]] = []
        for row in rows:
            resolved_id = resolve_row_id(row, id_field=id_field, followings_field=followings_field)
            if resolved_id is None:
                continue
            rows_with_ids.append((resolved_id, row))
        rows_by_id = {resolved_id: row for resolved_id, row in rows_with_ids}
        stories: list[dict] = []
        for current_id, row in rows_with_ids:
            story_ids = [current_id] + [str(item) for item in row.get(followings_field, [])]
            if len(story_ids) < num_future_panels + 1:
                continue
            resolved = [rows_by_id.get(story_id) for story_id in story_ids[: num_future_panels + 1]]
            if any(item is None for item in resolved):
                continue
            reference_row = resolved[0]
            target_rows = resolved[1:]
            reference_id = resolve_row_id(
                reference_row,
                id_field=id_field,
                followings_field=followings_field,
            )
            if reference_id is None:
                continue
            story = {
                "story_id": f"{source_split}_{reference_id}",
                "reference": {
                    "panel_id": reference_id,
                    "caption": reference_row[caption_field],
                    "image_path": reference_row[image_field],
                },
                "panels": [
                    {
                        "panel_id": resolve_row_id(
                            target_row,
                            id_field=id_field,
                            followings_field=followings_field,
                        ),
                        "caption": target_row[caption_field],
                        "image_path": target_row[image_field],
                        "index": panel_index,
                    }
                    for panel_index, target_row in enumerate(target_rows)
                ],
            }
            validate_story(story)
            stories.append(story)
        return stories

    if story_id_field and panel_index_field:
        grouped: dict[str, list[dict]] = defaultdict(list)
        for row in rows:
            grouped[str(row[story_id_field])].append(row)
        stories = []
        for story_key, story_rows in grouped.items():
            story_rows = sorted(story_rows, key=lambda item: int(item[panel_index_field]))
            if len(story_rows) < num_future_panels + 1:
                continue
            reference_row = story_rows[0]
            target_rows = story_rows[1 : num_future_panels + 1]
            story = {
                "story_id": f"{source_split}_{story_key}",
                "reference": {
                    "panel_id": str(reference_row.get("id", story_key)),
                    "caption": reference_row[caption_field],
                    "image_path": reference_row[image_field],
                },
                "panels": [
                    {
                        "panel_id": str(target_row.get("id", f"{story_key}_{panel_index + 1}")),
                        "caption": target_row[caption_field],
                        "image_path": target_row[image_field],
                        "index": panel_index,
                    }
                    for panel_index, target_row in enumerate(target_rows)
                ],
            }
            validate_story(story)
            stories.append(story)
        return stories

    raise ValueError(
        "Unable to infer a story structure. Use rows with a 'followings' field or "
        "provide --story-id-field and --panel-index-field."
    )
