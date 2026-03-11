from storyboard.manifests import build_story_manifest_from_rows, derive_panel_id_from_followings


def test_derive_panel_id_from_followings() -> None:
    row = {"followings": ["Pororo_ENGLISH1_1_Pororo_ENGLISH1_1_ep10_12"]}
    assert derive_panel_id_from_followings(row) == "Pororo_ENGLISH1_1_Pororo_ENGLISH1_1_ep10_11"


def test_build_story_manifest_from_followings_without_explicit_id() -> None:
    rows = [
        {
            "text": "panel 1",
            "followings": ["story_ep1_2", "story_ep1_3", "story_ep1_4", "story_ep1_5"],
            "image_path": "panel_1.png",
        },
        {
            "text": "panel 2",
            "followings": ["story_ep1_3", "story_ep1_4", "story_ep1_5", "story_ep1_6"],
            "image_path": "panel_2.png",
        },
        {
            "text": "panel 3",
            "followings": ["story_ep1_4", "story_ep1_5", "story_ep1_6", "story_ep1_7"],
            "image_path": "panel_3.png",
        },
        {
            "text": "panel 4",
            "followings": ["story_ep1_5", "story_ep1_6", "story_ep1_7", "story_ep1_8"],
            "image_path": "panel_4.png",
        },
        {
            "text": "panel 5",
            "followings": ["story_ep1_6", "story_ep1_7", "story_ep1_8", "story_ep1_9"],
            "image_path": "panel_5.png",
        },
    ]

    stories = build_story_manifest_from_rows(
        rows,
        dataset_name="dummy",
        source_split="train",
        num_future_panels=4,
    )

    assert len(stories) == 1
    assert stories[0]["story_id"] == "train_story_ep1_1"
    assert stories[0]["reference"]["panel_id"] == "story_ep1_1"
    assert [panel["panel_id"] for panel in stories[0]["panels"]] == [
        "story_ep1_2",
        "story_ep1_3",
        "story_ep1_4",
        "story_ep1_5",
    ]
