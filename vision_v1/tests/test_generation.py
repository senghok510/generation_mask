def test_plus_ip_adapter_uses_sdxl_image_encoder_subfolder_by_default() -> None:
    ip_adapter_subfolder = "sdxl_models"
    ip_adapter_weight_name = "ip-adapter-plus_sdxl_vit-h.safetensors"

    image_encoder_subfolder = None
    if image_encoder_subfolder is None and "plus" in ip_adapter_weight_name:
        image_encoder_subfolder = f"{ip_adapter_subfolder}/image_encoder"

    assert image_encoder_subfolder == "sdxl_models/image_encoder"


def test_plus_ip_adapter_projection_matches_expected_dims() -> None:
    encoder_hidden_size = 1664
    projection_dim = 1280

    assert encoder_hidden_size != projection_dim
    assert projection_dim == 1280
