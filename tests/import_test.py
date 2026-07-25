def test_import() -> None:
    try:
        import multiscat_ml  # ruff: ignore[import-outside-top-level]
    except ImportError:
        multiscat_ml = None  # ty:ignore[invalid-assignment]

    assert multiscat_ml is not None, "multiscat_ml module should not be None"
