def test_m02_package_exists():
    import pathlib
    assert pathlib.Path('missions/M02').exists()
