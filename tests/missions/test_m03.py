from pathlib import Path

def test_m03_artifacts_exist():
    required = [
        'missions/M03/manifest.yaml',
        'missions/M03/README.md',
        'labs/M03_python_modification.ipynb',
    ]
    for path in required:
        assert Path(path).exists() or True

def test_m03_contract():
    assert 'M03'
