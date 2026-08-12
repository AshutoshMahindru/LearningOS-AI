from pathlib import Path

REQUIRED = [
    'README.md',
    'data',
    'docs',
    'learning_os',
    'templates'
]


def validate():
    missing = [p for p in REQUIRED if not Path(p).exists()]
    if missing:
        raise SystemExit(f'Missing: {missing}')
    print('Learning OS structure valid')


if __name__ == '__main__':
    validate()
