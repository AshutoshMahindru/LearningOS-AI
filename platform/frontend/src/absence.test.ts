import { readdirSync, readFileSync, statSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { describe, expect, it } from 'vitest';

const SRC_ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)));

function walk(dir: string, acc: string[] = []): string[] {
  for (const name of readdirSync(dir)) {
    if (name === 'node_modules' || name === 'dist' || name === '.git') {
      continue;
    }
    const full = path.join(dir, name);
    if (statSync(full).isDirectory()) {
      walk(full, acc);
      continue;
    }
    if (!/\.(ts|tsx|js|jsx|css|html|json)$/.test(name)) {
      continue;
    }
    if (/\.test\.(ts|tsx)$/.test(name) || name === 'setup.ts' || name === 'http.ts') {
      continue;
    }
    acc.push(full);
  }
  return acc;
}

describe('generic frontend constitution', () => {
  it('does not special-case demo routes or flagship/fixture mission ids in src', () => {
    const forbidden = [
      '/missions/demo',
      'F01',
      'M01',
      'f01',
    ];
    const hits: string[] = [];

    for (const file of walk(SRC_ROOT)) {
      const text = readFileSync(file, 'utf8');
      const relative = path.relative(SRC_ROOT, file);
      for (const needle of forbidden) {
        if (text.includes(needle)) {
          hits.push(`${relative}: ${needle}`);
        }
      }
    }

    expect(hits).toEqual([]);
  });
});
