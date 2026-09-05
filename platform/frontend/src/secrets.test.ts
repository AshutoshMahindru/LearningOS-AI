import { readdirSync, readFileSync, statSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { describe, expect, it } from 'vitest';

const SRC_ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)));
const FRONTEND_ROOT = path.resolve(SRC_ROOT, '..');

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

describe('secret isolation', () => {
  it('does not embed provider secrets or persist api keys in source', () => {
    const openai = ['OPENAI', 'API', 'KEY'].join('_');
    const anthropic = ['ANTHROPIC', 'API', 'KEY'].join('_');
    const sk = `sk${'-'}`;
    const apiKey = `api${'Key'}`;
    const vitePrefix = `VITE${'_'}`;
    const importMetaEnv = ['import', 'meta', 'env'].join('.');
    const files = walk(FRONTEND_ROOT).filter((file) => !file.endsWith('package-lock.json'));
    const hits: string[] = [];

    for (const file of files) {
      const text = readFileSync(file, 'utf8');
      const relative = path.relative(FRONTEND_ROOT, file);
      if (text.includes(openai)) {
        hits.push(`${relative}: contains ${openai}`);
      }
      if (text.includes(anthropic)) {
        hits.push(`${relative}: contains ${anthropic}`);
      }
      if (text.includes(sk)) {
        hits.push(`${relative}: contains ${sk}`);
      }
      if (text.includes(vitePrefix)) {
        hits.push(`${relative}: contains ${vitePrefix} env binding`);
      }
      if (text.includes(importMetaEnv) && text.includes(vitePrefix)) {
        hits.push(`${relative}: contains ${importMetaEnv} ${vitePrefix} binding`);
      }
      const persist = new RegExp(
        `(localStorage|sessionStorage)\\.(setItem|getItem)\\([^\\n]*${apiKey}|${apiKey}[^\\n]*(localStorage|sessionStorage)`,
        'i',
      );
      if (persist.test(text)) {
        hits.push(`${relative}: ${apiKey} persistence`);
      }
    }

    expect(hits).toEqual([]);
  });
});
