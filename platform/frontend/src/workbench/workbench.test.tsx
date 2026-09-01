import { readFileSync, readdirSync, statSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { render, screen, waitFor } from '@testing-library/react';
import { MemoryRouter } from 'react-router-dom';
import { beforeEach, describe, expect, it, vi } from 'vitest';
import { AppRoutes } from '../App';
import { setAuthToken } from '../api/client';
import { AuthProvider } from '../context/AuthContext';
import { jsonResponse } from '../test/http';
import { parseStructuredResult } from './payload';
import { STRUCTURED_RESULT_BLOCK_TYPES } from './types';
import { Workbench } from './Workbench';

const DIR = path.dirname(fileURLToPath(import.meta.url));
const FIXTURE_PATH = path.join(DIR, 'fixtures', 'structured-result.json');
const fixture = parseStructuredResult(JSON.parse(readFileSync(FIXTURE_PATH, 'utf8')));

function walk(dir: string, acc: string[] = []): string[] {
  for (const name of readdirSync(dir)) {
    if (name === 'node_modules' || name === 'dist') {
      continue;
    }
    const full = path.join(dir, name);
    if (statSync(full).isDirectory()) {
      walk(full, acc);
      continue;
    }
    if (!/\.(ts|tsx|js|jsx|css|json)$/.test(name)) {
      continue;
    }
    if (/\.test\.(ts|tsx)$/.test(name)) {
      continue;
    }
    acc.push(full);
  }
  return acc;
}

function installApiMock() {
  vi.stubGlobal(
    'fetch',
    vi.fn(async (input: RequestInfo | URL) => {
      const url = String(input);
      if (url.includes('/auth/bootstrap')) {
        return jsonResponse({ token: 'loopback-token', token_type: 'bearer' });
      }
      if (url.includes('/learners/')) {
        return jsonResponse({ id: 'learner-1', username: 'ada', display_name: 'Ada' });
      }
      return jsonResponse({ error: { code: 'NOT_FOUND', message: 'missing', details: {} } }, 404);
    }),
  );
}

describe('workbench renderers', () => {
  it('renders a table block and a markdown block from fixture JSON', () => {
    render(<Workbench result={fixture} />);

    const table = screen.getByTestId('table-block');
    expect(table).toBeInTheDocument();
    expect(table).toHaveTextContent('id');
    expect(table).toHaveTextContent('val');
    expect(table).toHaveTextContent('2.5');
    expect(table).toHaveTextContent('3.8');
    expect(screen.getByRole('columnheader', { name: 'id' })).toBeInTheDocument();

    const markdown = screen.getByTestId('markdown-block');
    expect(markdown).toBeInTheDocument();
    expect(markdown).toHaveTextContent('Shape');
    expect(markdown).toHaveTextContent('Two rows survived cleaning.');
    expect(markdown).toHaveTextContent('no missing values');
  });

  it('maps every WP-137 block type from the fixture', () => {
    render(<Workbench result={fixture} />);
    for (const type of STRUCTURED_RESULT_BLOCK_TYPES) {
      expect(screen.getByTestId(`block-${type}`)).toHaveAttribute('data-block-type', type);
    }
  });

  it('renders a generic workspace tree from file props', () => {
    render(
      <Workbench
        files={[
          { path: 'src/main.py', kind: 'file' },
          { path: 'data/sample.csv', kind: 'file' },
        ]}
      />,
    );
    const tree = screen.getByTestId('workspace-tree');
    expect(tree).toHaveTextContent('main.py');
    expect(tree).toHaveTextContent('sample.csv');
    expect(tree).not.toHaveTextContent('missions');
  });

  it('does not special-case demo routes or flagship mission ids', () => {
    const forbidden = ['/missions/demo', 'F01', 'M01', 'f01'];
    const hits: string[] = [];
    for (const file of walk(DIR)) {
      const text = readFileSync(file, 'utf8');
      const relative = path.relative(DIR, file);
      for (const needle of forbidden) {
        if (text.includes(needle)) {
          hits.push(`${relative}: ${needle}`);
        }
      }
    }
    expect(hits).toEqual([]);
  });
});

describe('workbench surface', () => {
  beforeEach(() => {
    setAuthToken(null);
    sessionStorage.setItem(
      'learningos.learner',
      JSON.stringify({ id: 'learner-1', username: 'ada', display_name: 'Ada' }),
    );
    installApiMock();
  });

  it('replaces the unavailable workbench surface', async () => {
    render(
      <AuthProvider>
        <MemoryRouter initialEntries={['/workbench']}>
          <AppRoutes />
        </MemoryRouter>
      </AuthProvider>,
    );

    expect(await screen.findByTestId('workbench')).toBeInTheDocument();
    expect(screen.getByRole('heading', { name: 'Workbench' })).toBeInTheDocument();
    expect(screen.getByTestId('workbench-code-editor')).toBeInTheDocument();
    expect(screen.queryByText(/not available in G3/i)).not.toBeInTheDocument();
    await waitFor(() => {
      expect(document.body.innerHTML).not.toContain('/missions/demo');
    });
  });
});
