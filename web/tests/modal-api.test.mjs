import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { createRequire } from 'node:module';
import test from 'node:test';
import ts from 'typescript';

const require = createRequire(import.meta.url);
const wranglerRequire = createRequire(require.resolve('wrangler/package.json'));
const { Miniflare } = wranglerRequire('miniflare');
const source = readFileSync(new URL('../lib/modal-api.ts', import.meta.url), 'utf8')
  .replace('import "server-only";', '')
  .replace('import { getCloudflareContext } from "@opennextjs/cloudflare";', `
    async function getCloudflareContext() {
      return { env: {
        MODAL_ENDPOINT: 'https://backend.test',
        MODAL_PROXY_TOKEN_ID: 'test-id',
        MODAL_PROXY_TOKEN_SECRET: 'test-secret',
      } };
    }
  `);
const proxy = ts.transpileModule(source, {
  compilerOptions: { target: ts.ScriptTarget.ES2022, module: ts.ModuleKind.ES2022 },
}).outputText;

// Isolated proxy unit tests, not the Next.js app. No real secrets or GPU calls.
// This date is supported by the project's installed workerd binary.
async function withProxy(upstream, check) {
  const requests = [];
  const mf = new Miniflare({
    modules: true,
    compatibilityDate: '2026-07-21',
    compatibilityFlags: ['nodejs_compat', 'global_fetch_strictly_public'],
    script: `${proxy}\nexport default { fetch(request) {
      return proxyModal('/api/health', { signal: request.signal });
    } };`,
    outboundService: async (request) => {
      requests.push(request);
      return upstream(request);
    },
  });
  try {
    const response = await mf.dispatchFetch('https://proxy.test');
    await check(response, requests);
  } finally {
    await mf.dispose();
  }
}

test('Worker proxy can reach a healthy JSON backend with authentication', async () => {
  await withProxy(() => Response.json({ status: 'online' }), async (response, requests) => {
    assert.equal(response.status, 200);
    assert.deepEqual(await response.json(), { status: 'online' });
    assert.equal(requests.length, 1);
    assert.equal(requests[0].headers.get('Modal-Key'), 'test-id');
    assert.equal(requests[0].headers.get('Modal-Secret'), 'test-secret');
    assert.equal(response.headers.get('cache-control'), 'no-store');
  });
});

test('Worker proxy preserves pending results and polling hints', async () => {
  await withProxy(() => Response.json({ status: 'running' }, {
    status: 202, headers: { 'retry-after': '2' },
  }), async (response) => {
    assert.equal(response.status, 202);
    assert.equal(response.headers.get('retry-after'), '2');
    assert.deepEqual(await response.json(), { status: 'running' });
  });
});

test('Worker proxy blocks redirects without forwarding credentials to another host', async () => {
  await withProxy(() => Response.redirect('https://untrusted.test', 307), async (response, requests) => {
    assert.equal(response.status, 503);
    assert.equal(requests.length, 1);
    assert.equal(new URL(requests[0].url).hostname, 'backend.test');
    assert.match((await response.json()).message, /redirect/i);
  });
});

test('Worker proxy keeps expired-job responses intact', async () => {
  await withProxy(() => Response.json({ message: 'This job has expired.' }, { status: 404 }), async (response) => {
    assert.equal(response.status, 404);
    assert.deepEqual(await response.json(), { message: 'This job has expired.' });
  });
});
