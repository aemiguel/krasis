#!/usr/bin/env bun
/**
 * Krasis Session Bridge — connects a Session messenger identity to a
 * running Krasis HTTP server. Incoming messages are forwarded as
 * OpenAI-compatible chat completions; streamed responses are sent back
 * as Session messages.
 *
 * Identity (mnemonic) is stored in ~/.krasis/session_identity.
 *
 * Usage:
 *   node session-bridge.mjs [--port 8012] [--host 127.0.0.1]
 *
 * Environment:
 *   KRASIS_SESSION_IDENTITY  path to identity file (default ~/.krasis/session_identity)
 *   KRASIS_HOST              server host (default 127.0.0.1)
 *   KRASIS_PORT              server port (default 8012)
 */

import { Session, Poller, ready } from '@session.js/client';
import { generateSeedHex } from '@session.js/keypair';
import { encode as encodeMnemonic } from '@session.js/mnemonic';
import fs from 'node:fs';
import path from 'node:path';
import os from 'node:os';

// ── Config ──────────────────────────────────────────────────────────

const args = process.argv.slice(2);
function getArg(name, fallback) {
  const idx = args.indexOf(name);
  return idx >= 0 && idx + 1 < args.length ? args[idx + 1] : fallback;
}

const KRASIS_HOST = getArg('--host', process.env.KRASIS_HOST || '127.0.0.1');
const KRASIS_PORT = parseInt(getArg('--port', process.env.KRASIS_PORT || '8012'), 10);
const IDENTITY_PATH = getArg('--identity',
  process.env.KRASIS_SESSION_IDENTITY ||
  path.join(os.homedir(), '.krasis', 'session_identity'));

const KRASIS_URL = `http://${KRASIS_HOST}:${KRASIS_PORT}/v1/chat/completions`;

// ── Identity management ─────────────────────────────────────────────

function loadOrCreateIdentity() {
  const dir = path.dirname(IDENTITY_PATH);
  if (!fs.existsSync(dir)) {
    fs.mkdirSync(dir, { recursive: true });
  }

  if (fs.existsSync(IDENTITY_PATH)) {
    const mnemonic = fs.readFileSync(IDENTITY_PATH, 'utf-8').trim();
    if (mnemonic) return mnemonic;
  }

  // Generate new identity
  const seed = generateSeedHex();
  const mnemonic = encodeMnemonic(seed);
  fs.writeFileSync(IDENTITY_PATH, mnemonic + '\n', { mode: 0o600 });
  console.log(`[session-bridge] Created new Session identity: ${IDENTITY_PATH}`);
  return mnemonic;
}

// ── Per-conversation history ────────────────────────────────────────

const conversations = new Map(); // sessionID -> [{role, content}, ...]

function getHistory(from) {
  if (!conversations.has(from)) {
    conversations.set(from, []);
  }
  return conversations.get(from);
}

// ── Stream a chat completion from Krasis ────────────────────────────

async function streamCompletion(messages) {
  const body = JSON.stringify({
    model: 'krasis',
    messages,
    stream: true,
  });

  const res = await fetch(KRASIS_URL, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body,
  });

  if (!res.ok) {
    const text = await res.text();
    throw new Error(`Krasis HTTP ${res.status}: ${text}`);
  }

  // Read SSE stream, collect full response text
  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let full = '';
  let buf = '';

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buf += decoder.decode(value, { stream: true });

    // Process complete SSE lines
    const lines = buf.split('\n');
    buf = lines.pop(); // keep incomplete line in buffer

    for (const line of lines) {
      if (!line.startsWith('data: ')) continue;
      const data = line.slice(6).trim();
      if (data === '[DONE]') continue;
      try {
        const parsed = JSON.parse(data);
        const delta = parsed.choices?.[0]?.delta?.content;
        if (delta) full += delta;
      } catch {
        // skip malformed chunks
      }
    }
  }

  return full;
}

// ── Main ────────────────────────────────────────────────────────────

async function main() {
  await ready;

  const mnemonic = loadOrCreateIdentity();

  const session = new Session();
  session.setMnemonic(mnemonic, 'Krasis');

  const sessionID = session.getSessionID();
  console.log(`[session-bridge] Session ID: ${sessionID}`);
  console.log(`[session-bridge] Krasis endpoint: ${KRASIS_URL}`);
  console.log(`[session-bridge] Polling for messages...`);

  // Write Session ID to a file so the server can read and display it
  const idFile = path.join(path.dirname(IDENTITY_PATH), 'session_id');
  fs.writeFileSync(idFile, sessionID + '\n');

  const poller = new Poller();
  session.addPoller(poller);

  // Track in-flight requests to avoid duplicate processing
  const processing = new Set();

  session.on('message', async (msg) => {
    const text = msg.text?.trim();
    if (!text) return;

    const from = msg.from;
    const msgKey = `${from}:${msg.timestamp}`;
    if (processing.has(msgKey)) return;
    processing.add(msgKey);

    console.log(`[session-bridge] Message from ${from.slice(0, 8)}... (${text.length} chars)`);

    // Special commands
    if (text.toLowerCase() === '/clear') {
      conversations.delete(from);
      try {
        await session.sendMessage({ to: from, text: 'Conversation cleared.' });
      } catch (e) {
        console.error(`[session-bridge] Send error: ${e.message}`);
      }
      processing.delete(msgKey);
      return;
    }

    const history = getHistory(from);
    history.push({ role: 'user', content: text });

    // Keep history reasonable (last 20 messages)
    while (history.length > 20) {
      history.shift();
    }

    try {
      const response = await streamCompletion(history);
      if (response) {
        history.push({ role: 'assistant', content: response });

        // Session has a message size limit (~6KB). Split long responses.
        const MAX_CHUNK = 5000;
        for (let i = 0; i < response.length; i += MAX_CHUNK) {
          const chunk = response.slice(i, i + MAX_CHUNK);
          await session.sendMessage({ to: from, text: chunk });
        }
        console.log(`[session-bridge] Replied to ${from.slice(0, 8)}... (${response.length} chars)`);
      }
    } catch (e) {
      console.error(`[session-bridge] Error: ${e.message}`);
      try {
        await session.sendMessage({ to: from, text: `Error: ${e.message}` });
      } catch {
        // give up on sending error
      }
    }

    processing.delete(msgKey);
  });

  // Keep alive
  process.on('SIGINT', () => {
    console.log('\n[session-bridge] Stopping...');
    poller.stopPolling();
    process.exit(0);
  });
  process.on('SIGTERM', () => {
    poller.stopPolling();
    process.exit(0);
  });
}

main().catch(e => {
  console.error(`[session-bridge] Fatal: ${e.message}`);
  process.exit(1);
});
