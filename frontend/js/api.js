/**
 * API client for CiteQuest-RAG.
 *
 * Search returns JSON; AI Overview delivers SSE events through callbacks.
 */

const API_BASE = "";

/**
 * Generic POST helper.
 */
async function apiPost(path, body, signal) {
  const resp = await fetch(`${API_BASE}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
    signal,
  });

  if (!resp.ok) {
    const text = await resp.text().catch(() => "unknown error");
    throw new Error(`HTTP ${resp.status}: ${text}`);
  }

  return resp.json();
}

/**
 * POST /search — perform academic search with optional AI overview.
 */
async function searchPapers({
  query,
  top_k = 10,
  mode = "hybrid",
  alpha = 0.5,
  include_overview = true,
  year_from = null,
  year_to = null,
  signal = null,
} = {}) {
  return apiPost("/search", {
    query,
    top_k,
    mode,
    alpha,
    include_overview,
    year_from,
    year_to,
  }, signal);
}

/**
 * POST /ask/stream — SSE streaming RAG with real-time phase updates.
 *
 * @param {Object} body — request body { question, top_k, retrieval_mode, alpha }
 * @param {Function} onEvent — called with (eventType, data) for each SSE event
 * @param {Function} onDone — called when the stream ends cleanly
 * @param {Function} onError — called with (error) on network/parse failure
 * @returns {AbortController} — call .abort() to cancel
 */
function askQuestionStream({ body, onEvent, onDone, onError } = {}) {
  const controller = new AbortController();

  (async () => {
    try {
      const resp = await fetch(`${API_BASE}/ask/stream`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
        signal: controller.signal,
      });

      if (!resp.ok) {
        const text = await resp.text().catch(() => "unknown error");
        throw new Error(`HTTP ${resp.status}: ${text}`);
      }

      const reader = resp.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });

        // SSE events are separated by double newlines
        const parts = buffer.split("\n\n");
        buffer = parts.pop() || "";  // keep incomplete event in buffer

        for (const part of parts) {
          if (!part.trim()) continue;

          let eventType = "message";
          let data = "";

          for (const line of part.split("\n")) {
            if (line.startsWith("event: ")) {
              eventType = line.slice(7).trim();
            } else if (line.startsWith("data: ")) {
              data = line.slice(6);
            }
          }

          // Try to parse data as JSON
          let parsed = data;
          try { parsed = JSON.parse(data); } catch (_) { /* plain string */ }

          if (eventType === "done") {
            if (onDone) onDone();
            return;
          }

          if (onEvent) onEvent(eventType, parsed);
        }
      }

      if (onDone) onDone();
    } catch (e) {
      if (e.name === "AbortError") return;
      console.error("SSE stream error:", e);
      if (onError) onError(e);
    }
  })();

  return controller;
}
