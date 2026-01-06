type Env = {
  RUNPOD_API_KEY?: string;
  RUNPOD_ENDPOINT_ID?: string;
  RUNPOD_API_BASE?: string;
  CLIENT_API_KEY?: string;
  // Preferred: direct HTTP upstream (RunPod Serverless Load Balancing / HTTP Workers).
  UPSTREAM_BASE_URL?: string;
  // Optional upstream auth (Bearer). If set, worker injects it when proxying upstream.
  UPSTREAM_API_KEY?: string;
  // If true, forwards client Authorization header to upstream (when UPSTREAM_API_KEY not set).
  FORWARD_AUTH?: string;
};

type Json = Record<string, unknown>;

function corsHeaders(req: Request): Headers {
  const headers = new Headers();
  headers.set("access-control-allow-origin", "*");
  headers.set("access-control-allow-methods", "GET,POST,OPTIONS");
  headers.set("access-control-allow-headers", "content-type,authorization");
  headers.set("access-control-max-age", "86400");
  const vary = req.headers.get("vary");
  headers.set("vary", vary ? `${vary}, Origin` : "Origin");
  return headers;
}

function jsonResponse(req: Request, body: unknown, init?: ResponseInit) {
  const headers = corsHeaders(req);
  if (init?.headers) new Headers(init.headers).forEach((v, k) => headers.set(k, v));
  headers.set("content-type", "application/json; charset=utf-8");
  return new Response(JSON.stringify(body), { ...init, headers });
}

function textResponse(req: Request, text: string, init?: ResponseInit) {
  const headers = corsHeaders(req);
  if (init?.headers) new Headers(init.headers).forEach((v, k) => headers.set(k, v));
  headers.set("content-type", "text/plain; charset=utf-8");
  const status = init?.status ?? 200;
  // Per Fetch spec, these statuses must not include a response body.
  if (status === 101 || status === 204 || status === 205 || status === 304) {
    return new Response(null, { ...init, status, headers });
  }
  return new Response(text, { ...init, status, headers });
}

function authOk(req: Request, env: Env): boolean {
  const expected = (env.CLIENT_API_KEY || "").trim();
  if (!expected) return true;
  const header = req.headers.get("authorization") || "";
  const m = header.match(/^Bearer\s+(.+)$/i);
  return Boolean(m && m[1] === expected);
}

function boolEnv(v: string | undefined): boolean {
  const s = (v || "").trim().toLowerCase();
  return s === "1" || s === "true" || s === "yes" || s === "on";
}

async function parseJsonBody(req: Request): Promise<Json> {
  const ct = req.headers.get("content-type") || "";
  if (!ct.toLowerCase().includes("application/json")) {
    throw new Error("Content-Type must be application/json");
  }
  const body = await req.json();
  if (!body || typeof body !== "object" || Array.isArray(body)) {
    throw new Error("JSON body must be an object");
  }
  return body as Json;
}

function runpodBase(env: Env) {
  return (env.RUNPOD_API_BASE || "https://api.runpod.ai/v2").replace(/\/+$/, "");
}

type RunPodRunResponse = { id: string };
type RunPodStatus =
  | "IN_QUEUE"
  | "IN_PROGRESS"
  | "COMPLETED"
  | "FAILED"
  | "CANCELLED"
  | "TIMED_OUT";

type RunPodStatusResponse = {
  id: string;
  status: RunPodStatus;
  output?: unknown;
  error?: unknown;
};

function withTimeout(ms: number) {
  const controller = new AbortController();
  const t = setTimeout(() => controller.abort(), ms);
  return { signal: controller.signal, cancel: () => clearTimeout(t) };
}

function requireRunpod(env: Env): { apiKey: string; endpointId: string } {
  const apiKey = (env.RUNPOD_API_KEY || "").trim();
  const endpointId = (env.RUNPOD_ENDPOINT_ID || "").trim();
  if (!apiKey) throw new Error("RUNPOD_API_KEY not configured");
  if (!endpointId) throw new Error("RUNPOD_ENDPOINT_ID not configured");
  return { apiKey, endpointId };
}

async function runpodRun(env: Env, input: Json): Promise<RunPodRunResponse> {
  const { apiKey, endpointId } = requireRunpod(env);
  const url = `${runpodBase(env)}/${endpointId}/run`;
  const { signal, cancel } = withTimeout(10_000);
  try {
    const res = await fetch(url, {
      method: "POST",
      headers: {
        authorization: `Bearer ${apiKey}`,
        "content-type": "application/json",
      },
      body: JSON.stringify({ input }),
      signal,
    });
    const text = await res.text();
    if (!res.ok) throw new Error(`RunPod /run failed (${res.status}): ${text}`);
    const data = JSON.parse(text);
    if (!data?.id) throw new Error(`RunPod /run returned unexpected body: ${text}`);
    return { id: String(data.id) };
  } finally {
    cancel();
  }
}

async function runpodStatus(env: Env, jobId: string): Promise<RunPodStatusResponse> {
  const { apiKey, endpointId } = requireRunpod(env);
  const url = `${runpodBase(env)}/${endpointId}/status/${jobId}`;
  const { signal, cancel } = withTimeout(10_000);
  try {
    const res = await fetch(url, {
      method: "GET",
      headers: { authorization: `Bearer ${apiKey}` },
      signal,
    });
    const text = await res.text();
    if (!res.ok) throw new Error(`RunPod /status failed (${res.status}): ${text}`);
    return JSON.parse(text) as RunPodStatusResponse;
  } finally {
    cancel();
  }
}

function upstreamBase(env: Env): string | null {
  const raw = (env.UPSTREAM_BASE_URL || "").trim();
  if (!raw) return null;
  return raw.replace(/\/+$/, "");
}

async function upstreamProxy(req: Request, env: Env, url: URL, bodyOverride?: string): Promise<Response> {
  const base = upstreamBase(env);
  if (!base) return jsonResponse(req, { error: "UPSTREAM_BASE_URL not configured" }, { status: 500 });

  const target = `${base}${url.pathname}${url.search}`;

  const headers = new Headers(req.headers);
  headers.delete("host");
  headers.delete("content-length");

  const forwardAuth = boolEnv(env.FORWARD_AUTH);
  if (env.UPSTREAM_API_KEY && env.UPSTREAM_API_KEY.trim()) {
    headers.set("authorization", `Bearer ${env.UPSTREAM_API_KEY.trim()}`);
  } else if (!forwardAuth) {
    headers.delete("authorization");
  }

  const init: RequestInit = {
    method: req.method,
    headers,
    body: bodyOverride !== undefined ? bodyOverride : req.body,
    redirect: "manual",
  };

  const res = await fetch(target, init);

  // Copy upstream headers, then add CORS headers (do not buffer body to preserve streaming).
  const outHeaders = corsHeaders(req);
  res.headers.forEach((v, k) => outHeaders.set(k, v));
  return new Response(res.body, { status: res.status, headers: outHeaders });
}

async function waitForCompletion(env: Env, jobId: string, opts: { maxWaitMs: number; pollMs: number }) {
  const deadline = Date.now() + opts.maxWaitMs;
  // eslint-disable-next-line no-constant-condition
  while (true) {
    const status = await runpodStatus(env, jobId);
    if (status.status === "COMPLETED" || status.status === "FAILED" || status.status === "CANCELLED") return status;
    if (Date.now() >= deadline) return status;
    await new Promise((r) => setTimeout(r, opts.pollMs));
  }
}

function routeToInput(pathname: string, body: Json): Json {
  const p = pathname.replace(/\/+$/, "");
  if (p === "" || p === "/") return { action: "imme", ...body };
  if (p === "/imme") return { action: "imme", ...body };
  if (p === "/translate") return { action: "translate", ...body };
  if (p === "/detect") return { action: "detect", ...body };
  return { action: p.slice(1), ...body };
}

function asStringArray(v: unknown): string[] | null {
  if (!Array.isArray(v)) return null;
  for (const item of v) {
    if (typeof item !== "string") return null;
  }
  return v as string[];
}

export default {
  async fetch(req: Request, env: Env): Promise<Response> {
    const url = new URL(req.url);

    if (req.method === "OPTIONS") return new Response(null, { status: 204, headers: corsHeaders(req) });
    if (!authOk(req, env)) return jsonResponse(req, { error: "unauthorized" }, { status: 401 });

    const mode = (url.searchParams.get("mode") || "").trim().toLowerCase(); // "upstream" | "job"
    const hasUpstream = Boolean(upstreamBase(env));
    const useUpstream = hasUpstream && mode !== "job";

    // Proxy job status for async clients: GET /job/<id>
    const jobMatch = url.pathname.match(/^\/job\/([^/]+)$/);
    if (req.method === "GET" && jobMatch) {
      if (useUpstream) {
        return jsonResponse(req, { error: "job status is not available in upstream mode; add ?mode=job" }, { status: 400 });
      }
      try {
        const jobId = decodeURIComponent(jobMatch[1]);
        const status = await runpodStatus(env, jobId);
        return jsonResponse(req, status, { status: 200 });
      } catch (e) {
        return jsonResponse(req, { error: String(e) }, { status: 502 });
      }
    }

    if (req.method === "GET" && (url.pathname === "/health" || url.pathname === "/")) {
      if (useUpstream) {
        // If upstream is configured, surface upstream health directly.
        const upstreamUrl = new URL(req.url);
        upstreamUrl.pathname = "/health";
        upstreamUrl.search = "";
        return upstreamProxy(req, env, upstreamUrl);
      }
      return jsonResponse(req, { ok: true, endpoint: env.RUNPOD_ENDPOINT_ID }, { status: 200 });
    }

    // In upstream mode, proxy everything (supports SSE streaming).
    if (useUpstream) {
      if (req.method === "POST") {
        let body: Json;
        try {
          body = await parseJsonBody(req);
        } catch (e) {
          return jsonResponse(req, { error: String(e) }, { status: 400 });
        }

        // Guardrails to avoid pathological requests.
        if (url.pathname.replace(/\/+$/, "") === "/imme") {
          const texts = asStringArray(body.text_list);
          if (texts) {
            const maxTexts = 256;
            if (texts.length > maxTexts) {
              return jsonResponse(
                req,
                { error: `text_list too large (${texts.length}); please split and retry`, max_texts: maxTexts },
                { status: 413 }
              );
            }
          }
        }

        // Re-serialize since we've consumed the request body.
        return upstreamProxy(req, env, url, JSON.stringify(body));
      }

      // Non-POST: proxy directly (e.g. GET /v1/models).
      return upstreamProxy(req, env, url);
    }

    // Job mode (Traditional Serverless): translate HTTP routes into RunPod job API.
    if (req.method !== "POST") return jsonResponse(req, { error: "method not allowed" }, { status: 405 });

    let body: Json;
    try {
      body = await parseJsonBody(req);
    } catch (e) {
      return jsonResponse(req, { error: String(e) }, { status: 400 });
    }

    const input = routeToInput(url.pathname, body);
    const forceAsync = url.searchParams.get("async") === "1";

    // Gateway-side guardrails to avoid Serverless OOM / pathological requests.
    if (input.action === "imme") {
      const texts = asStringArray(input.text_list);
      if (texts) {
        const maxTexts = 256;
        if (texts.length > maxTexts) {
          return jsonResponse(
            req,
            { error: `text_list too large (${texts.length}); please split and retry`, max_texts: maxTexts },
            { status: 413 }
          );
        }
      }
    }

    try {
      const run = await runpodRun(env, input);

      if (forceAsync) {
        return jsonResponse(
          req,
          { id: run.id, status: "SUBMITTED", status_url: `/job/${encodeURIComponent(run.id)}` },
          { status: 202 }
        );
      }

      const status = await waitForCompletion(env, run.id, { maxWaitMs: 25_000, pollMs: 800 });
      if (status.status === "COMPLETED") return jsonResponse(req, status.output ?? {}, { status: 200 });
      if (status.status === "FAILED") return jsonResponse(req, { error: "runpod job failed", runpod: status }, { status: 502 });

      return jsonResponse(
        req,
        { id: status.id, status: status.status, status_url: `/job/${encodeURIComponent(status.id)}` },
        { status: 202 }
      );
    } catch (e) {
      return jsonResponse(req, { error: String(e) }, { status: 502 });
    }
  },
};
