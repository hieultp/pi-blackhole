import { describe, expect, it, vi } from "vitest";
import type { Model } from "@earendil-works/pi-ai";
import type { Observation } from "../src/om/ledger/index.js";
import {
  buildWarmFetch,
  extractJsonObject,
  KvWarmCaptureStore,
  runWarmOrFallback,
  runWarmObserver,
  validateWarmDropperOutput,
  validateWarmObserverOutput,
  validateWarmReflectorOutput,
  warmPromptFitsBudget,
  workerCanWarm,
  type KvWarmChain,
  type WarmDropperArgs,
  type WarmObserverArgs,
  type WarmReflectorArgs,
} from "../src/om/kv-warm.js";

const model = {
  provider: "test-provider",
  api: "openai-completions",
  id: "test-model",
  name: "Test Model",
  baseUrl: "http://127.0.0.1:1234/v1",
  input: ["text"],
  cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
  contextWindow: 100_000,
  maxTokens: 4096,
  reasoning: true,
} as Model<any>;

function payload(lastUser = "hello") {
  return {
    model: model.id,
    stream: true,
    max_tokens: 1024,
    temperature: 0.4,
    messages: [
      { role: "system", content: "system prompt" },
      { role: "user", content: lastUser },
    ],
    tools: [{ type: "function", function: { name: "read", parameters: { type: "object" } } }],
  };
}

function projection(lastUser = "hello") {
  return [{ role: "user", content: lastUser }];
}

function captureChain(store = new KvWarmCaptureStore()): KvWarmChain {
  store.capture("session", payload(), model, 1_000);
  const chain = store.get("session", projection(), model, 60_000, 1_000);
  if (!chain) throw new Error("fixture capture did not validate");
  return chain;
}

function common() {
  return {
    chain: {} as KvWarmChain,
    model,
    apiKey: "unused",
    sessionId: "session",
    maxOutputTokens: 256,
    inputBudget: 100_000,
  };
}

describe("KV-warm capture safety", () => {
  it("requires exact model, endpoint, current projected tail, and freshness", () => {
    const store = new KvWarmCaptureStore();
    expect(store.capture("s", payload(), model, 1_000)).toBe(true);
    expect(store.get("s", projection(), model, 600_000, 1_000)).toBeDefined();
    expect(store.get("s", projection("changed"), model, 600_000, 1_000)).toBeUndefined();
    expect(
      store.get("s", projection(), { ...model, baseUrl: "http://other/v1" }, 600_000, 1_000),
    ).toBeUndefined();

    store.capture("s", payload(), model, 1_000);
    expect(store.get("s", projection(), model, 500, 1_501)).toBeUndefined();
    expect(store.get("s", projection(), model, 600_000, 1_501)).toBeUndefined();
  });

  it("discards unsupported, non-text, and oversized provider payloads", () => {
    const store = new KvWarmCaptureStore();
    expect(
      store.capture(
        "s",
        { ...payload(), messages: [{ role: "user", content: [{ type: "image_url" }] }] },
        model,
      ),
    ).toBe(false);
    expect(store.capture("s", { ...payload(), max_tokens: undefined }, model)).toBe(false);
    expect(
      store.capture(
        "s",
        { ...payload(), messages: [{ role: "user", content: "x".repeat(1_100_000) }] },
        model,
      ),
    ).toBe(false);
    expect(
      store.capture(
        "s",
        { ...payload(), messages: [{ role: "user", content: "中".repeat(800_000) }] },
        model,
      ),
    ).toBe(false);
    expect(store.capture("s", payload(), { ...model, baseUrl: undefined })).toBe(false);
    expect(store.capture("s", payload(), { ...model, api: "anthropic-messages" })).toBe(false);
  });

  it("bounds retained sessions and invalidates a prior capture on an unsupported request", () => {
    const store = new KvWarmCaptureStore();
    for (let i = 0; i < 5; i++) store.capture(`s${i}`, payload(`u${i}`), model, 1_000 + i);
    expect(store.get("s0", projection("u0"), model, 60_000, 2_000)).toBeUndefined();
    expect(store.get("s4", projection("u4"), model, 60_000, 2_000)).toBeDefined();

    expect(store.capture("s4", { not: "a request" }, model)).toBe(false);
    expect(store.get("s4", projection("u4"), model, 60_000, Date.now())).toBeUndefined();
  });
});

describe("KV-warm request and preflight", () => {
  it("replaces only the body while preserving transport options and worker limits", async () => {
    const chain = captureChain();
    chain.capture.payload.prompt_cache_key = "session-key";
    chain.capture.payload.prompt_cache_retention = "24h";
    const sent: Array<{ input: RequestInfo | URL; init?: RequestInit }> = [];
    const baseFetch = vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      sent.push({ input, init });
      return new Response(null, { status: 200 });
    });
    const messages = [...chain.messages, { role: "user", content: "worker task" }];
    const wrapped = buildWarmFetch(baseFetch as typeof fetch, chain.capture, messages, 77);
    const headers = { authorization: "Bearer worker", "x-trace": "kept" };
    await wrapped("http://127.0.0.1:1234/v1/chat/completions", {
      method: "POST",
      headers,
      body: JSON.stringify({
        model: model.id,
        stream: true,
        max_tokens: 1000,
        messages: [],
        reasoning_effort: "high",
        prompt_cache_key: "worker-key",
        prompt_cache_retention: "24h",
      }),
    });

    expect(sent[0].input).toBe("http://127.0.0.1:1234/v1/chat/completions");
    expect(sent[0].init).toMatchObject({ method: "POST", headers });
    const sentBody = JSON.parse(String(sent[0].init?.body));
    expect(sentBody.messages).toEqual(messages);
    expect(sentBody.tools).toEqual(chain.capture.payload.tools);
    expect(sentBody.max_tokens).toBe(77);
    expect(sentBody.reasoning_effort).toBe("high");
    expect(sentBody.prompt_cache_key).toBe("worker-key");
    expect(sentBody.prompt_cache_retention).toBe("24h");
    expect(sentBody.temperature).toBe(0.4);

    await wrapped("http://127.0.0.1:1234/v1/chat/completions", {
      method: "POST",
      body: JSON.stringify({
        model: model.id,
        stream: true,
        max_completion_tokens: 1000,
        messages: [],
      }),
    });
    const noCacheBody = JSON.parse(String(sent[1].init?.body));
    expect(noCacheBody).not.toHaveProperty("prompt_cache_key");
    expect(noCacheBody).not.toHaveProperty("prompt_cache_retention");
    expect(noCacheBody).not.toHaveProperty("max_tokens");
    expect(noCacheBody.max_completion_tokens).toBe(77);
  });

  it("fails closed on a non-OpenAI request body", async () => {
    const chain = captureChain();
    const unsupported = vi.fn();
    const baseFetch = vi.fn();
    const wrapped = buildWarmFetch(
      baseFetch as typeof fetch,
      chain.capture,
      chain.messages,
      77,
      undefined,
      unsupported,
    );
    await expect(wrapped("http://localhost", { method: "POST", body: "not json" })).rejects.toThrow(
      "KV warm request shape unsupported",
    );
    expect(unsupported).toHaveBeenCalledOnce();
    expect(baseFetch).not.toHaveBeenCalled();
  });

  it("requires matching workers and budgets the full corrective retry transcript", () => {
    const chain = captureChain();
    expect(workerCanWarm(chain, model)).toBe(true);
    expect(workerCanWarm(chain, { ...model, id: "other" } as Model<any>)).toBe(false);
    expect(warmPromptFitsBudget(chain, "small task", 256, 100_000)).toBe(true);
    expect(warmPromptFitsBudget(chain, "small task", 99_000, 100_000)).toBe(true);
    expect(warmPromptFitsBudget(chain, "small task", 256, 100)).toBe(false);
    expect(
      warmPromptFitsBudget(chain, "small task", 256, 100_000, [
        ...chain.messages,
        { role: "assistant", content: "x".repeat(500_000) },
        { role: "user", content: "correct" },
      ]),
    ).toBe(false);
  });

  it("routes a warm worker through pi-ai's fetch option and substitutes the captured prefix", async () => {
    const chain = captureChain();
    const capturedMessages = [...chain.messages];
    const responseText = '{"observations":[]}';
    const sse = [
      `data: ${JSON.stringify({ id: "chatcmpl-test", object: "chat.completion.chunk", created: 1, model: model.id, choices: [{ index: 0, delta: { role: "assistant", content: responseText }, finish_reason: null }] })}\n\n`,
      `data: ${JSON.stringify({ id: "chatcmpl-test", object: "chat.completion.chunk", created: 1, model: model.id, choices: [{ index: 0, delta: {}, finish_reason: "stop" }] })}\n\n`,
      "data: [DONE]\n\n",
    ].join("");
    let requestBody: Record<string, unknown> | undefined;
    const mockFetch = vi.fn(async (_input: RequestInfo | URL, init?: RequestInit) => {
      requestBody = JSON.parse(String(init?.body));
      return new Response(sse, { status: 200, headers: { "content-type": "text/event-stream" } });
    });
    vi.stubGlobal("fetch", mockFetch);
    try {
      const result = await runWarmObserver({
        ...common(),
        chain,
        priorReflections: [],
        priorObservations: [],
        priorObservationIds: [],
        chunk: "source-addressed conversation",
        allowedSourceEntryIds: ["entry-1"],
        inputBudget: 1_000_000,
      });
      expect(mockFetch).toHaveBeenCalledOnce();
      expect(result?.coverageComplete).toBe(true);
      expect(requestBody?.messages).toEqual([
        ...capturedMessages,
        { role: "user", content: expect.stringContaining("source-addressed conversation") },
      ]);
      expect(requestBody?.max_tokens ?? requestBody?.max_completion_tokens).toBe(256);
    } finally {
      vi.unstubAllGlobals();
    }
  });

  it("ends after a tool-call response and uses only one corrective warm retry", async () => {
    const chain = captureChain();
    const toolCallSse = [
      `data: ${JSON.stringify({ id: "chatcmpl-tool", object: "chat.completion.chunk", created: 1, model: model.id, choices: [{ index: 0, delta: { tool_calls: [{ index: 0, id: "call-1", type: "function", function: { name: "read", arguments: "{}" } }] }, finish_reason: null }] })}\n\n`,
      `data: ${JSON.stringify({ id: "chatcmpl-tool", object: "chat.completion.chunk", created: 1, model: model.id, choices: [{ index: 0, delta: {}, finish_reason: "tool_calls" }] })}\n\n`,
      "data: [DONE]\n\n",
    ].join("");
    const validSse = [
      `data: ${JSON.stringify({ id: "chatcmpl-retry", object: "chat.completion.chunk", created: 1, model: model.id, choices: [{ index: 0, delta: { role: "assistant", content: '{"observations":[]}' }, finish_reason: null }] })}\n\n`,
      `data: ${JSON.stringify({ id: "chatcmpl-retry", object: "chat.completion.chunk", created: 1, model: model.id, choices: [{ index: 0, delta: {}, finish_reason: "stop" }] })}\n\n`,
      "data: [DONE]\n\n",
    ].join("");
    const requests: Record<string, unknown>[] = [];
    const mockFetch = vi.fn(async (_input: RequestInfo | URL, init?: RequestInit) => {
      requests.push(JSON.parse(String(init?.body)));
      const sse = requests.length === 1 ? toolCallSse : validSse;
      return new Response(sse, { status: 200, headers: { "content-type": "text/event-stream" } });
    });
    vi.stubGlobal("fetch", mockFetch);
    try {
      const result = await runWarmObserver({
        ...common(),
        chain,
        priorReflections: [],
        priorObservations: [],
        priorObservationIds: [],
        chunk: "source-addressed conversation",
        allowedSourceEntryIds: ["entry-1"],
        inputBudget: 1_000_000,
      });
      expect(result?.coverageComplete).toBe(true);
      expect(mockFetch).toHaveBeenCalledTimes(2);
      const retryMessages = requests[1].messages as Array<{ role: string; content: string }>;
      expect(retryMessages[3].content).toContain("previous response was invalid");
    } finally {
      vi.unstubAllGlobals();
    }
  });

  it("invalidates the chain and runs the normal worker when warm mode is unavailable", async () => {
    const chain = captureChain();
    const cold = vi.fn(async () => "chunk result");
    await expect(runWarmOrFallback(chain, async () => undefined, cold)).resolves.toBe(
      "chunk result",
    );
    expect(chain.invalidated).toBe(true);
    expect(cold).toHaveBeenCalledOnce();
  });
});

describe("KV-warm JSON results", () => {
  it("accepts fenced JSON and extracts an object from surrounding prose", () => {
    expect(extractJsonObject('```json\n{"observations":[]}\n```')).toEqual({ observations: [] });
    expect(extractJsonObject('Result: {"ids":[]} done')).toEqual({ ids: [] });
    expect(extractJsonObject("not JSON")).toBeUndefined();
  });

  it("maps an empty valid observer result to complete coverage", () => {
    const args = {
      ...common(),
      priorReflections: [],
      priorObservations: [],
      priorObservationIds: [],
      chunk: "conversation",
      allowedSourceEntryIds: ["entry-1"],
      sourceEntryTimestamps: { "entry-1": "2026-01-01 12:00" },
    } as WarmObserverArgs;
    expect(validateWarmObserverOutput({ observations: [] }, args)).toEqual({
      ok: true,
      value: {
        observations: undefined,
        emptyReason: { kind: "no_new_content" },
        coverageComplete: true,
      },
    });
    expect(
      validateWarmObserverOutput(
        { observations: [{ content: "fact", relevance: "high", sourceEntryIds: ["unknown"] }] },
        args,
      ).ok,
    ).toBe(false);
  });

  it("accepts empty reflector and dropper JSON as successful no-op results", () => {
    const observation: Observation = {
      id: "0123456789ab",
      content: "source fact",
      timestamp: "2026-01-01 12:00",
      relevance: "high",
      sourceEntryIds: ["entry-1"],
      tokenCount: 3,
    };
    const reflectorArgs = {
      ...common(),
      reflections: [],
      observations: [observation],
      supersedesEnabled: false,
      timestampsEnabled: true,
    } as WarmReflectorArgs;
    expect(validateWarmReflectorOutput({ reflections: [] }, reflectorArgs)).toEqual({
      ok: true,
      value: { reflections: undefined },
    });

    const dropperArgs = {
      ...common(),
      reflections: [],
      observations: [observation],
      budgetTokens: 100,
    } as WarmDropperArgs;
    expect(validateWarmDropperOutput({ ids: [] }, dropperArgs)).toEqual({
      ok: true,
      value: undefined,
    });
  });
});
