import { agentLoop, type AgentLoopConfig } from "@earendil-works/pi-agent-core";
import { Buffer } from "node:buffer";
import type { CacheRetention, Message, Model, ModelThinkingLevel } from "@earendil-works/pi-ai";
import { streamSimple } from "@earendil-works/pi-ai/compat";
import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";
import { hashId } from "./ids.js";
import { estimateStringTokens } from "./tokens.js";
import { createBridgeStreamFn, createProviderFetch, neverThrow } from "./provider-stream.js";
import { buildAgentContext } from "./agents/agent-context.js";
import type { ObserverResult } from "./agents/observer/agent.js";
import { OBSERVER_SYSTEM } from "./agents/observer/prompts.js";
import {
  deriveObservationTimestamp,
  normalizeDropObservationIds,
  normalizeReflectionContent,
  normalizeSourceEntryIds,
  normalizeSupportingObservationIds,
} from "./agents/record-validation.js";
import { buildReflectorSystem } from "./agents/reflector/prompts.js";
import type { ReflectorResult } from "./agents/reflector/agent.js";
import { DROPPER_SYSTEM } from "./agents/dropper/prompts.js";
import {
  dropUrgencyForFullness,
  maxDropCountForPool,
  observationPoolFullness,
  selectDropCandidates,
} from "./agents/dropper/selection.js";
import {
  coverageTierForObservation,
  observationToDropperLine,
  reflectionCoverageMap,
} from "./agents/dropper/coverage.js";
import {
  isMemoryId,
  isReflectionTimestamp,
  observationToSummaryLine,
  reflectionToSummaryLine,
  type Observation,
  type Reflection,
  type Relevance,
} from "./ledger/index.js";
import { truncateRecordContent } from "./serialize.js";
import { WorkerStreamError, workerStreamErrorMessage } from "./retryable-error.js";
import type { WorkerStage } from "./worker-attempt.js";
import type { Runtime } from "./runtime.js";

const MAX_CAPTURE_SESSIONS = 4;
const MAX_CAPTURE_BYTES = 2 * 1024 * 1024;
const MAX_RETRY_REPLY_CHARS = 32_000;
const WARM_RETRY_HEADROOM_TOKENS = 512;
const WARM_ESTIMATE_FACTOR = 1.25;
const WARM_UNSUPPORTED = "KV warm request shape unsupported";
const REASONING_REQUEST_FIELD = /^(?:reasoning|thinking|think|enable_thinking|chat_template)/i;

export interface KvWarmModelIdentity {
  provider: string;
  api: string;
  id: string;
  baseUrl: string;
}

export interface KvWarmCapture {
  payload: Record<string, unknown>;
  model: KvWarmModelIdentity;
  capturedAt: number;
  tailText: string;
}

export interface KvWarmChain {
  capture: KvWarmCapture;
  messages: unknown[];
  invalidated: boolean;
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function modelIdentity(value: unknown): KvWarmModelIdentity | undefined {
  if (!isRecord(value)) return undefined;
  const { provider, api, id, baseUrl } = value;
  if (
    typeof provider !== "string" ||
    typeof api !== "string" ||
    typeof id !== "string" ||
    provider.length === 0 ||
    id.length === 0 ||
    typeof baseUrl !== "string" ||
    baseUrl.length === 0
  ) {
    return undefined;
  }
  return { provider, api, id, baseUrl };
}

function sameModel(a: KvWarmModelIdentity, b: KvWarmModelIdentity): boolean {
  return (
    a.provider === b.provider &&
    a.api === b.api &&
    a.id === b.id &&
    a.baseUrl === b.baseUrl &&
    a.api === "openai-completions"
  );
}

function contentText(content: unknown): string | undefined {
  if (typeof content === "string") return content;
  if (!Array.isArray(content)) return undefined;
  const parts: string[] = [];
  for (const part of content) {
    if (!isRecord(part)) continue;
    if ((part.type === "text" || part.type === "input_text") && typeof part.text === "string") {
      parts.push(part.text);
    }
  }
  return parts.length ? parts.join("\n") : undefined;
}

function lastUserText(messages: unknown): string | undefined {
  if (!Array.isArray(messages)) return undefined;
  for (let i = messages.length - 1; i >= 0; i--) {
    const message = messages[i];
    if (isRecord(message) && message.role === "user") return contentText(message.content);
  }
  return undefined;
}

function hasTextOnlyMessages(messages: unknown[]): boolean {
  return messages.every((message) => {
    if (!isRecord(message)) return false;
    const { content } = message;
    if (content === undefined || content === null || typeof content === "string") return true;
    return (
      Array.isArray(content) &&
      content.every(
        (part) =>
          isRecord(part) &&
          (part.type === "text" || part.type === "input_text") &&
          typeof part.text === "string",
      )
    );
  });
}

function validRequestPayload(
  payload: unknown,
  model: KvWarmModelIdentity,
): payload is Record<string, unknown> {
  if (!isRecord(payload) || !Array.isArray(payload.messages)) return false;
  if (payload.model !== model.id || payload.stream !== true) return false;
  if (payload.tools !== undefined && !Array.isArray(payload.tools)) return false;
  if (!payload.messages.every((message) => isRecord(message) && typeof message.role === "string"))
    return false;
  if (!hasTextOnlyMessages(payload.messages)) return false;
  const tokenField =
    typeof payload.max_tokens === "number"
      ? "max_tokens"
      : typeof payload.max_completion_tokens === "number"
        ? "max_completion_tokens"
        : undefined;
  return tokenField !== undefined && (payload[tokenField] as number) > 0;
}

export class KvWarmCaptureStore {
  private readonly captures = new Map<string, KvWarmCapture>();

  capture(
    sessionId: string,
    rawPayload: unknown,
    rawModel: unknown,
    capturedAt = Date.now(),
  ): boolean {
    this.captures.delete(sessionId);
    const model = modelIdentity(rawModel);
    if (!model || model.api !== "openai-completions" || !isRecord(rawPayload)) return false;
    let serialized: string;
    try {
      serialized = JSON.stringify(rawPayload);
    } catch {
      return false;
    }
    if (
      serialized.length > MAX_CAPTURE_BYTES / 2 ||
      Buffer.byteLength(serialized, "utf8") > MAX_CAPTURE_BYTES
    )
      return false;

    let payload: unknown;
    try {
      payload = JSON.parse(serialized);
    } catch {
      return false;
    }
    if (!validRequestPayload(payload, model)) return false;
    const tailText = lastUserText(payload.messages);
    if (tailText === undefined || tailText.length === 0) return false;

    this.captures.set(sessionId, { payload, model, capturedAt, tailText });
    while (this.captures.size > MAX_CAPTURE_SESSIONS) {
      const oldestSession = this.captures.keys().next().value;
      if (oldestSession === undefined) break;
      this.captures.delete(oldestSession);
    }
    return true;
  }

  get(
    sessionId: string,
    projectionMessages: unknown,
    sessionModel: unknown,
    staleMs: number,
    now = Date.now(),
  ): KvWarmChain | undefined {
    const capture = this.captures.get(sessionId);
    if (!capture) return undefined;
    const projectionTail = lastUserText(projectionMessages);
    const identity = modelIdentity(sessionModel);
    if (
      !identity ||
      !sameModel(capture.model, identity) ||
      projectionTail !== capture.tailText ||
      !Number.isFinite(staleMs) ||
      staleMs <= 0 ||
      now < capture.capturedAt ||
      now - capture.capturedAt > staleMs
    ) {
      this.captures.delete(sessionId);
      return undefined;
    }
    return {
      capture,
      messages: [...(capture.payload.messages as unknown[])],
      invalidated: false,
    };
  }

  delete(sessionId: string): void {
    this.captures.delete(sessionId);
  }

  clear(): void {
    this.captures.clear();
  }
}

const stores = new WeakMap<object, KvWarmCaptureStore>();

export function registerKvWarmCapture(pi: ExtensionAPI, runtime: Runtime): void {
  const store = new KvWarmCaptureStore();
  stores.set(runtime, store);
  pi.on("before_provider_request", (event: any, ctx: any) => {
    if (runtime.config.kvWarmWorkers === "off") {
      store.clear();
      return;
    }
    try {
      const sessionId = ctx.sessionManager.getSessionId();
      store.capture(sessionId, event.payload, ctx.model);
    } catch {
      // A disposed session context or unsupported provider payload simply disables warming.
    }
  });
  pi.on("session_shutdown", (_event: any, ctx: any) => {
    try {
      store.delete(ctx.sessionManager.getSessionId());
    } catch {
      // Session cleanup is best-effort during shutdown.
    }
  });
}

export function getKvWarmChain(
  runtime: Runtime,
  sessionId: string,
  sessionManager: { buildSessionProjection?: () => { messages?: unknown } },
  sessionModel: unknown,
  staleMs: number,
  now = Date.now(),
): KvWarmChain | undefined {
  if (runtime.config.kvWarmWorkers === "off") {
    stores.get(runtime)?.clear();
    return undefined;
  }
  const store = stores.get(runtime);
  if (!store || typeof sessionManager.buildSessionProjection !== "function") return undefined;
  try {
    const projection = sessionManager.buildSessionProjection();
    return store.get(sessionId, projection.messages, sessionModel, staleMs, now);
  } catch {
    return undefined;
  }
}

export async function runWarmOrFallback<T>(
  chain: KvWarmChain | undefined,
  runWarm: (chain: KvWarmChain) => Promise<T | undefined>,
  runCold: () => Promise<T>,
): Promise<T> {
  if (!chain || chain.invalidated) return runCold();
  const result = await runWarm(chain);
  if (result !== undefined) return result;
  chain.invalidated = true;
  return runCold();
}

interface WarmCommonArgs {
  chain: KvWarmChain;
  model: Model<any>;
  apiKey: string;
  headers?: Record<string, string>;
  env?: Record<string, string>;
  sessionId: string;
  maxOutputTokens: number;
  inputBudget: number;
  signal?: AbortSignal;
  providerIdleTimeoutMs?: number;
  cacheRetention?: CacheRetention;
  thinkingLevel?: ModelThinkingLevel;
  modelRegistry?: any;
}

export interface WarmObserverArgs extends WarmCommonArgs {
  priorReflections: string[];
  priorObservations: string[];
  priorObservationIds?: string[];
  chunk: string;
  allowedSourceEntryIds: string[];
  sourceEntryTimestamps?: Record<string, string>;
}

export interface WarmReflectorArgs extends WarmCommonArgs {
  supersedesEnabled?: boolean;
  timestampsEnabled?: boolean;
  supersedableReflectionIds?: readonly string[];
  renderReflection?: (reflection: Reflection) => string;
  reflections: Reflection[];
  observations: Observation[];
  existingReflectionsSummary?: string;
  existingObservationsSummary?: string;
}

export interface WarmDropperArgs extends WarmCommonArgs {
  renderReflection?: (reflection: Reflection) => string;
  reflections: Reflection[];
  observations: Observation[];
  existingObservationsSummary?: string;
  budgetTokens: number;
  skipFullness?: number;
  pressure?: { tokens: number; maxDrops: number };
  rawProposals?: boolean;
}

export type WarmValidation<T> = { ok: true; value: T } | { ok: false; error: string };

interface WarmResponse {
  text: string;
  stopReason?: string;
  errorMessage?: string;
  hasToolCall: boolean;
}

function estimateWarmPromptTokens(chain: KvWarmChain, messages: readonly unknown[]): number {
  const promptPayload = { ...chain.capture.payload, messages: [...messages] };
  return Math.ceil(estimateStringTokens(JSON.stringify(promptPayload)) * WARM_ESTIMATE_FACTOR);
}

export function warmPromptFitsBudget(
  chain: KvWarmChain,
  prompt: string,
  maxOutputTokens: number,
  inputBudget: number,
  messages: readonly unknown[] = [...chain.messages, { role: "user", content: prompt }],
): boolean {
  const output = Math.floor(maxOutputTokens);
  // inputBudget already excludes the configured output allowance and provider safety margin.
  return (
    Number.isFinite(output) &&
    output > 0 &&
    estimateWarmPromptTokens(chain, messages) + WARM_RETRY_HEADROOM_TOKENS <= inputBudget
  );
}

export function workerCanWarm(chain: KvWarmChain, model: Model<any>): boolean {
  const identity = modelIdentity(model);
  return identity !== undefined && sameModel(chain.capture.model, identity);
}

function getTokenField(
  payload: Record<string, unknown>,
): "max_tokens" | "max_completion_tokens" | undefined {
  if (typeof payload.max_tokens === "number") return "max_tokens";
  if (typeof payload.max_completion_tokens === "number") return "max_completion_tokens";
  return undefined;
}

export function buildWarmFetch(
  baseFetch: typeof fetch,
  capture: KvWarmCapture,
  messages: readonly unknown[],
  maxOutputTokens: number,
  onIntercept: () => void = () => {},
  onUnsupported: () => void = () => {},
): typeof fetch {
  return (input, init) => {
    onIntercept();
    if (!init || typeof init.body !== "string") {
      onUnsupported();
      return Promise.reject(new Error(WARM_UNSUPPORTED));
    }
    let outgoing: unknown;
    try {
      outgoing = JSON.parse(init.body);
    } catch {
      onUnsupported();
      return Promise.reject(new Error(WARM_UNSUPPORTED));
    }
    if (
      !isRecord(outgoing) ||
      !Array.isArray(outgoing.messages) ||
      outgoing.model !== capture.model.id
    ) {
      onUnsupported();
      return Promise.reject(new Error(WARM_UNSUPPORTED));
    }
    const tokenField = getTokenField(outgoing);
    if (!tokenField) {
      onUnsupported();
      return Promise.reject(new Error(WARM_UNSUPPORTED));
    }
    const body: Record<string, unknown> = {
      ...capture.payload,
      messages: [...messages],
      [tokenField]: maxOutputTokens,
    };
    delete body[tokenField === "max_tokens" ? "max_completion_tokens" : "max_tokens"];
    delete body.prompt_cache_key;
    delete body.prompt_cache_retention;
    for (const [field, value] of Object.entries(outgoing)) {
      if (
        REASONING_REQUEST_FIELD.test(field) ||
        field === "prompt_cache_key" ||
        field === "prompt_cache_retention"
      ) {
        body[field] = value;
      }
    }
    return baseFetch(input, { ...init, body: JSON.stringify(body) });
  };
}

function assistantText(message: unknown): { text: string; hasToolCall: boolean } {
  if (!isRecord(message) || !Array.isArray(message.content))
    return { text: "", hasToolCall: false };
  const text: string[] = [];
  let hasToolCall = false;
  for (const block of message.content) {
    if (!isRecord(block)) continue;
    if (block.type === "text" && typeof block.text === "string") text.push(block.text);
    if (block.type === "toolCall") {
      hasToolCall = true;
      text.push(
        `[tool call ${String(block.name ?? "unknown")}: ${JSON.stringify(block.arguments ?? {})}]`,
      );
    }
  }
  return { text: text.join(""), hasToolCall };
}

async function runWarmResponse(
  args: WarmCommonArgs,
  stage: WorkerStage,
  prompt: string,
  messages: unknown[],
): Promise<WarmResponse | undefined> {
  if (!workerCanWarm(args.chain, args.model)) return undefined;
  const maxOutputTokens = Math.floor(args.maxOutputTokens);
  if (!warmPromptFitsBudget(args.chain, prompt, maxOutputTokens, args.inputBudget, messages)) {
    return undefined;
  }

  let intercepted = false;
  let unsupported = false;
  const providerFetch = createProviderFetch(args.providerIdleTimeoutMs) ?? fetch;
  const bridge = createBridgeStreamFn(streamSimple, args.modelRegistry);
  const streamFn = neverThrow((model: any, context: any, options: any) =>
    bridge(model, context, {
      ...options,
      fetch: buildWarmFetch(
        options?.fetch ?? providerFetch,
        args.chain.capture,
        messages,
        maxOutputTokens,
        () => {
          intercepted = true;
        },
        () => {
          unsupported = true;
        },
      ),
    }),
  );

  const loop = agentLoop;
  const config: AgentLoopConfig & {
    fetch?: typeof fetch;
    sessionId?: string;
    cacheRetention?: CacheRetention;
  } = {
    model: args.model,
    apiKey: args.apiKey,
    headers: args.headers,
    env: args.env,
    sessionId: args.sessionId,
    cacheRetention: args.cacheRetention,
    fetch: providerFetch === fetch ? undefined : providerFetch,
    maxTokens: maxOutputTokens,
    // Warm workers have no executable tools; a tool call must not start an unbounded extra turn.
    finishTurn: () => ({ action: "end" }),
    convertToLlm: (items: any[]) => items as Message[],
    ...(args.model.reasoning && args.thinkingLevel !== "off"
      ? { reasoning: args.thinkingLevel ?? "low" }
      : {}),
  };
  const placeholder: Message[] = [
    { role: "user", content: [{ type: "text", text: prompt }], timestamp: Date.now() },
  ];
  const stream = loop(placeholder, buildAgentContext("", []), config, args.signal, streamFn);
  let messagesAtEnd: unknown[] = [];
  try {
    for await (const event of stream) {
      if (event.type === "agent_end") {
        messagesAtEnd = (event as { messages?: unknown[] }).messages ?? [];
      }
    }
    await stream.result();
  } catch (error) {
    if (args.signal?.aborted) throw args.signal.reason ?? error;
    throw new WorkerStreamError(workerStreamErrorMessage(stageName(stage), String(error)), 0);
  }

  if (unsupported || !intercepted) return undefined;
  const last = [...messagesAtEnd]
    .reverse()
    .find((message) => isRecord(message) && typeof message.stopReason === "string");
  if (!last || !isRecord(last)) {
    if (args.signal?.aborted) throw args.signal.reason ?? new Error("Worker attempt aborted");
    throw new WorkerStreamError(
      workerStreamErrorMessage(stageName(stage), "No final assistant response"),
      0,
    );
  }
  const extracted = assistantText(last);
  const stopReason = typeof last.stopReason === "string" ? last.stopReason : undefined;
  if (stopReason === "stop") {
    return {
      text: extracted.text,
      stopReason,
      errorMessage: typeof last.errorMessage === "string" ? last.errorMessage : undefined,
      hasToolCall: extracted.hasToolCall,
    };
  }
  if (args.signal?.aborted) throw args.signal.reason ?? new Error("Worker attempt aborted");
  if (stopReason === "error" || stopReason === "aborted") {
    if (stopReason === "aborted") {
      throw new WorkerStreamError(workerStreamErrorMessage(stageName(stage), "aborted"), 0);
    }
    throw new WorkerStreamError(
      workerStreamErrorMessage(stageName(stage), String(last.errorMessage ?? "Unknown API error")),
      0,
    );
  }
  return {
    text: extracted.text,
    stopReason,
    errorMessage: typeof last.errorMessage === "string" ? last.errorMessage : undefined,
    hasToolCall: extracted.hasToolCall || stopReason === "toolUse",
  };
}

function stageName(stage: WorkerStage): "Observer" | "Reflector" | "Dropper" {
  return stage === "observer" ? "Observer" : stage === "reflector" ? "Reflector" : "Dropper";
}

export function extractJsonObject(text: string): unknown {
  const fence = text.match(/```(?:json)?\s*([\s\S]*?)```/i);
  const source = (fence?.[1] ?? text).trim();
  for (const candidate of [
    source,
    source.slice(source.indexOf("{"), source.lastIndexOf("}") + 1),
  ]) {
    if (!candidate || candidate.startsWith("undefined")) continue;
    try {
      const parsed: unknown = JSON.parse(candidate);
      if (isRecord(parsed)) return parsed;
    } catch {
      // Try the fenced or outer-object candidate.
    }
  }
  return undefined;
}

async function runWarmJson<T>(
  args: WarmCommonArgs,
  stage: WorkerStage,
  prompt: string,
  validate: (value: unknown) => WarmValidation<T>,
): Promise<WarmValidation<T> | undefined> {
  const baseMessages = [...args.chain.messages];
  const turns: unknown[] = [];
  for (let attempt = 0; attempt < 2; attempt++) {
    const message = { role: "user", content: prompt };
    const messages = [...baseMessages, ...turns, message];
    const response = await runWarmResponse(args, stage, prompt, messages);
    if (!response) return undefined;

    const parsed =
      response.stopReason === "stop" && !response.hasToolCall
        ? validate(extractJsonObject(response.text))
        : {
            ok: false as const,
            error: response.hasToolCall
              ? "You emitted a tool call; return JSON only."
              : "The response was incomplete; return one complete JSON object.",
          };
    if (parsed.ok) {
      args.chain.messages = [...messages, { role: "assistant", content: response.text }];
      return parsed;
    }
    if (attempt > 0 || response.text.length > MAX_RETRY_REPLY_CHARS) return undefined;
    turns.push(
      { role: "assistant", content: response.text || "[empty response]" },
      {
        role: "user",
        content:
          `The previous response was invalid: ${parsed.error}. ` +
          `Correct every listed issue and return exactly one complete JSON object. Do not call tools.`,
      },
    );
  }
  return undefined;
}

function replaceRequired(source: string, anchor: string, replacement: string): string {
  const next = source.replace(anchor, replacement);
  if (next === source) throw new Error(`KV warm prompt anchor missing: ${anchor.slice(0, 64)}`);
  return next;
}

function observerWarmSystem(): string {
  let prompt = replaceRequired(
    OBSERVER_SYSTEM,
    "by calling the record_observations tool.",
    "by returning exactly one JSON object.",
  );
  const workflow = /How you work:\n[\s\S]*?\n\nWhat to emit:/;
  const warmWorkflow =
    "How you work:\n1. Read current reflections and observations.\n" +
    "2. Read the source-addressed chunk.\n" +
    "3. Return one complete JSON object containing all warranted observations, or an empty observations array if nothing is new.\n\nWhat to emit:";
  const next = prompt.replace(workflow, warmWorkflow);
  if (next === prompt) throw new Error("KV warm observer workflow anchor missing");
  prompt = replaceRequired(
    next,
    "close the run with a single record_observations call carrying an empty observations array and complete=true.",
    "return an empty observations array in the JSON object.",
  );
  return prompt;
}

function reflectorWarmSystem(timestamps: boolean, supersedes: boolean): string {
  let prompt = buildReflectorSystem({ timestamps, supersedes });
  prompt = replaceRequired(
    prompt,
    "by calling record_reflections.",
    "by returning one JSON object.",
  );
  prompt = replaceRequired(
    prompt,
    "Set complete=true only when the full active observation set has been reviewed and no further reflections remain. Set complete=false when another batch or correction is needed.",
    "Review the full supplied observation set and return one complete JSON object; use an empty reflections array when nothing new is durable.",
  );
  return replaceRequired(
    prompt,
    "in that case do not call the tool and reply briefly.",
    "in that case return an empty reflections array.",
  );
}

function dropperWarmSystem(): string {
  let prompt = replaceRequired(
    DROPPER_SYSTEM,
    "by calling drop_observations with their ids.",
    "by returning a JSON object containing only safe-to-drop ids.",
  );
  prompt = replaceRequired(
    prompt,
    "You can only call drop_observations with ids from the current observations list.",
    "You may return only ids from the current observations list.",
  );
  return replaceRequired(
    prompt,
    "do not call the tool and reply briefly.",
    "return an empty ids array.",
  );
}

function jsonOnly(prompt: string, contract: string): string {
  return `${prompt}\n\nDo not call any tools. Reply with exactly one JSON object and no markdown: ${contract}`;
}

function observerPrompt(args: WarmObserverArgs): string {
  const reflections = args.priorReflections.length
    ? args.priorReflections.join("\n")
    : "(none yet)";
  const observations = args.priorObservations.length
    ? args.priorObservations.join("\n")
    : "(none yet)";
  const prompt = [
    observerWarmSystem(),
    "CURRENT REFLECTIONS:",
    reflections,
    "CURRENT OBSERVATIONS:",
    observations,
    "Compress this new source-addressed conversation chunk into observations. Do not repeat facts already captured above.",
    "NEW CONVERSATION CHUNK:",
    args.chunk.trim(),
  ].join("\n\n");
  const supersedes = args.priorObservationIds ? ',"supersedes":["..."]' : "";
  return jsonOnly(
    prompt,
    `{"observations":[{"content":"...","relevance":"low|medium|high|critical","sourceEntryIds":["..."]${supersedes}}]}`,
  );
}

function reflectorPrompt(args: WarmReflectorArgs): string {
  const existingReflections = args.existingReflectionsSummary
    ? `EXISTING REFLECTIONS (for context only — do NOT re-process these):\n${args.existingReflectionsSummary}`
    : "";
  const existingObservations = args.existingObservationsSummary
    ? `EXISTING OBSERVATIONS (for context only — do NOT re-process these):\n${args.existingObservationsSummary}`
    : "";
  const prompt = [
    reflectorWarmSystem(args.timestampsEnabled === true, args.supersedesEnabled === true),
    existingReflections,
    existingObservations,
    "NEW REFLECTIONS TO PROCESS:",
    args.reflections.length
      ? args.reflections.map(args.renderReflection ?? reflectionToSummaryLine).join("\n")
      : "(none yet)",
    "NEW OBSERVATIONS TO PROCESS:",
    args.observations.length
      ? args.observations.map(observationToSummaryLine).join("\n")
      : "(none yet)",
    "Crystallize only missing durable facts or patterns. Return no reflections if nothing new is stable enough.",
  ]
    .filter(Boolean)
    .join("\n\n");
  const supersedes = args.supersedesEnabled ? ',"supersedes":["..."]' : "";
  return jsonOnly(
    prompt,
    `{"reflections":[{"content":"...","supportingObservationIds":["..."]${supersedes}}]}`,
  );
}

function dropperPrompt(args: WarmDropperArgs): string {
  const observationTokens =
    args.pressure?.tokens ??
    args.observations.reduce((sum, observation) => sum + observation.tokenCount, 0);
  const fullness = observationPoolFullness(observationTokens, args.budgetTokens);
  const urgency = dropUrgencyForFullness(fullness);
  const maxDropsAllowed =
    args.pressure?.maxDrops ??
    maxDropCountForPool(args.observations, observationTokens, args.budgetTokens, args.skipFullness);
  const coverageById = reflectionCoverageMap(args.observations, args.reflections);
  const existingObservations = args.existingObservationsSummary
    ? `EXISTING ACTIVE OBSERVATIONS (for context only — these are NOT candidates for dropping):\n${args.existingObservationsSummary}`
    : "";
  const fullnessPercent = Math.round(fullness * 100);
  const prompt = [
    dropperWarmSystem(),
    "CURRENT REFLECTIONS:",
    args.reflections.length
      ? args.reflections.map(args.renderReflection ?? reflectionToSummaryLine).join("\n")
      : "(none yet)",
    existingObservations,
    "NEW OBSERVATIONS TO EVALUATE FOR DROPPING:",
    args.observations.length
      ? args.observations
          .map((observation) =>
            observationToDropperLine(
              observation,
              coverageTierForObservation(observation, coverageById),
            ),
          )
          .join("\n")
      : "(none yet)",
    `Observation pool pressure: ~${observationTokens.toLocaleString()} tokens; target budget: ~${args.budgetTokens.toLocaleString()} tokens; fullness: ${fullnessPercent}%.`,
    `Drop urgency: ${urgency}.`,
    `Pool-wide maximum drops: ${maxDropsAllowed} (hard upper bound, not a target). Drop fewer or none if uncertain.`,
  ]
    .filter(Boolean)
    .join("\n\n");
  return jsonOnly(prompt, '{"ids":["..."]}');
}

export function validateWarmObserverOutput(
  value: unknown,
  args: WarmObserverArgs,
): WarmValidation<ObserverResult> {
  if (
    !isRecord(value) ||
    !Array.isArray(value.observations) ||
    value.complete === false ||
    (value.complete !== undefined && typeof value.complete !== "boolean")
  ) {
    return { ok: false, error: 'Expected {"observations":[...]} with a complete one-shot result.' };
  }
  const allowed = new Set(["low", "medium", "high", "critical"]);
  const accumulated = new Map<string, Observation>();
  let rejected = 0;
  let duplicates = 0;
  const supersedable = new Set(args.priorObservationIds ?? []);
  for (const proposal of value.observations) {
    if (
      !isRecord(proposal) ||
      typeof proposal.content !== "string" ||
      proposal.content.length === 0 ||
      typeof proposal.relevance !== "string" ||
      !allowed.has(proposal.relevance) ||
      !Array.isArray(proposal.sourceEntryIds) ||
      !proposal.sourceEntryIds.every((id) => typeof id === "string" && id.length > 0) ||
      (proposal.supersedes !== undefined &&
        (!Array.isArray(proposal.supersedes) ||
          !proposal.supersedes.every((id) => typeof id === "string" && id.length > 0)))
    ) {
      return {
        ok: false,
        error:
          "Every observation needs non-empty content, valid relevance, and string sourceEntryIds.",
      };
    }
    const sourceEntryIds = normalizeSourceEntryIds(
      proposal.sourceEntryIds,
      args.allowedSourceEntryIds,
    );
    if (!sourceEntryIds) {
      rejected++;
      continue;
    }
    const content = truncateRecordContent(proposal.content);
    const id = hashId(content);
    if (accumulated.has(id)) {
      duplicates++;
      continue;
    }
    const supersedes = [...new Set((proposal.supersedes as string[] | undefined) ?? [])].filter(
      (target) => target !== id && supersedable.has(target),
    );
    accumulated.set(id, {
      id,
      content,
      timestamp: deriveObservationTimestamp(sourceEntryIds, args.sourceEntryTimestamps),
      relevance: proposal.relevance as Relevance,
      sourceEntryIds,
      tokenCount: estimateStringTokens(content),
      ...(supersedes.length ? { supersedes } : {}),
    });
  }
  if (rejected > 0)
    return { ok: false, error: `${rejected} observation(s) had no valid sourceEntryIds.` };
  if (accumulated.size > 0)
    return { ok: true, value: { observations: Array.from(accumulated.values()) } };
  return {
    ok: true,
    value: {
      observations: undefined,
      emptyReason:
        duplicates > 0 ? { kind: "all_duplicates", count: duplicates } : { kind: "no_new_content" },
      coverageComplete: true,
    },
  };
}

export function validateWarmReflectorOutput(
  value: unknown,
  args: WarmReflectorArgs,
): WarmValidation<ReflectorResult> {
  if (
    !isRecord(value) ||
    !Array.isArray(value.reflections) ||
    value.complete === false ||
    (value.complete !== undefined && typeof value.complete !== "boolean")
  ) {
    return { ok: false, error: 'Expected {"reflections":[...]}.' };
  }
  const allowedIds = args.observations.map((observation) => observation.id);
  const existingIds = new Set(args.reflections.map((reflection) => reflection.id));
  const shownIds = new Set([...existingIds, ...(args.supersedableReflectionIds ?? [])]);
  const accumulated = new Map<string, Reflection>();
  let rejected = 0;
  for (const proposal of value.reflections) {
    if (
      !isRecord(proposal) ||
      typeof proposal.content !== "string" ||
      !Array.isArray(proposal.supportingObservationIds) ||
      !proposal.supportingObservationIds.every((id) => typeof id === "string" && id.length > 0) ||
      (proposal.supersedes !== undefined &&
        (!args.supersedesEnabled ||
          !Array.isArray(proposal.supersedes) ||
          !proposal.supersedes.every((id) => typeof id === "string" && id.length > 0)))
    ) {
      return {
        ok: false,
        error: "Every reflection needs content and string supportingObservationIds.",
      };
    }
    const content = normalizeReflectionContent(proposal.content);
    const supportingObservationIds = normalizeSupportingObservationIds(
      proposal.supportingObservationIds,
      allowedIds,
    );
    if (!content || !supportingObservationIds) {
      rejected++;
      continue;
    }
    const id = hashId(content);
    if (existingIds.has(id) || accumulated.has(id)) continue;
    const timestamp = args.observations
      .filter((observation) => supportingObservationIds.includes(observation.id))
      .map((observation) => observation.timestamp)
      .filter(isReflectionTimestamp)
      .sort()
      .at(-1);
    const supersedes: string[] = [];
    if (args.supersedesEnabled && Array.isArray(proposal.supersedes)) {
      for (const target of proposal.supersedes) {
        if (
          !isMemoryId(target) ||
          target === id ||
          !shownIds.has(target) ||
          accumulated.has(target)
        )
          continue;
        if (!supersedes.includes(target)) supersedes.push(target);
      }
    }
    accumulated.set(id, {
      ...(timestamp ? { timestamp } : {}),
      ...(supersedes.length ? { supersedes } : {}),
      id,
      content,
      supportingObservationIds,
      tokenCount: estimateStringTokens(content),
    });
  }
  if (rejected > 0)
    return { ok: false, error: `${rejected} reflection(s) had invalid content or supporting ids.` };
  return {
    ok: true,
    value: { reflections: accumulated.size ? Array.from(accumulated.values()) : undefined },
  };
}

export function validateWarmDropperOutput(
  value: unknown,
  args: WarmDropperArgs,
): WarmValidation<string[] | undefined> {
  if (
    !isRecord(value) ||
    !Array.isArray(value.ids) ||
    !value.ids.every((id) => typeof id === "string" && id.length > 0) ||
    (value.reason !== undefined && typeof value.reason !== "string")
  ) {
    return { ok: false, error: 'Expected {"ids":[string,...]} with an optional string reason.' };
  }
  const proposals = normalizeDropObservationIds(value.ids, args.observations) ?? [];
  if (args.rawProposals) return { ok: true, value: proposals.length ? proposals : undefined };
  const observationTokens =
    args.pressure?.tokens ??
    args.observations.reduce((sum, observation) => sum + observation.tokenCount, 0);
  const maxDropsAllowed =
    args.pressure?.maxDrops ??
    maxDropCountForPool(args.observations, observationTokens, args.budgetTokens, args.skipFullness);
  const selected = selectDropCandidates(
    proposals,
    args.observations,
    maxDropsAllowed,
    args.reflections,
  );
  return { ok: true, value: selected.length ? selected : undefined };
}

export async function runWarmObserver(args: WarmObserverArgs): Promise<ObserverResult | undefined> {
  const result = await runWarmJson(args, "observer", observerPrompt(args), (output) =>
    validateWarmObserverOutput(output, args),
  );
  return result?.ok ? result.value : undefined;
}

export async function runWarmReflector(
  args: WarmReflectorArgs,
): Promise<ReflectorResult | undefined> {
  const result = await runWarmJson(args, "reflector", reflectorPrompt(args), (output) =>
    validateWarmReflectorOutput(output, args),
  );
  return result?.ok ? result.value : undefined;
}

export async function runWarmDropper(
  args: WarmDropperArgs,
): Promise<WarmValidation<string[] | undefined> | undefined> {
  return runWarmJson(args, "dropper", dropperPrompt(args), (output) =>
    validateWarmDropperOutput(output, args),
  );
}
