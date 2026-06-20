/**
 * Express server for the AI Model Router web demo.
 *
 * Endpoints:
 *   POST /api/route   — classify prompt, select model, call it, return everything
 *   POST /api/classify — classify only (no model call)
 *   GET  /api/models  — return model registry
 *   GET  /             — serve the frontend
 */

import express from "express";
import path from "path";
import dotenv from "dotenv";
import OpenAI from "openai";

import { MODELS, TASK_TYPES, getBestModelForTask, getRankedModels, type TaskType, type ModelInfo } from "./models";
import { classifyByRules, type ClassificationResult } from "./classifier";

dotenv.config({ path: path.resolve(__dirname, "../../.env") });

const app = express();
app.use(express.json({ limit: '32kb' }));
app.use(express.static(path.join(__dirname, "../public")));

const PORT = process.env.PORT ?? 3000;

// --- OpenRouter client ---

function getClient(): OpenAI {
  const apiKey = process.env.OPENROUTER_API_KEY;
  if (!apiKey) {
    throw new Error("OPENROUTER_API_KEY not set");
  }
  return new OpenAI({ baseURL: "https://openrouter.ai/api/v1", apiKey });
}

// --- API endpoints ---

app.get("/api/models", (_req, res) => {
  const models = Object.values(MODELS).map((m) => ({
    ...m,
    scoresByTask: m.scores,
  }));
  res.json({ models, taskTypes: TASK_TYPES });
});

app.post("/api/classify", (req, res) => {
  const { prompt } = req.body;
  if (!prompt || typeof prompt !== "string") {
    res.status(400).json({ error: "Missing 'prompt' in request body" });
    return;
  }

  const classification = classifyByRules(prompt);
  const ranked = getRankedModels(classification.taskType);
  const best = getBestModelForTask(classification.taskType);

  res.json({
    classification,
    selectedModel: best,
    alternatives: ranked.filter((r) => r.model.id !== best.id).slice(0, 3),
  });
});

app.post("/api/route", async (req, res) => {
  const { prompt, budget = "best" } = req.body;
  if (!prompt || typeof prompt !== "string") {
    res.status(400).json({ error: "Missing 'prompt' in request body" });
    return;
  }

  // Step 1: Classify
  const classification = classifyByRules(prompt);

  // Step 2: Select model
  const validBudget = (["best", "balanced", "cheap"] as const).includes(budget) ? budget : "best";
  const model = getBestModelForTask(classification.taskType, validBudget as "best" | "balanced" | "cheap");
  const ranked = getRankedModels(classification.taskType);
  const score = model.scores[classification.taskType] ?? 0;

  // Step 3: Call the model
  let content: string;
  let latencyMs: number;
  try {
    const client = getClient();
    const start = performance.now();
    const response = await client.chat.completions.create({
      model: model.id,
      messages: [{ role: "user", content: prompt }],
      max_tokens: 1024,
    });
    latencyMs = Math.round(performance.now() - start);
    content = response.choices[0]?.message?.content ?? "";
  } catch (err: any) {
    res.status(502).json({
      error: "Model call failed",
      classification,
      selectedModel: model,
    });
    return;
  }

  // Estimate cost
  const estInputTokens = prompt.length / 4;
  const estOutputTokens = content.length / 4;
  const estimatedCost =
    (estInputTokens / 1_000_000) * model.costPerMillionInput +
    (estOutputTokens / 1_000_000) * model.costPerMillionOutput;

  res.json({
    content,
    classification,
    decision: {
      model,
      taskType: classification.taskType,
      score,
      reasoning: buildReasoning(classification, model, score, validBudget),
      alternatives: ranked
        .filter((r) => r.model.id !== model.id)
        .slice(0, 3)
        .map((r) => ({ model: r.model, score: r.score })),
    },
    latencyMs,
    estimatedCost: Math.round(estimatedCost * 1_000_000) / 1_000_000,
  });
});

function buildReasoning(
  classification: ClassificationResult,
  model: ModelInfo,
  score: number,
  budget: string,
): string {
  const parts = [
    `Classified as '${classification.taskType}' (confidence: ${classification.confidence}, keywords: [${classification.keywordsMatched.join(", ")}])`,
    `Selected ${model.name} with score ${score}/10 (budget: ${budget})`,
  ];
  return parts.join(". ") + ".";
}

// --- Fallback: serve index.html for SPA ---

app.get("*", (_req, res) => {
  res.sendFile(path.join(__dirname, "../public/index.html"));
});

// --- Start ---

const HOST = process.env.HOST ?? '127.0.0.1';
app.listen(Number(PORT), HOST, () => {
  if (HOST !== '127.0.0.1' && HOST !== 'localhost') {
    console.warn(
      `[WARNING] Server bound to ${HOST} — this exposes unauthenticated model calls to the network.`
    );
  }
  console.log(`AI Model Router web demo running at http://${HOST}:${PORT}`);
});
