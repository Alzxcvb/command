/** AI Model Router — Frontend App */

const messagesEl = document.getElementById("messages");
const form = document.getElementById("chat-form");
const input = document.getElementById("prompt-input");
const budgetSelect = document.getElementById("budget-select");
const sendBtn = document.getElementById("send-btn");
const sideEmpty = document.getElementById("side-empty");
const sideContent = document.getElementById("side-content");

// Example prompt buttons
document.querySelectorAll(".example-btn").forEach((btn) => {
  btn.addEventListener("click", () => {
    input.value = btn.dataset.prompt;
    input.focus();
  });
});

// Submit on Enter (Shift+Enter for newline)
input.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    form.dispatchEvent(new Event("submit"));
  }
});

form.addEventListener("submit", async (e) => {
  e.preventDefault();
  const prompt = input.value.trim();
  if (!prompt) return;

  // Add user message
  addMessage(prompt, "user");
  input.value = "";
  sendBtn.disabled = true;

  // Add loading message
  const loadingEl = addMessage("Classifying and routing...", "assistant loading");

  try {
    const res = await fetch("/api/route", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ prompt, budget: budgetSelect.value }),
    });

    const data = await res.json();

    // Remove loading message
    loadingEl.remove();

    if (!res.ok) {
      addMessage(`Error: ${data.error || "Unknown error"}\n${data.detail || ""}`, "assistant");
      updateSidePanel(data);
      sendBtn.disabled = false;
      return;
    }

    // Add response with model tag
    const msgEl = addMessage("", "assistant");
    const tagEl = document.createElement("div");
    tagEl.className = "model-tag";
    tagEl.textContent = `${data.decision.model.name} · ${data.decision.taskType} · ${data.latencyMs}ms`;
    msgEl.prepend(tagEl);

    const textEl = document.createElement("div");
    textEl.textContent = data.content;
    msgEl.appendChild(textEl);

    // Update side panel
    updateSidePanel(data);
  } catch (err) {
    loadingEl.remove();
    addMessage(`Network error: ${err.message}`, "assistant");
  }

  sendBtn.disabled = false;
  input.focus();
});

function addMessage(text, className) {
  const el = document.createElement("div");
  el.className = `message ${className}`;
  el.textContent = text;
  messagesEl.appendChild(el);
  messagesEl.scrollTop = messagesEl.scrollHeight;
  return el;
}

function updateSidePanel(data) {
  sideEmpty.style.display = "none";
  sideContent.style.display = "block";

  const classification = data.classification;
  const decision = data.decision;

  if (!classification) return;

  // Decision card
  const taskTypeEl = document.getElementById("d-task-type");
  taskTypeEl.textContent = classification.taskType;
  taskTypeEl.style.background = getTaskColor(classification.taskType);

  document.getElementById("d-confidence").textContent =
    `${Math.round(classification.confidence * 100)}%`;
  document.getElementById("d-keywords").textContent =
    classification.keywordsMatched?.join(", ") || "(none)";

  if (!decision) return;

  // Selected model
  document.getElementById("d-model-name").textContent = decision.model.name;
  document.getElementById("d-model-score").textContent =
    `Score: ${decision.score}/10`;
  document.getElementById("d-model-cost").textContent =
    `$${decision.model.costPerMillionInput}/M in`;

  document.getElementById("d-latency").textContent =
    data.latencyMs ? `Latency: ${data.latencyMs}ms` : "";
  document.getElementById("d-est-cost").textContent =
    data.estimatedCost != null ? `Est: $${data.estimatedCost.toFixed(6)}` : "";

  // Reasoning
  document.getElementById("d-reasoning").textContent = decision.reasoning;

  // Alternatives
  const altEl = document.getElementById("d-alternatives");
  altEl.innerHTML = "";
  for (const alt of decision.alternatives || []) {
    const row = document.createElement("div");
    row.className = "alt-model";
    const nameSpan = document.createElement("span");
    nameSpan.className = "alt-name";
    nameSpan.textContent = alt.model.name;
    const scoreSpan = document.createElement("span");
    scoreSpan.className = "alt-score";
    scoreSpan.textContent = `${alt.score}/10 · $${alt.model.costPerMillionInput}/M`;
    row.appendChild(nameSpan);
    row.appendChild(scoreSpan);
    altEl.appendChild(row);
  }

  // Score chart
  document.getElementById("d-task-label").textContent = classification.taskType;
  buildScoreChart(classification.taskType, decision.model.id);
}

function buildScoreChart(taskType, selectedModelId) {
  const chartEl = document.getElementById("d-score-chart");
  chartEl.innerHTML = "";

  // We'll fetch models from the API to build the chart
  fetch("/api/models")
    .then((r) => r.json())
    .then((data) => {
      const models = data.models
        .map((m) => ({ name: m.name, id: m.id, score: m.scores[taskType] || 0 }))
        .sort((a, b) => b.score - a.score);

      for (const m of models) {
        const row = document.createElement("div");
        row.className = "score-bar-row";

        const pct = (m.score / 10) * 100;
        const color = m.id === selectedModelId ? "var(--accent)" : scoreColor(m.score);

        const labelSpan = document.createElement("span");
        labelSpan.className = "score-bar-label";
        labelSpan.textContent = m.name;

        const track = document.createElement("div");
        track.className = "score-bar-track";
        const fill = document.createElement("div");
        fill.className = "score-bar-fill";
        const safePct = Math.min(100, Math.max(0, pct));
        fill.style.width = `${safePct}%`;
        fill.style.background = color;
        track.appendChild(fill);

        const valueSpan = document.createElement("span");
        valueSpan.className = "score-bar-value";
        valueSpan.textContent = String(m.score);

        row.appendChild(labelSpan);
        row.appendChild(track);
        row.appendChild(valueSpan);
        chartEl.appendChild(row);
      }
    });
}

function scoreColor(score) {
  if (score >= 9) return "var(--green)";
  if (score >= 8) return "var(--yellow)";
  if (score >= 7) return "var(--orange)";
  return "var(--red)";
}

function getTaskColor(taskType) {
  const colors = {
    code: "#6c8aff",
    writing: "#a78bfa",
    reasoning: "#facc15",
    summarization: "#4ade80",
    conversation: "#38bdf8",
    research: "#fb923c",
    translation: "#f472b6",
    data: "#2dd4bf",
  };
  return colors[taskType] || "var(--accent)";
}
