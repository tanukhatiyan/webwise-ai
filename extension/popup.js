/**
 * WebWise AI — Chrome Extension Popup Controller
 * ===============================================
 * Handles webpage text extraction, backend communication,
 * Server-Sent Events (SSE) streaming, long-webpage chunking,
 * state persistence, detached pop-out window, and PDF export.
 */

// ── Backend Configuration ──────────────────────────────────────────────────
const BACKEND_URL = "http://127.0.0.1:7864";
const HEALTH_URL = `${BACKEND_URL}/health`;
const SUMMARIZE_URL = `${BACKEND_URL}/summarize_stream_status`;

// ── Chunking & Safety Limits ───────────────────────────────────────────────
// Backend max limit is 50,000 characters. We chunk at 35,000 chars leaving
// a 15,000 character safety margin to guarantee we never violate the backend schema.
const CHUNK_MAX_CHARS = 35000;
const SINGLE_PASS_LIMIT = 40000;
const MAX_PROCESSABLE_CHUNKS = 8; // Max ~280,000 chars to prevent overloading local LLM

// ── Task Mode Button Labels ────────────────────────────────────────────────
const TASK_LABELS = {
  summarize: "Summarize",
  key_points: "Extract Key Points",
  explain_simply: "Explain Simply",
  generate_questions: "Generate Questions",
};

// ── DOM Elements ───────────────────────────────────────────────────────────
const statusDot = document.getElementById("status-dot");
const pageTitleEl = document.getElementById("page-title");
const providerSelect = document.getElementById("provider-select");
const providerBadge = document.getElementById("provider-badge");
const taskSelect = document.getElementById("task-select");
const btnSummarize = document.getElementById("btn-summarize");
const btnCancel = document.getElementById("btn-cancel");
const btnPopout = document.getElementById("btn-popout");
const btnText = btnSummarize.querySelector(".btn-text");
const btnLoading = btnSummarize.querySelector(".btn-loading");
const outputSection = document.getElementById("output-section");
const outputText = document.getElementById("output-text");
const btnCopy = document.getElementById("btn-copy");
const btnPdf = document.getElementById("btn-pdf");
const btnClear = document.getElementById("btn-clear");
const errorBanner = document.getElementById("error-banner");
const errorMessage = document.getElementById("error-message");
const copyToast = document.getElementById("copy-toast");

// ── Application State ──────────────────────────────────────────────────────
let isBackendReady = false;
let isAnalyzing = false;
let isGeneratingPdf = false;
let abortController = null;
let currentTabUrl = "";
let extractedContent = "";
let extractedTitle = "";
let toastTimer = null;
let currentHealthData = null;

// ── Helper: Error Banner ───────────────────────────────────────────────────
function showError(msg) {
  errorMessage.textContent = msg;
  errorBanner.hidden = false;
}

function hideError() {
  errorBanner.hidden = true;
  errorMessage.textContent = "";
}

// ── Helper: Parse HTTP Errors ──────────────────────────────────────────────
function formatErrorMessage(errJson, status) {
  if (errJson && errJson.detail) {
    if (Array.isArray(errJson.detail)) {
      // Pydantic validation error array
      const messages = errJson.detail.map((d) => {
        if (d.type === "string_too_long") {
          return "Content exceeds maximum limit of 50,000 characters.";
        }
        const field = d.loc ? d.loc.filter((x) => x !== "body").join(".") : "field";
        return `${field ? field + ": " : ""}${d.msg || d.type}`;
      });
      return messages.join("; ");
    }
    if (typeof errJson.detail === "string") {
      return errJson.detail;
    }
  }
  return `Server returned error (${status})`;
}

// ── Helper: Copy/Action Toast ──────────────────────────────────────────────
function showToast(text = "Copied!") {
  copyToast.textContent = text;
  copyToast.hidden = false;
  if (toastTimer) clearTimeout(toastTimer);
  toastTimer = setTimeout(() => {
    copyToast.hidden = true;
  }, 2200);
}

// ── Helper: Update Button Label ────────────────────────────────────────────
function updateButtonLabel() {
  const selectedTask = taskSelect.value;
  btnText.textContent = TASK_LABELS[selectedTask] || "Analyze";
}

// ── Helper: Smart Scroll (Prevents Jumping during Selection) ───────────────
function smartScroll(element) {
  if (!element) return;
  // Only auto-scroll if the user is already near the bottom (within 70px)
  const isNearBottom = element.scrollHeight - element.scrollTop - element.clientHeight < 70;
  if (isNearBottom) {
    element.scrollTop = element.scrollHeight;
  }
}

// ── Helper: Filename Sanitizer ─────────────────────────────────────────────
function sanitizeFilename(title) {
  if (!title || typeof title !== "string") return "webwise-summary.pdf";
  let slug = title
    .toLowerCase()
    .replace(/[^\w\s-]/g, "")
    .trim()
    .replace(/[\s_]+/g, "-")
    .replace(/-+/g, "-");

  if (!slug || slug === "-") slug = "webwise-summary";
  if (slug.length > 50) slug = slug.slice(0, 50).replace(/-+$/, "");
  return `${slug}-summary.pdf`;
}

// ── State Persistence (Preserve Result Across Popups) ──────────────────────
async function saveAnalysisState(summaryText) {
  if (!summaryText || !summaryText.trim()) return;
  try {
    if (chrome.storage && chrome.storage.local) {
      await chrome.storage.local.set({
        lastAnalysis: {
          url: currentTabUrl,
          title: extractedTitle,
          content: extractedContent,
          summary: summaryText.trim(),
          task: taskSelect.value,
          timestamp: Date.now(),
        },
      });
    }
  } catch (e) {
    console.warn("Could not save analysis state:", e);
  }
}

async function restoreSavedAnalysis(activeUrl) {
  try {
    if (!chrome.storage || !chrome.storage.local) return;
    const data = await chrome.storage.local.get("lastAnalysis");
    if (data && data.lastAnalysis) {
      const saved = data.lastAnalysis;
      const isRecent = Date.now() - (saved.timestamp || 0) < 4 * 60 * 60 * 1000; // 4 hours
      // Restore if on the same page or if generated recently
      if (saved.summary && (saved.url === activeUrl || isRecent)) {
        outputText.textContent = saved.summary;
        outputSection.hidden = false;
        if (saved.task && taskSelect) {
          taskSelect.value = saved.task;
          updateButtonLabel();
        }
        if (saved.title && !extractedTitle) {
          extractedTitle = saved.title;
        }
      }
    }
  } catch (e) {
    console.warn("Could not restore saved analysis:", e);
  }
}

// ── Text Chunking Utility (Paragraph & Sentence Aware) ─────────────────────
function splitTextIntoChunks(text, maxChars = CHUNK_MAX_CHARS) {
  text = text.trim();
  if (!text) return [];
  if (text.length <= maxChars) return [text];

  const chunks = [];
  const paragraphs = text.split(/\n\s*\n/);
  let currentChunk = [];
  let currentLen = 0;

  for (let para of paragraphs) {
    para = para.trim();
    if (!para) continue;

    // If a single paragraph is longer than maxChars, split by sentences
    if (para.length > maxChars) {
      if (currentChunk.length > 0) {
        chunks.push(currentChunk.join("\n\n"));
        currentChunk = [];
        currentLen = 0;
      }

      const sentences = para.split(/(?<=[.!?])\s+/);
      let sentChunk = [];
      let sentLen = 0;

      for (let sent of sentences) {
        sent = sent.trim();
        if (!sent) continue;

        if (sent.length > maxChars) {
          if (sentChunk.length > 0) {
            chunks.push(sentChunk.join(" "));
            sentChunk = [];
            sentLen = 0;
          }
          const words = sent.split(/\s+/);
          let wordChunk = [];
          let wordLen = 0;
          for (let w of words) {
            if (wordLen + w.length + 1 > maxChars && wordChunk.length > 0) {
              chunks.push(wordChunk.join(" "));
              wordChunk = [w];
              wordLen = w.length;
            } else {
              wordChunk.push(w);
              wordLen += w.length + 1;
            }
          }
          if (wordChunk.length > 0) {
            chunks.push(wordChunk.join(" "));
          }
        } else if (sentLen + sent.length + 1 > maxChars && sentChunk.length > 0) {
          chunks.push(sentChunk.join(" "));
          sentChunk = [sent];
          sentLen = sent.length;
        } else {
          sentChunk.push(sent);
          sentLen += sent.length + 1;
        }
      }

      if (sentChunk.length > 0) {
        chunks.push(sentChunk.join(" "));
      }
      continue;
    }

    const addedLen = currentLen === 0 ? para.length : para.length + 2;
    if (currentLen + addedLen > maxChars && currentChunk.length > 0) {
      chunks.push(currentChunk.join("\n\n"));
      currentChunk = [para];
      currentLen = para.length;
    } else {
      currentChunk.push(para);
      currentLen += addedLen;
    }
  }

  if (currentChunk.length > 0) {
    chunks.push(currentChunk.join("\n\n"));
  }

  return chunks;
}

// ── In-Tab Content Extraction Function ─────────────────────────────────────
function extractPageContent() {
  try {
    const title = document.title || document.querySelector("h1")?.innerText || "";

    const candidates = [
      document.querySelector("article"),
      document.querySelector("main"),
      document.querySelector('[role="main"]'),
      document.querySelector(".article-content"),
      document.querySelector(".post-content"),
      document.querySelector("#content"),
      document.body,
    ];

    let targetElement = null;
    for (const el of candidates) {
      if (el && el.innerText && el.innerText.trim().length > 100) {
        targetElement = el;
        break;
      }
    }

    if (!targetElement) {
      targetElement = document.body;
    }

    if (!targetElement) {
      return { success: false, title: "", content: "", error: "No document body found." };
    }

    // Clone element to avoid modifying the live DOM
    const clone = targetElement.cloneNode(true);

    // Remove noise elements
    const unwantedSelectors = [
      "script",
      "style",
      "noscript",
      "iframe",
      "svg",
      "canvas",
      "nav",
      "header",
      "footer",
      "aside",
      "form",
      "button",
      "dialog",
      '[aria-hidden="true"]',
      ".ad",
      ".ads",
      ".advertisement",
      ".sidebar",
      ".comments",
      ".comment-list",
      ".nav",
      ".menu",
      // Wikipedia-specific noise
      ".reflist",
      ".mw-references-wrap",
      ".reference",
      ".mw-editsection",
      ".navbox",
      ".vertical-navbox",
      ".catlinks",
      ".infobox",
      ".thumbcaption",
      "#toc",
      ".toc",
      ".vector-header",
      ".vector-page-toolbar",
      ".mw-footer",
      ".noprint",
    ];

    clone.querySelectorAll(unwantedSelectors.join(", ")).forEach((el) => el.remove());

    let rawText = clone.innerText || clone.textContent || "";

    // Clean citation markers like [1], [24], [citation needed], [edit]
    let cleanText = rawText
      .replace(/\[\d+\]/g, "")
      .replace(/\[(?:edit|citation needed)\]/gi, "")
      .replace(/\r\n/g, "\n")
      .replace(/[ \t]+/g, " ")
      .replace(/\n\s*\n\s*\n+/g, "\n\n")
      .trim();

    return {
      success: true,
      title: title.trim(),
      content: cleanText,
      url: window.location.href,
    };
  } catch (err) {
    return {
      success: false,
      title: document.title || "",
      content: "",
      error: err.message || "Failed to extract content",
    };
  }
}

// ── Step 1: Health Check Backend & Multi-Provider Readiness ───────────────
function updateProviderReadiness() {
  const provider = providerSelect ? providerSelect.value : "ollama";

  if (providerBadge) {
    if (provider === "groq") {
      providerBadge.textContent = "Cloud";
      providerBadge.className = "badge badge-cloud";
    } else {
      providerBadge.textContent = "Local";
      providerBadge.className = "badge";
    }
  }

  if (!currentHealthData) {
    statusDot.className = "status-dot status-offline";
    statusDot.title = "Backend server is offline (127.0.0.1:7864)";
    isBackendReady = false;
    btnSummarize.disabled = true;
    showError("Backend is offline. Start it: python run.py");
    return;
  }

  const data = currentHealthData;

  if (provider === "groq") {
    if (data.api_healthy && data.groq_configured) {
      statusDot.className = "status-dot status-ready";
      statusDot.title = `Backend Ready | Groq Cloud: ${data.groq_model || "llama-3.3-70b-versatile"}`;
      isBackendReady = true;
      hideError();
      if (extractedContent && extractedContent.length >= 20) {
        btnSummarize.disabled = false;
      }
    } else {
      statusDot.className = "status-dot status-warning";
      statusDot.title = "Groq Cloud API key is not configured on backend";
      isBackendReady = false;
      btnSummarize.disabled = true;
      showError("Groq Cloud is not configured. Please add GROQ_API_KEY to your backend .env file.");
    }
  } else {
    // Local Ollama
    if (data.api_healthy && data.ollama_reachable && data.model_available) {
      statusDot.className = "status-dot status-ready";
      statusDot.title = `Backend Ready | Ollama: ${data.model_name}`;
      isBackendReady = true;
      hideError();
      if (extractedContent && extractedContent.length >= 20) {
        btnSummarize.disabled = false;
      }
    } else if (!data.ollama_reachable) {
      statusDot.className = "status-dot status-error";
      statusDot.title = "Ollama is not running (127.0.0.1:11434)";
      isBackendReady = false;
      btnSummarize.disabled = true;
      showError("Ollama is not running. Please start it: ollama serve");
    } else if (!data.model_available) {
      statusDot.className = "status-dot status-warning";
      statusDot.title = `Model '${data.model_name}' not pulled in Ollama`;
      isBackendReady = false;
      btnSummarize.disabled = true;
      showError(`Model '${data.model_name}' missing. Run: ollama pull ${data.model_name}`);
    }
  }
}

async function checkHealth() {
  statusDot.className = "status-dot";
  statusDot.title = "Checking backend status...";

  try {
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 3500);

    const response = await fetch(HEALTH_URL, {
      signal: controller.signal,
      cache: "no-store",
    });
    clearTimeout(timeoutId);

    if (!response.ok) {
      throw new Error(`HTTP ${response.status}`);
    }

    const data = await response.json();
    currentHealthData = data;

    // Dynamically update provider options with live server state
    if (providerSelect) {
      const ollamaOpt = providerSelect.querySelector('option[value="ollama"]');
      const groqOpt = providerSelect.querySelector('option[value="groq"]');

      if (ollamaOpt) {
        if (data.ollama_reachable && data.model_available) {
          ollamaOpt.textContent = `🖥️ Local Ollama (${data.model_name})`;
        } else if (!data.ollama_reachable) {
          ollamaOpt.textContent = "🖥️ Local Ollama (Offline)";
        } else {
          ollamaOpt.textContent = `🖥️ Local Ollama (${data.model_name} missing)`;
        }
      }

      if (groqOpt) {
        if (data.groq_configured) {
          groqOpt.textContent = `⚡ Groq Cloud (${data.groq_model || "Fast"})`;
        } else {
          groqOpt.textContent = "⚡ Groq Cloud (API Key Not Set)";
        }
      }
    }

    updateProviderReadiness();
  } catch (err) {
    currentHealthData = null;
    statusDot.className = "status-dot status-offline";
    statusDot.title = "Backend server is offline (127.0.0.1:7864)";
    isBackendReady = false;
    btnSummarize.disabled = true;
    showError("Backend is offline. Start it: python run.py");
  }
}

// ── Step 2: Initialize Active Tab & Extract Content ────────────────────────
async function initActiveTab() {
  pageTitleEl.textContent = "Detecting active tab...";
  btnSummarize.disabled = true;

  try {
    const isDetached = window.location.search.includes("detached=true");
    let tabs = [];

    if (isDetached) {
      // In detached window, inspect the active tab of the last focused browser window
      tabs = await chrome.tabs.query({ active: true, lastFocusedWindow: true });
      if (!tabs || tabs.length === 0 || (tabs[0].url && tabs[0].url.startsWith("chrome-extension://"))) {
        tabs = await chrome.tabs.query({ active: true });
        tabs = tabs.filter((t) => t.url && !t.url.startsWith("chrome-extension://"));
      }
    } else {
      tabs = await chrome.tabs.query({ active: true, currentWindow: true });
    }

    if (!tabs || tabs.length === 0) {
      pageTitleEl.textContent = "No active tab found";
      showError("Unable to access current tab.");
      return;
    }

    const tab = tabs[0];
    const url = tab.url || "";
    currentTabUrl = url;

    // Check for restricted / system URLs
    const isRestricted =
      url.startsWith("chrome://") ||
      url.startsWith("chrome-extension://") ||
      url.startsWith("edge://") ||
      url.startsWith("about:") ||
      url.startsWith("view-source:") ||
      url.startsWith("devtools://") ||
      url.includes("chromewebstore.google.com");

    if (isRestricted) {
      pageTitleEl.textContent = "System / Internal Page";
      showError("WebWise AI cannot run on browser system pages. Please navigate to a webpage or article.");
      btnSummarize.disabled = true;
      return;
    }

    pageTitleEl.textContent = tab.title || url;

    // Attempt restoring any previously generated analysis for this page
    await restoreSavedAnalysis(url);

    // Inject content extraction script
    const results = await chrome.scripting.executeScript({
      target: { tabId: tab.id },
      func: extractPageContent,
    });

    if (!results || !results[0] || !results[0].result) {
      showError("Could not inspect page content. Please refresh the page and try again.");
      return;
    }

    const res = results[0].result;
    if (!res.success || !res.content || res.content.length < 20) {
      extractedContent = "";
      extractedTitle = tab.title || "";
      showError("No readable article text found on this page (page may be empty or dynamically rendered).");
      btnSummarize.disabled = true;
      return;
    }

    extractedContent = res.content;
    extractedTitle = res.title || tab.title || "";

    // Display title with page size info if long
    if (extractedContent.length > SINGLE_PASS_LIMIT) {
      const kChars = (extractedContent.length / 1000).toFixed(1);
      pageTitleEl.textContent = `${tab.title || url} [~${kChars}k chars]`;
      pageTitleEl.title = `Extracted ${extractedContent.length.toLocaleString()} characters (will use sectioned analysis)`;
    } else {
      pageTitleEl.textContent = tab.title || url;
      pageTitleEl.title = `Extracted ${extractedContent.length.toLocaleString()} characters`;
    }

    if (isBackendReady) {
      btnSummarize.disabled = false;
      hideError();
    }
  } catch (err) {
    pageTitleEl.textContent = "Access Restricted";
    showError(`Could not access tab: ${err.message || err}`);
    btnSummarize.disabled = true;
  }
}

// ── Step 3: Stream a Single Request to the Backend ─────────────────────────
async function streamSummarizeRequest(content, task, title, onChunk, signal) {
  const currentProvider = providerSelect ? providerSelect.value : undefined;
  const payload = {
    content: content,
    task: task,
    title: title || undefined,
    provider: currentProvider,
  };

  const response = await fetch(SUMMARIZE_URL, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Accept: "text/event-stream",
    },
    body: JSON.stringify(payload),
    signal: signal,
  });

  if (!response.ok) {
    let errJson = null;
    try {
      errJson = await response.json();
    } catch (_) {}
    throw new Error(formatErrorMessage(errJson, response.status));
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder("utf-8");
  let buffer = "";
  let accumulated = "";

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;

    buffer += decoder.decode(value, { stream: true });
    const events = buffer.split("\n\n");
    buffer = events.pop();

    for (const event of events) {
      if (!event.trim()) continue;
      const lines = event.split("\n");
      for (const line of lines) {
        if (line.startsWith("data: ")) {
          const dataStr = line.slice(6).trim();
          if (!dataStr) continue;

          try {
            const parsed = JSON.parse(dataStr);
            if (parsed.type === "chunk") {
              accumulated += parsed.content;
              if (onChunk) onChunk(parsed.content, accumulated);
            } else if (parsed.type === "error") {
              throw new Error(parsed.message || "Error during model inference");
            }
          } catch (jsonErr) {
            if (jsonErr.message && !jsonErr.message.includes("JSON")) {
              throw jsonErr;
            }
          }
        }
      }
    }
  }

  return accumulated;
}

// ── Step 4: Full Analysis Controller (Single-Pass & Chunked) ────────────────
async function startAnalysis() {
  if (isAnalyzing || !isBackendReady || !extractedContent) return;

  isAnalyzing = true;
  hideError();

  btnSummarize.disabled = true;
  btnText.hidden = true;
  btnLoading.hidden = false;
  btnCancel.hidden = false;

  outputSection.hidden = false;
  outputText.textContent = "";

  abortController = new AbortController();
  const signal = abortController.signal;

  try {
    const chunks = splitTextIntoChunks(extractedContent, CHUNK_MAX_CHARS);

    if (chunks.length === 0) {
      throw new Error("No readable text available to analyze.");
    }

    // ── Scenario A: Single-Pass Analysis ───────────────────────────────────
    if (chunks.length === 1 && extractedContent.length <= SINGLE_PASS_LIMIT) {
      btnLoading.innerHTML = '<span class="spinner"></span> Analyzing...';

      const finalResult = await streamSummarizeRequest(
        chunks[0],
        taskSelect.value,
        extractedTitle,
        (chunk, accumulated) => {
          outputText.textContent = accumulated;
          smartScroll(outputText);
        },
        signal
      );

      if (!finalResult.trim()) {
        outputText.textContent = "No summary generated. Please try again.";
      } else {
        await saveAnalysisState(finalResult);
      }
      return;
    }

    // ── Scenario B: Long Webpage (Multi-Section Chunking) ───────────────────
    const totalChunks = Math.min(chunks.length, MAX_PROCESSABLE_CHUNKS);
    const isTruncated = chunks.length > MAX_PROCESSABLE_CHUNKS;

    outputText.textContent =
      `[Document is long (${extractedContent.length.toLocaleString()} characters) — analyzing in ${totalChunks} sections]\n\n` +
      (isTruncated ? `(Note: Capped at the first ${MAX_PROCESSABLE_CHUNKS} major sections to prevent local AI overload)\n\n` : "");

    const sectionSummaries = [];

    for (let i = 0; i < totalChunks; i++) {
      if (signal.aborted) return;

      btnLoading.innerHTML = `<span class="spinner"></span> Section ${i + 1}/${totalChunks}...`;

      const sectionTitle = extractedTitle
        ? `${extractedTitle} (Part ${i + 1}/${totalChunks})`
        : `Section ${i + 1}/${totalChunks}`;

      const progressHeader =
        `[Analyzing section ${i + 1} of ${totalChunks} (${(chunks[i].length / 1000).toFixed(1)}k chars)...]\n\n` +
        sectionSummaries.map((s, idx) => `✓ Section ${idx + 1} processed.`).join("\n") +
        (sectionSummaries.length > 0 ? "\n\n" : "");

      outputText.textContent = progressHeader;

      try {
        let chunkAccumulated = "";
        const summary = await streamSummarizeRequest(
          chunks[i],
          "summarize",
          sectionTitle,
          (chunk, acc) => {
            chunkAccumulated = acc;
            outputText.textContent = progressHeader + `Section ${i + 1} output:\n` + chunkAccumulated;
            smartScroll(outputText);
          },
          signal
        );

        sectionSummaries.push(summary.trim() || `[Section ${i + 1} summary completed]`);
      } catch (chunkErr) {
        if (signal.aborted) return;
        console.warn(`Error processing chunk ${i + 1}:`, chunkErr);
        sectionSummaries.push(`[Section ${i + 1}: Key takeaways could not be fully extracted]`);
      }
    }

    if (signal.aborted) return;

    if (sectionSummaries.length === 0) {
      throw new Error("Unable to analyze sections of this webpage.");
    }

    // ── Final Synthesis: Combine section summaries into the final result ────
    btnLoading.innerHTML = `<span class="spinner"></span> Synthesizing ${TASK_LABELS[taskSelect.value]}...`;

    outputText.textContent = "Synthesizing comprehensive final output from all sections...\n";

    const combinedSummaries = sectionSummaries
      .map((summary, idx) => `=== Section ${idx + 1} Key Insights ===\n${summary}`)
      .join("\n\n");

    const finalResult = await streamSummarizeRequest(
      combinedSummaries,
      taskSelect.value,
      extractedTitle,
      (chunk, accumulated) => {
        outputText.textContent = accumulated;
        smartScroll(outputText);
      },
      signal
    );

    const resultToSave = finalResult.trim() || combinedSummaries;
    outputText.textContent = resultToSave;
    await saveAnalysisState(resultToSave);
  } catch (err) {
    if (err.name === "AbortError" || signal.aborted) {
      if (!outputText.textContent.trim()) {
        outputText.textContent = "(Analysis cancelled)";
      } else {
        outputText.textContent += "\n\n(Analysis cancelled)";
      }
    } else {
      showError(err.message || "Failed to communicate with backend.");
    }
  } finally {
    isAnalyzing = false;
    abortController = null;
    btnSummarize.disabled = !isBackendReady || !extractedContent;
    btnText.hidden = false;
    btnLoading.hidden = true;
    btnCancel.hidden = true;
    updateButtonLabel();
  }
}

// ── Step 5: High-Quality Client-Side PDF Generation (jsPDF) ─────────────────
async function generateSummaryPdf() {
  const summaryText = outputText.textContent.trim();
  if (!summaryText || summaryText === "(Analysis cancelled)") {
    showError("No summary available to download. Please run an analysis first.");
    return;
  }

  if (isGeneratingPdf) return;
  isGeneratingPdf = true;
  btnPdf.disabled = true;

  try {
    const { jsPDF } = window.jspdf || {};
    if (!jsPDF) {
      throw new Error("PDF generation engine is not loaded.");
    }

    const doc = new jsPDF({
      orientation: "portrait",
      unit: "mm",
      format: "a4",
    });

    const pageWidth = doc.internal.pageSize.getWidth(); // 210mm
    const pageHeight = doc.internal.pageSize.getHeight(); // 297mm
    const marginLeft = 20;
    const marginRight = 20;
    const marginTop = 22;
    const marginBottom = 20;
    const printableWidth = pageWidth - marginLeft - marginRight; // 170mm

    let cursorY = marginTop;

    function checkPageOverflow(neededHeight) {
      if (cursorY + neededHeight > pageHeight - marginBottom) {
        doc.addPage();
        cursorY = marginTop;
        return true;
      }
      return false;
    }

    // ── Header Banner Bar ──────────────────────────────────────────────────
    doc.setFillColor(79, 70, 229); // Primary Indigo
    doc.rect(marginLeft, cursorY, printableWidth, 1.2, "F");
    cursorY += 6;

    // Brand Label
    doc.setFont("helvetica", "bold");
    doc.setFontSize(9.5);
    doc.setTextColor(79, 70, 229);
    doc.text("WEBWISE AI — REAL-TIME WEB SUMMARIZER", marginLeft, cursorY);
    cursorY += 6;

    // Document Title
    const titleText = extractedTitle || "Webpage Analysis Summary";
    doc.setFont("helvetica", "bold");
    doc.setFontSize(15);
    doc.setTextColor(15, 23, 42); // slate-900
    const titleLines = doc.splitTextToSize(titleText, printableWidth);
    doc.text(titleLines, marginLeft, cursorY);
    cursorY += titleLines.length * 6.5 + 2;

    // Metadata Block
    doc.setFont("helvetica", "normal");
    doc.setFontSize(8.5);
    doc.setTextColor(100, 116, 139); // slate-500

    const modeLabel = TASK_LABELS[taskSelect.value] || "Summary";
    const dateStr = new Date().toLocaleString(undefined, {
      year: "numeric",
      month: "long",
      day: "numeric",
      hour: "2-digit",
      minute: "2-digit",
    });

    doc.text(`Analysis Mode: ${modeLabel}`, marginLeft, cursorY);
    cursorY += 4.5;
    doc.text(`Generated: ${dateStr}`, marginLeft, cursorY);
    cursorY += 4.5;

    if (currentTabUrl) {
      const urlLines = doc.splitTextToSize(`Source URL: ${currentTabUrl}`, printableWidth);
      doc.text(urlLines, marginLeft, cursorY);
      cursorY += urlLines.length * 4.5;
    }

    const activeProvider = providerSelect ? providerSelect.value : "ollama";
    let engineLabel = "Engine: Local Qwen3 AI via Ollama (100% Private Inference)";
    if (activeProvider === "groq") {
      const groqModel = currentHealthData && currentHealthData.groq_model ? currentHealthData.groq_model : "llama-3.3-70b-versatile";
      engineLabel = `Engine: Groq Cloud AI (${groqModel}) via WebWise AI`;
    }
    doc.text(engineLabel, marginLeft, cursorY);
    cursorY += 6;

    // Divider Line
    doc.setDrawColor(226, 232, 240); // slate-200
    doc.setLineWidth(0.4);
    doc.line(marginLeft, cursorY, marginLeft + printableWidth, cursorY);
    cursorY += 8;

    // ── Content Body Rendering with Auto-Pagination ────────────────────────
    const rawParagraphs = summaryText.split(/\n\s*\n/);

    for (const paragraph of rawParagraphs) {
      const trimmedPara = paragraph.trim();
      if (!trimmedPara) continue;

      // Check for Section / Header markers (e.g., === Section 1 === or # Heading)
      if (trimmedPara.startsWith("===") || trimmedPara.startsWith("#")) {
        checkPageOverflow(12);
        doc.setFont("helvetica", "bold");
        doc.setFontSize(11.5);
        doc.setTextColor(79, 70, 229);
        const headingLines = doc.splitTextToSize(trimmedPara.replace(/[#=]/g, "").trim(), printableWidth);
        doc.text(headingLines, marginLeft, cursorY);
        cursorY += headingLines.length * 5.5 + 3;
        continue;
      }

      // Check for bullet or numbered list items
      const lines = trimmedPara.split("\n");
      const isList = lines.some((l) => /^[•\-\*]\s+|^\d+\.\s+/.test(l.trim()));

      if (isList) {
        for (const line of lines) {
          const trimmedLine = line.trim();
          if (!trimmedLine) continue;

          const isBullet = /^[•\-\*]\s+/.test(trimmedLine);
          const isNumbered = /^\d+\.\s+/.test(trimmedLine);

          doc.setFont("helvetica", "normal");
          doc.setFontSize(9.5);
          doc.setTextColor(30, 41, 59);

          if (isBullet || isNumbered) {
            const numMatch = trimmedLine.match(/^\d+\.\s+/);
            const bulletPrefix = isBullet ? "• " : (numMatch ? numMatch[0] : "");
            const itemText = trimmedLine.replace(/^[•\-\*]\s+|^\d+\.\s+/, "");
            const bulletIndent = 6;
            const itemLines = doc.splitTextToSize(itemText, printableWidth - bulletIndent);
            const lineBlockHeight = itemLines.length * 4.8;

            checkPageOverflow(lineBlockHeight + 2);

            // Draw bullet marker in accent color
            doc.setFont("helvetica", "bold");
            doc.setTextColor(79, 70, 229);
            doc.text(bulletPrefix, marginLeft, cursorY);

            // Draw item body
            doc.setFont("helvetica", "normal");
            doc.setTextColor(30, 41, 59);
            doc.text(itemLines, marginLeft + bulletIndent, cursorY);
            cursorY += lineBlockHeight + 2;
          } else {
            const subLines = doc.splitTextToSize(trimmedLine, printableWidth - 6);
            checkPageOverflow(subLines.length * 4.8 + 2);
            doc.text(subLines, marginLeft + 6, cursorY);
            cursorY += subLines.length * 4.8 + 2;
          }
        }
        cursorY += 2;
      } else {
        // Standard prose paragraph
        doc.setFont("helvetica", "normal");
        doc.setFontSize(9.5);
        doc.setTextColor(30, 41, 59);
        const pLines = doc.splitTextToSize(trimmedPara, printableWidth);
        const pHeight = pLines.length * 4.8;

        checkPageOverflow(pHeight + 4);
        doc.text(pLines, marginLeft, cursorY);
        cursorY += pHeight + 4;
      }
    }

    // ── Running Footers on All Pages ───────────────────────────────────────
    const totalPages = doc.internal.pages.length - 1;
    for (let p = 1; p <= totalPages; p++) {
      doc.setPage(p);
      doc.setFont("helvetica", "normal");
      doc.setFontSize(8);
      doc.setTextColor(148, 163, 184); // slate-400

      doc.setDrawColor(241, 245, 249);
      doc.setLineWidth(0.3);
      doc.line(marginLeft, pageHeight - 12, marginLeft + printableWidth, pageHeight - 12);

      doc.text("WebWise AI · Local & Private Web Summarizer", marginLeft, pageHeight - 7);
      doc.text(`Page ${p} of ${totalPages}`, marginLeft + printableWidth, pageHeight - 7, { align: "right" });
    }

    // Save and download PDF
    const filename = sanitizeFilename(extractedTitle);
    doc.save(filename);
    showToast("PDF downloaded!");
  } catch (err) {
    console.error("PDF generation error:", err);
    showError("Could not generate PDF: " + err.message);
  } finally {
    isGeneratingPdf = false;
    btnPdf.disabled = false;
  }
}

// ── Step 6: Event Handlers ─────────────────────────────────────────────────

// Task selection change
taskSelect.addEventListener("change", () => {
  updateButtonLabel();
});

// Provider selection change
if (providerSelect) {
  providerSelect.addEventListener("change", async () => {
    updateProviderReadiness();
    try {
      if (chrome.storage && chrome.storage.local) {
        await chrome.storage.local.set({ selectedProvider: providerSelect.value });
      }
    } catch (_) {}
  });
}

// Summarize button click
btnSummarize.addEventListener("click", (e) => {
  e.preventDefault();
  e.stopPropagation();
  startAnalysis();
});

// Cancel button click
btnCancel.addEventListener("click", (e) => {
  e.preventDefault();
  e.stopPropagation();
  if (abortController) {
    abortController.abort();
  }
});

// Pop-out / Detach to separate window button
if (btnPopout) {
  btnPopout.addEventListener("click", (e) => {
    e.preventDefault();
    e.stopPropagation();
    chrome.windows.create({
      url: chrome.runtime.getURL("popup.html?detached=true"),
      type: "popup",
      width: 440,
      height: 640,
    });
    window.close(); // Close the transient flyout popup
  });
}

// Copy output to clipboard
btnCopy.addEventListener("click", async (e) => {
  e.preventDefault();
  e.stopPropagation();
  const text = outputText.textContent.trim();
  if (!text) return;
  try {
    await navigator.clipboard.writeText(text);
    showToast("Copied to clipboard!");
  } catch (err) {
    showError("Could not copy to clipboard: " + err.message);
  }
});

// Download PDF button
if (btnPdf) {
  btnPdf.addEventListener("click", (e) => {
    e.preventDefault();
    e.stopPropagation();
    generateSummaryPdf();
  });
}

// Clear output
btnClear.addEventListener("click", async (e) => {
  e.preventDefault();
  e.stopPropagation();
  outputText.textContent = "";
  outputSection.hidden = true;
  hideError();
  try {
    if (chrome.storage && chrome.storage.local) {
      await chrome.storage.local.remove("lastAnalysis");
    }
  } catch (_) {}
});

// ── Lifecycle Initialization ───────────────────────────────────────────────
document.addEventListener("DOMContentLoaded", async () => {
  const isDetached = window.location.search.includes("detached=true");
  if (isDetached) {
    document.body.classList.add("detached");
    if (btnPopout) btnPopout.hidden = true;
  }

  // Restore saved provider preference
  try {
    if (chrome.storage && chrome.storage.local) {
      const data = await chrome.storage.local.get("selectedProvider");
      if (data && data.selectedProvider && providerSelect) {
        providerSelect.value = data.selectedProvider;
      }
    }
  } catch (_) {}

  updateButtonLabel();
  checkHealth();
  initActiveTab();
});
