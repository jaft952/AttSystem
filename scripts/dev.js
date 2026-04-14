const predictedName = document.getElementById("predictedName");
const predictedStatus = document.getElementById("predictedStatus");
const confidenceValue = document.getElementById("confidenceValue");
const confidenceBar = document.getElementById("confidenceBar");
const bboxValue = document.getElementById("bboxValue");
const cameraHint = document.getElementById("cameraHint");
const cameraVideo = document.getElementById("cameraVideo");
const cameraOverlay = document.getElementById("cameraOverlay");
const cameraToggle = document.getElementById("cameraToggle");
const modelSelector = document.getElementById("modelSelector");
const modelSwitchStatus = document.getElementById("modelSwitchStatus");
const acceptedRate = document.getElementById("acceptedRate");
const knownRate = document.getElementById("knownRate");
const noFaceRate = document.getElementById("noFaceRate");
const framesProcessed = document.getElementById("framesProcessed");
const preferredModel = document.getElementById("preferredModel");
const modelType = document.getElementById("modelType");
const runtimeModelName = document.getElementById("runtimeModelName");
const runtimeThreshold = document.getElementById("runtimeThreshold");

const config = window.APP_CONFIG || {};
let actionBusy = false;
let cameraRunning = false;
let latestTimer = null;
let busy = false;
let currentModel = config.modelType || "cbir_method1";
const MODEL_PREF_KEY = "attsystem_selected_model";
const recentStatuses = [];
const WINDOW_SIZE = 60;
const SUPPORTED_MODELS = ["cbir_method1", "cbir_method2"];

function getModelLabel(model) {
  if (model === "cbir_method2") {
    return "CBIR Method 2";
  }
  return "CBIR Method 1";
}

function setPreferredModel(value) {
  if (preferredModel) {
    preferredModel.textContent = getModelLabel(
      (value || currentModel).toLowerCase(),
    );
  }
}

function applyRuntimeModelUi(model, threshold) {
  const resolvedModel = (model || currentModel || "cbir_method1").toLowerCase();
  currentModel = resolvedModel;

  if (modelType) {
    modelType.textContent = getModelLabel(resolvedModel);
  }
  if (runtimeModelName) {
    runtimeModelName.textContent = getModelLabel(resolvedModel);
  }
  if (runtimeThreshold && typeof threshold === "number") {
    runtimeThreshold.textContent = String(threshold);
  }
}

function setPredictionState(name, status, confidence, bbox) {
  predictedName.textContent = name || "Waiting...";
  predictedStatus.textContent = status || "No prediction yet";
  confidenceValue.textContent =
    typeof confidence === "number" ? confidence.toFixed(3) : "--";

  const percent =
    typeof confidence === "number"
      ? Math.max(0, Math.min(100, 100 - confidence))
      : 0;
  confidenceBar.style.width = `${percent}%`;

  if (bbox) {
    bboxValue.textContent = `${bbox.x}, ${bbox.y}, ${bbox.w} x ${bbox.h}`;
  } else {
    bboxValue.textContent = "--";
  }
}

function setCameraHint(message) {
  cameraHint.textContent = message;
}

async function readJson(response) {
  try {
    return await response.json();
  } catch (_) {
    return {};
  }
}

function updateRollingStats(status, prediction, frameCount) {
  if (status) {
    recentStatuses.push({ status, prediction });
    if (recentStatuses.length > WINDOW_SIZE) {
      recentStatuses.shift();
    }
  }

  const total = recentStatuses.length;
  const accepted = recentStatuses.filter(
    (x) => x.prediction && x.prediction.accepted,
  ).length;
  const known = recentStatuses.filter(
    (x) =>
      x.prediction &&
      x.prediction.raw_name &&
      x.prediction.raw_name !== "unknown",
  ).length;
  const noFace = recentStatuses.filter((x) => x.status === "no_face").length;

  acceptedRate.textContent =
    total > 0 ? `${((accepted / total) * 100).toFixed(1)}%` : "--";
  knownRate.textContent =
    total > 0 ? `${((known / total) * 100).toFixed(1)}%` : "--";
  noFaceRate.textContent =
    total > 0 ? `${((noFace / total) * 100).toFixed(1)}%` : "--";
  framesProcessed.textContent = String(frameCount || 0);
}

function drawOverlayBbox(bbox, sourceWidth, sourceHeight) {
  const displayWidth = cameraVideo.clientWidth;
  const displayHeight = cameraVideo.clientHeight;
  if (!displayWidth || !displayHeight) {
    return;
  }

  cameraOverlay.width = displayWidth;
  cameraOverlay.height = displayHeight;
  const ctx = cameraOverlay.getContext("2d");
  if (!ctx) {
    return;
  }

  ctx.clearRect(0, 0, displayWidth, displayHeight);
  if (!bbox || !sourceWidth || !sourceHeight) {
    return;
  }

  const scale = Math.min(
    displayWidth / sourceWidth,
    displayHeight / sourceHeight,
  );
  const offsetX = (displayWidth - sourceWidth * scale) / 2;
  const offsetY = (displayHeight - sourceHeight * scale) / 2;

  const x = offsetX + bbox.x * scale;
  const y = offsetY + bbox.y * scale;
  const w = bbox.w * scale;
  const h = bbox.h * scale;

  ctx.strokeStyle = "#f7f7f7";
  ctx.lineWidth = 2;
  ctx.strokeRect(x, y, w, h);
}

function stopPolling() {
  if (latestTimer) {
    clearInterval(latestTimer);
    latestTimer = null;
  }
  busy = false;
}

function startPolling() {
  stopPolling();
  fetchLatest();
  latestTimer = setInterval(fetchLatest, 350);
}

async function fetchLatest() {
  if (busy || !config.latestUrl) {
    return;
  }

  busy = true;
  try {
    const response = await fetch(config.latestUrl, { cache: "no-store" });
    const data = await response.json();

    if (!response.ok) {
      throw new Error(data.error || "Could not load latest prediction");
    }

    const prediction = data.prediction || {};
    setPredictionState(
      prediction.name || "Waiting...",
      prediction.message || "No prediction yet",
      prediction.confidence,
      prediction.bbox || null,
    );

    updateRollingStats(prediction.status, prediction, data.frame_count);
    applyRuntimeModelUi(data.model_type || currentModel);

    if (data.running) {
      setCameraHint("Backend stream active.");
    } else {
      setCameraHint("Camera stream is not running.");
    }
  } catch (error) {
    setPredictionState("Error", error.message, null, null);
    setCameraHint(error.message);
  } finally {
    busy = false;
  }
}

async function startCameraStream({ ignoreBusy = false } = {}) {
  if (actionBusy && !ignoreBusy) {
    return;
  }
  cameraToggle.disabled = true;

  try {
    const response = await fetch(config.cameraStartUrl, { method: "POST" });
    const data = await readJson(response);
    if (!response.ok) {
      throw new Error(data.error || "Could not start camera");
    }

    cameraRunning = true;
    cameraVideo.src = `${config.videoFeedUrl}?t=${Date.now()}`;
    cameraToggle.textContent = "Close Camera";
    setCameraHint("Backend camera active.");
    startPolling();
  } catch (error) {
    cameraRunning = false;
    setCameraHint(error.message || "Camera access failed.");
    cameraToggle.textContent = "Open Camera";
  } finally {
    cameraToggle.disabled = false;
  }
}

async function stopCameraStream() {
  try {
    await fetch(config.cameraStopUrl, { method: "POST" });
  } catch (_) {
    // ignore
  }

  cameraRunning = false;
  cameraVideo.removeAttribute("src");
  cameraVideo.src = "";
  cameraToggle.textContent = "Open Camera";
  stopPolling();
  drawOverlayBbox(null, 0, 0);
  setCameraHint("Camera stream is paused.");
  setPredictionState("Waiting...", "Camera paused.", null, null);
}

function setSwitchingState(isSwitching, statusText) {
  actionBusy = isSwitching;
  if (modelSelector) {
    modelSelector.disabled = isSwitching;
  }
  if (cameraToggle) {
    cameraToggle.disabled = isSwitching;
  }
  if (statusText) {
    modelSwitchStatus.textContent = statusText;
    modelSwitchStatus.style.display = "block";
  }
}

async function applyModelPreference(targetModel, messagePrefix) {
  const wasRunning = cameraRunning;
  setSwitchingState(true, `${messagePrefix} ${getModelLabel(targetModel)}...`);

  try {
    if (wasRunning) {
      await stopCameraStream();
    }

    const response = await fetch("/api/model/switch", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ model_type: targetModel }),
    });

    const result = await readJson(response);
    if (!response.ok) {
      console.error("Model switch request failed", {
        status: response.status,
        targetModel,
        error: result.error || "Failed to switch model",
      });
      throw new Error(result.error || "Failed to switch model");
    }

    const resolved = (result.current_model || targetModel).toLowerCase();
    localStorage.setItem(MODEL_PREF_KEY, resolved);
    setPreferredModel(resolved);
    if (modelSelector) {
      modelSelector.value = resolved;
    }
    applyRuntimeModelUi(resolved, result.threshold);
    modelSwitchStatus.textContent = `Using ${getModelLabel(resolved)}`;

    if (wasRunning) {
      await startCameraStream({ ignoreBusy: true });
    }
  } catch (error) {
    localStorage.setItem(MODEL_PREF_KEY, currentModel);
    setPreferredModel(currentModel);
    if (modelSelector) {
      modelSelector.value = currentModel;
    }
    applyRuntimeModelUi(currentModel);
    modelSwitchStatus.textContent = `Model switch error: ${error.message}`;

    if (wasRunning) {
      await startCameraStream({ ignoreBusy: true });
    }
  } finally {
    setSwitchingState(false);
  }
}

async function switchModel(newModelType) {
  const normalized = String(newModelType || "").toLowerCase();
  if (actionBusy || !SUPPORTED_MODELS.includes(normalized)) {
    return;
  }

  if (normalized === currentModel) {
    localStorage.setItem(MODEL_PREF_KEY, normalized);
    setPreferredModel(normalized);
    applyRuntimeModelUi(normalized);
    modelSwitchStatus.textContent = `Using ${getModelLabel(normalized)}`;
    modelSwitchStatus.style.display = "block";
    return;
  }

  await applyModelPreference(normalized, "Switching to");
}

async function syncModelPreferenceOnLoad() {
  const saved = localStorage.getItem(MODEL_PREF_KEY);
  if (!saved || !SUPPORTED_MODELS.includes(saved)) {
    localStorage.setItem(MODEL_PREF_KEY, currentModel);
    setPreferredModel(currentModel);
    return;
  }

  setPreferredModel(saved);

  if (modelSelector) {
    modelSelector.value = saved;
  }

  if (saved === currentModel) {
    return;
  }

  await applyModelPreference(saved, "Applying saved model");
}

cameraToggle.addEventListener("click", async () => {
  if (cameraRunning) {
    await stopCameraStream();
  } else {
    await startCameraStream();
  }
});

modelSelector.addEventListener("change", (e) => {
  switchModel(e.target.value);
});

window.addEventListener("beforeunload", () => {
  stopPolling();
  if (cameraRunning) {
    fetch(config.cameraStopUrl || "/api/camera/stop", {
      method: "POST",
      keepalive: true,
    }).catch(() => {});
  }
});

setPredictionState("Waiting...", "Starting backend camera...", null, null);
setPreferredModel(localStorage.getItem(MODEL_PREF_KEY) || currentModel);
applyRuntimeModelUi(currentModel);
syncModelPreferenceOnLoad().then(() => {
  startCameraStream();
});
