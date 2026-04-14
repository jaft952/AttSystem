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
const modelType = document.getElementById("modelType");

const config = window.APP_CONFIG || {};
let actionBusy = false;
let cameraRunning = false;
let latestTimer = null;
let busy = false;
let currentModel = config.modelType || "lbph";
const recentStatuses = [];
const WINDOW_SIZE = 60;

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
    modelType.textContent = (data.model_type || currentModel).toUpperCase();

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

async function startCameraStream() {
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

async function switchModel(newModelType) {
  if (actionBusy) {
    return;
  }

  actionBusy = true;
  modelSelector.disabled = true;

  try {
    modelSwitchStatus.textContent = `Switching to ${newModelType.toUpperCase()}...`;
    modelSwitchStatus.style.display = "block";

    const response = await fetch("/api/model/switch", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ model_type: newModelType }),
    });

    const result = await response.json();
    if (!response.ok) {
      throw new Error(result.error || "Failed to switch model");
    }

    currentModel = newModelType;
    modelSwitchStatus.textContent = `Switched to ${newModelType.toUpperCase()}`;

    setTimeout(() => {
      location.reload();
    }, 800);
  } catch (error) {
    modelSwitchStatus.textContent = `Error: ${error.message}`;
    modelSelector.value = currentModel;
  } finally {
    actionBusy = false;
    modelSelector.disabled = false;
  }
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
startCameraStream();
