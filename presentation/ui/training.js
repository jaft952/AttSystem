const predictedName = document.getElementById("predictedName");
const predictedStatus = document.getElementById("predictedStatus");
const confidenceValue = document.getElementById("confidenceValue");
const confidenceBar = document.getElementById("confidenceBar");
const bboxValue = document.getElementById("bboxValue");
const cameraHint = document.getElementById("cameraHint");
const cameraVideo = document.getElementById("cameraVideo");
const cameraOverlay = document.getElementById("cameraOverlay");
const cameraToggle = document.getElementById("cameraToggle");
const retrainOverlay = document.getElementById("retrainOverlay");
const retrainOverlayText = document.getElementById("retrainOverlayText");
const feedbackStatus = document.getElementById("feedbackStatus");
const feedbackLabel = document.getElementById("feedbackLabel");
const confirmBtn = document.getElementById("confirmBtn");
const correctBtn = document.getElementById("correctBtn");
const retrainBtn = document.getElementById("retrainBtn");

const config = window.APP_CONFIG || {};
let actionBusy = false;
let retrainTimer = null;
let retraining = false;
let cameraRunning = false;
let latestTimer = null;
let busy = false;

function setPredictionState(name, status, confidence, bbox) {
  if (
    !predictedName ||
    !predictedStatus ||
    !confidenceValue ||
    !confidenceBar ||
    !bboxValue
  ) {
    return;
  }

  predictedName.textContent = name || "Waiting...";
  predictedStatus.textContent = status || "No prediction yet";
  confidenceValue.textContent =
    typeof confidence === "number" ? confidence.toFixed(2) : "--";

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

function setFeedbackState(message) {
  if (feedbackStatus) {
    feedbackStatus.textContent = message;
  }
}

function setRetrainOverlay(visible, message) {
  if (!retrainOverlay) {
    return;
  }

  retrainOverlay.classList.toggle("hidden", !visible);
  if (retrainOverlayText && message) {
    retrainOverlayText.textContent = message;
  }
}

function setCameraHint(message) {
  if (cameraHint) {
    cameraHint.textContent = message;
  }
}

function parseApiError(data, fallback) {
  return data && data.error ? data.error : fallback;
}

async function readJson(response) {
  try {
    return await response.json();
  } catch (_) {
    return {};
  }
}

function drawOverlayBbox(bbox, sourceWidth, sourceHeight) {
  if (!cameraOverlay || !cameraVideo) {
    return;
  }

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
    const message =
      prediction.message ||
      (data.last_error ? data.last_error : "Streaming live from Flask.");

    setPredictionState(
      prediction.name || "Waiting...",
      message,
      prediction.confidence,
      prediction.bbox || null,
    );

    if (prediction.status === "ok") {
      setFeedbackState(
        prediction.accepted
          ? "You can confirm this prediction or save a correction."
          : "Prediction is below threshold. Save a correction if you know the right person.",
      );
    } else if (prediction.status === "no_face") {
      setFeedbackState("No face available for feedback.");
    } else if (prediction.status === "invalid_roi") {
      setFeedbackState(
        "Face crop was invalid. Move closer or improve lighting.",
      );
    }

    if (data.running) {
      setCameraHint(
        `Backend stream active. Frames processed: ${data.frame_count || 0}`,
      );
    } else if (!retraining) {
      setCameraHint("Camera stream is not running.");
    }
  } catch (error) {
    console.error(error);
    setPredictionState("Error", error.message, null, null);
    setCameraHint(error.message);
  } finally {
    busy = false;
  }
}

async function startCameraStream() {
  if (!cameraVideo || !config.videoFeedUrl || !config.cameraStartUrl) {
    return;
  }

  cameraToggle.disabled = true;

  try {
    const response = await fetch(config.cameraStartUrl, { method: "POST" });
    const data = await readJson(response);
    if (!response.ok) {
      throw new Error(parseApiError(data, "Could not start camera"));
    }

    cameraRunning = true;
    cameraVideo.src = `${config.videoFeedUrl}?t=${Date.now()}`;
    cameraToggle.textContent = "Close Camera";
    setCameraHint("Backend camera active. Running low-latency predictions.");
    startPolling();
  } catch (error) {
    console.error(error);
    cameraRunning = false;
    setCameraHint(error.message || "Camera access failed.");
    cameraToggle.textContent = "Open Camera";
  } finally {
    cameraToggle.disabled = retraining;
  }
}

async function stopCameraStream({ fromRetrain = false } = {}) {
  if (!cameraVideo || !config.cameraStopUrl) {
    return;
  }

  try {
    await fetch(config.cameraStopUrl, { method: "POST" });
  } catch (error) {
    console.error(error);
  }

  cameraRunning = false;
  cameraVideo.removeAttribute("src");
  cameraVideo.src = "";
  cameraToggle.textContent = "Open Camera";
  stopPolling();
  drawOverlayBbox(null, 0, 0);
  if (!fromRetrain) {
    setCameraHint("Camera stream is paused.");
    setPredictionState("Waiting...", "Camera paused.", null, null);
  }
}

function toggleActionButtons(disabled) {
  if (confirmBtn) confirmBtn.disabled = disabled;
  if (correctBtn) correctBtn.disabled = disabled;
  if (retrainBtn) retrainBtn.disabled = disabled;
  if (feedbackLabel) feedbackLabel.disabled = disabled;
}

async function submitFeedback(action) {
  if (actionBusy || retraining) {
    return;
  }

  actionBusy = true;
  toggleActionButtons(true);

  try {
    const payload = { action };
    if (action === "correct") {
      payload.label = feedbackLabel ? feedbackLabel.value : "";
    }

    const response = await fetch("/api/feedback", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });

    const data = await readJson(response);
    if (!response.ok) {
      throw new Error(parseApiError(data, "Could not save feedback"));
    }

    setFeedbackState(`Saved: ${data.saved_label}`);
  } catch (error) {
    console.error(error);
    setFeedbackState(error.message);
  } finally {
    actionBusy = false;
    toggleActionButtons(false);
  }
}

async function retrainModel() {
  if (actionBusy) {
    return;
  }

  actionBusy = true;
  toggleActionButtons(true);
  retraining = true;
  setRetrainOverlay(true, "Retraining model... this can take a while");
  if (cameraToggle) {
    cameraToggle.disabled = true;
  }
  setFeedbackState("Retraining started. Camera is paused to save resources.");
  setPredictionState(
    "Retraining...",
    "Camera paused. Model update is running in background.",
    null,
    null,
  );
  await stopCameraStream({ fromRetrain: true });

  try {
    const response = await fetch(config.retrainUrl, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
    });
    const data = await readJson(response);

    if (!response.ok) {
      throw new Error(parseApiError(data, "Could not start retraining"));
    }

    if (data.status !== "started" && data.status !== "running") {
      throw new Error("Unexpected retraining response from backend.");
    }

    setFeedbackState("Retraining in progress...");
    startRetrainPolling();
  } catch (error) {
    console.error(error);
    setFeedbackState(error.message);
    retraining = false;
    actionBusy = false;
    toggleActionButtons(false);
    setRetrainOverlay(false, "");
    await startCameraStream();
  } finally {
    if (!retraining && cameraToggle) {
      cameraToggle.disabled = false;
    }
  }
}

function startRetrainPolling() {
  if (retrainTimer) {
    clearInterval(retrainTimer);
  }

  pollRetrainStatus();
  retrainTimer = setInterval(pollRetrainStatus, 1000);
}

async function pollRetrainStatus() {
  if (!config.retrainStatusUrl) {
    return;
  }

  try {
    const response = await fetch(config.retrainStatusUrl, {
      cache: "no-store",
    });
    const data = await readJson(response);
    if (!response.ok) {
      throw new Error(parseApiError(data, "Could not read retraining status"));
    }

    if (data.status === "running") {
      const startedAt = data.started_at ? Date.parse(data.started_at) : null;
      const elapsed =
        startedAt && Number.isFinite(startedAt)
          ? Math.max(0, Math.floor((Date.now() - startedAt) / 1000))
          : null;
      const elapsedText = elapsed === null ? "" : ` (${elapsed}s)`;
      setRetrainOverlay(
        true,
        `Retraining model in background${elapsedText}. Camera stays paused.`,
      );
      return;
    }

    if (data.status === "completed") {
      const result = data.result || {};
      setFeedbackState(
        `Retrained: ${result.training_samples || 0} samples (${result.feedback_samples || 0} feedback)`,
      );
      await finalizeRetraining();
      return;
    }

    if (data.status === "failed") {
      setFeedbackState(data.error || "Retraining failed.");
      await finalizeRetraining();
    }
  } catch (error) {
    console.error(error);
    if (error && error.message === "Failed to fetch") {
      setFeedbackState("Connection interrupted. Retrying retrain status...");
      return;
    }
    setFeedbackState(error.message);
  }
}

async function finalizeRetraining() {
  if (retrainTimer) {
    clearInterval(retrainTimer);
    retrainTimer = null;
  }

  retraining = false;
  actionBusy = false;
  setRetrainOverlay(false, "");
  toggleActionButtons(false);
  await startCameraStream();
}

async function syncRetrainStateOnLoad() {
  if (!config.retrainStatusUrl) {
    return;
  }
  try {
    const response = await fetch(config.retrainStatusUrl, {
      cache: "no-store",
    });
    const data = await readJson(response);
    if (!response.ok) {
      return;
    }
    if (data.status === "running") {
      retraining = true;
      actionBusy = true;
      toggleActionButtons(true);
      setRetrainOverlay(true, "Retraining model... this can take a while");
      if (cameraToggle) {
        cameraToggle.disabled = true;
      }
      await stopCameraStream({ fromRetrain: true });
      setFeedbackState("Retraining in progress...");
      startRetrainPolling();
    }
  } catch (error) {
    console.error(error);
  }
}

if (cameraToggle) {
  cameraToggle.addEventListener("click", async () => {
    if (retraining) {
      return;
    }
    if (cameraRunning) {
      await stopCameraStream();
    } else {
      await startCameraStream();
    }
  });
}
if (confirmBtn)
  confirmBtn.addEventListener("click", () => submitFeedback("confirm"));
if (correctBtn)
  correctBtn.addEventListener("click", () => submitFeedback("correct"));
if (retrainBtn) retrainBtn.addEventListener("click", retrainModel);

window.addEventListener("beforeunload", () => {
  stopPolling();
  if (cameraRunning) {
    fetch(config.cameraStopUrl || "/api/camera/stop", {
      method: "POST",
      keepalive: true,
    }).catch(() => {});
  }
  if (retrainTimer) {
    clearInterval(retrainTimer);
  }
});

setPredictionState("Waiting...", "Starting backend camera...", null, null);
setFeedbackState("Ready");
syncRetrainStateOnLoad().then(() => {
  if (!retraining) {
    startCameraStream();
  }
});
