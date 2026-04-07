const predictedName = document.getElementById("predictedName");
const predictedStatus = document.getElementById("predictedStatus");
const confidenceValue = document.getElementById("confidenceValue");
const confidenceBar = document.getElementById("confidenceBar");
const bboxValue = document.getElementById("bboxValue");
const cameraHint = document.getElementById("cameraHint");
const feedbackStatus = document.getElementById("feedbackStatus");
const feedbackLabel = document.getElementById("feedbackLabel");
const confirmBtn = document.getElementById("confirmBtn");
const correctBtn = document.getElementById("correctBtn");
const retrainBtn = document.getElementById("retrainBtn");

const config = window.APP_CONFIG || {};
let busy = false;
let actionBusy = false;
let latestTimer = null;

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

function setCameraHint(message) {
  if (cameraHint) {
    cameraHint.textContent = message;
  }
}

function toggleActionButtons(disabled) {
  if (confirmBtn) confirmBtn.disabled = disabled;
  if (correctBtn) correctBtn.disabled = disabled;
  if (retrainBtn) retrainBtn.disabled = disabled;
  if (feedbackLabel) feedbackLabel.disabled = disabled;
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
    } else {
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

async function submitFeedback(action) {
  if (actionBusy || !config.latestUrl) {
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

    const data = await response.json();
    if (!response.ok) {
      throw new Error(data.error || "Could not save feedback");
    }

    setFeedbackState(`Saved: ${data.saved_label}`);
    await fetchLatest();
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
  setFeedbackState("Retraining model...");

  try {
    const response = await fetch("/api/retrain", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
    });
    const data = await response.json();

    if (!response.ok) {
      throw new Error(data.error || "Retraining failed");
    }

    setFeedbackState(
      `Retrained: ${data.training_samples} samples (${data.feedback_samples} feedback)`,
    );
    await fetchLatest();
  } catch (error) {
    console.error(error);
    setFeedbackState(error.message);
  } finally {
    actionBusy = false;
    toggleActionButtons(false);
  }
}

function startPolling() {
  if (latestTimer) {
    clearInterval(latestTimer);
  }

  fetchLatest();
  latestTimer = setInterval(fetchLatest, 350);
}

if (confirmBtn)
  confirmBtn.addEventListener("click", () => submitFeedback("confirm"));
if (correctBtn)
  correctBtn.addEventListener("click", () => submitFeedback("correct"));
if (retrainBtn) retrainBtn.addEventListener("click", retrainModel);

window.addEventListener("beforeunload", () => {
  if (latestTimer) {
    clearInterval(latestTimer);
  }
});

setPredictionState("Waiting...", "Loading backend stream...", null, null);
setFeedbackState("Ready");
startPolling();
