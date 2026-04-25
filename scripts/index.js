const cameraVideo = document.getElementById("cameraVideo");
const cameraOverlay = document.getElementById("cameraOverlay");
const cameraToggle = document.getElementById("cameraToggle");
const cameraHint = document.getElementById("cameraHint");
const markStatus = document.getElementById("markStatus");
const todayDate = document.getElementById("todayDate");
const identifiedName = document.getElementById("identifiedName");
const identifiedStatus = document.getElementById("identifiedStatus");
const presentList = document.getElementById("presentList");
const absentList = document.getElementById("absentList");

const config = window.APP_CONFIG || {};
let cameraRunning = false;
let predictionSource = null;
let latestTimer = null;
let attendanceTimer = null;
let lastPrediction = null;
let autoMarkState = {
  label: null,
  startedAt: null,
  inFlight: false,
};
let presentNames = new Set();
const MODEL_PREF_KEY = "attsystem_selected_model";
const SUPPORTED_MODELS = ["cbir_method1", "cbir_method2", "cbir_method3"];
const AUTO_MARK_CONFIDENCE_THRESHOLD = 0.55;
const AUTO_MARK_HOLD_MS = 1000;
const BBOX_HEIGHT_SCALE = 2;
const BBOX_UPWARD_BIAS = 0.55;

// Source frame dimensions the server encodes at.
const FRAME_W = 640;
const FRAME_H = 480;

function setText(element, value) {
  if (element) {
    element.textContent = value;
  }
}

function setHint(text) {
  setText(cameraHint, text);
}

function setMarkStatus(text) {
  setText(markStatus, text);
}

async function readJson(response) {
  try {
    return await response.json();
  } catch (_) {
    return {};
  }
}

function formatTimestamp(value) {
  if (!value) {
    return "";
  }
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) {
    return value;
  }
  return date.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
}

// ---------------------------------------------------------------------------
// Canvas overlay — draws the face bounding box in the browser so the MJPEG
// stream carries raw frames only (consistent encode time, lower CPU cost).
// ---------------------------------------------------------------------------

function syncOverlaySize() {
  if (!cameraOverlay) return;
  const w = cameraOverlay.offsetWidth;
  const h = cameraOverlay.offsetHeight;
  if (cameraOverlay.width !== w || cameraOverlay.height !== h) {
    cameraOverlay.width = w;
    cameraOverlay.height = h;
  }
}

function clearOverlay() {
  if (!cameraOverlay) return;
  syncOverlaySize();
  const ctx = cameraOverlay.getContext("2d");
  ctx.clearRect(0, 0, cameraOverlay.width, cameraOverlay.height);
}

function expandFaceBbox(bbox, frameWidth, frameHeight) {
  if (!bbox) {
    return null;
  }

  const extraHeight = bbox.h * (BBOX_HEIGHT_SCALE - 1);
  const expandedY = bbox.y - extraHeight * BBOX_UPWARD_BIAS;
  const expandedHeight = bbox.h + extraHeight;

  const clampedX = Math.max(0, Math.min(bbox.x, frameWidth));
  const clampedY = Math.max(0, Math.min(expandedY, frameHeight));
  const maxWidth = Math.max(0, frameWidth - clampedX);
  const maxHeight = Math.max(0, frameHeight - clampedY);

  return {
    x: clampedX,
    y: clampedY,
    w: Math.max(0, Math.min(bbox.w, maxWidth)),
    h: Math.max(0, Math.min(expandedHeight, maxHeight)),
  };
}

function drawOverlay(prediction) {
  if (!cameraOverlay) return;
  syncOverlaySize();
  const ctx = cameraOverlay.getContext("2d");
  const w = cameraOverlay.width;
  const h = cameraOverlay.height;
  ctx.clearRect(0, 0, w, h);

  const bbox = prediction && prediction.bbox;
  if (!bbox) return;

  const adjustedBbox = expandFaceBbox(bbox, FRAME_W, FRAME_H);
  if (!adjustedBbox) return;

  const scaleX = w / FRAME_W;
  const scaleY = h / FRAME_H;
  const bx = adjustedBbox.x * scaleX;
  const by = adjustedBbox.y * scaleY;
  const bw = adjustedBbox.w * scaleX;
  const bh = adjustedBbox.h * scaleY;

  ctx.strokeStyle = "#ffffff";
  ctx.lineWidth = 2;
  ctx.strokeRect(bx, by, bw, bh);

  const label = prediction.name || "";
  if (label) {
    ctx.font = "bold 13px Manrope, sans-serif";
    const textMetrics = ctx.measureText(label);
    const textW = textMetrics.width;
    const tagH = 20;
    const tagY = by > tagH ? by - tagH : by + bh;
    ctx.fillStyle = "rgba(255, 255, 255, 0.88)";
    ctx.fillRect(bx, tagY, textW + 8, tagH);
    ctx.fillStyle = "#09090b";
    ctx.fillText(label, bx + 4, tagY + 14);
  }
}

// ---------------------------------------------------------------------------
// SSE prediction stream — replaces the 450 ms REST poll
// ---------------------------------------------------------------------------

function startPredictionStream() {
  if (!config.predictionStreamUrl) {
    // No SSE URL: fall back to polling
    if (!latestTimer) {
      latestTimer = setInterval(fetchLatest, 450);
    }
    return;
  }
  if (predictionSource) return;

  predictionSource = new EventSource(config.predictionStreamUrl);

  predictionSource.onmessage = function (event) {
    try {
      const prediction = JSON.parse(event.data);
      handlePredictionUpdate(prediction);
    } catch (err) {
      console.warn("SSE prediction parse error:", err, event.data);
    }
  };

  predictionSource.onerror = function () {
    // SSE connection lost — close it and fall back to REST polling.
    if (predictionSource) {
      predictionSource.close();
      predictionSource = null;
    }
    if (cameraRunning && !latestTimer) {
      latestTimer = setInterval(fetchLatest, 450);
    }
  };
}

function stopPredictionStream() {
  if (predictionSource) {
    predictionSource.close();
    predictionSource = null;
  }
  if (latestTimer) {
    clearInterval(latestTimer);
    latestTimer = null;
  }
}

function handlePredictionUpdate(prediction) {
  lastPrediction = prediction;
  const name = prediction.name || "Waiting...";
  const status = prediction.message || "No prediction yet.";
  setText(identifiedName, name);
  setText(identifiedStatus, status);
  drawOverlay(prediction);
  maybeAutoMark(prediction);
}

function canAutoMark(prediction) {
  return Boolean(
    prediction &&
    prediction.status === "ok" &&
    prediction.accepted &&
    prediction.raw_name &&
    prediction.raw_name !== "unknown" &&
    typeof prediction.confidence === "number" &&
    prediction.confidence <= AUTO_MARK_CONFIDENCE_THRESHOLD,
  );
}

function getAutoMarkCandidate(prediction) {
  if (!canAutoMark(prediction)) {
    return null;
  }

  const label = String(prediction.raw_name || "").trim();
  if (!label || presentNames.has(label)) {
    return null;
  }

  return {
    label,
    confidence: prediction.confidence,
  };
}

function resetAutoMarkState() {
  autoMarkState = {
    label: null,
    startedAt: null,
    inFlight: false,
  };
}

async function maybeAutoMark(prediction) {
  const candidate = getAutoMarkCandidate(prediction);
  
  if (!candidate || autoMarkState.inFlight) {
    if (!candidate) {
      resetAutoMarkState();
    }
    return;
  }

  const now = Date.now();
  if (autoMarkState.label !== candidate.label) {
    autoMarkState.label = candidate.label;
    autoMarkState.startedAt = now;
    setMarkStatus(
      `Auto-marking ${candidate.label} after 1 second of stable confidence...`,
    );
    return;
  }

  if (!autoMarkState.startedAt) {
    autoMarkState.startedAt = now;
    return;
  }

  const elapsed = now - autoMarkState.startedAt;
  if (elapsed < AUTO_MARK_HOLD_MS) {
    return;
  }

  autoMarkState.inFlight = true;
  try {
    await markAttendance(candidate.label, candidate.confidence);
    setMarkStatus(`Auto-marked attendance for ${candidate.label}.`);
    presentNames.add(candidate.label);
  } finally {
    autoMarkState.inFlight = false;
    resetAutoMarkState();
  }
}

function renderAttendance(attendance) {
  if (!attendance) {
    return;
  }

  setText(todayDate, attendance.date || new Date().toISOString().slice(0, 10));

  if (presentList) {
    presentList.innerHTML = "";
    const present = Array.isArray(attendance.present) ? attendance.present : [];
    presentNames = new Set();

    if (present.length === 0) {
      const li = document.createElement("li");
      li.className = "empty";
      li.textContent = "No one marked present yet.";
      presentList.appendChild(li);
    } else {
      present.forEach((person) => {
        const li = document.createElement("li");
        li.className = "present-item";
        const name = person && person.name ? person.name : "unknown";
        presentNames.add(name);
        const seen = formatTimestamp(person && person.last_seen_at);
        const label = document.createElement("span");
        label.className = "present-label";
        label.textContent = seen ? `${name} (${seen})` : name;

        const actions = document.createElement("span");
        actions.className = "present-actions";

        const button = document.createElement("button");
        button.type = "button";
        button.className = "button danger inline-button";
        button.textContent = "Set Absent";
        button.addEventListener("click", () => {
          unmarkAttendance(name);
        });

        actions.appendChild(button);
        li.appendChild(label);
        li.appendChild(actions);
        presentList.appendChild(li);
      });
    }
  }

  if (absentList) {
    absentList.innerHTML = "";
    const absent = Array.isArray(attendance.absent) ? attendance.absent : [];
    if (absent.length === 0) {
      const li = document.createElement("li");
      li.className = "empty";
      li.textContent = "Everyone is present.";
      absentList.appendChild(li);
    } else {
      absent.forEach((name) => {
        const li = document.createElement("li");
        li.textContent = name;
        absentList.appendChild(li);
      });
    }
  }
}

async function fetchAttendance() {
  if (!config.attendanceTodayUrl) {
    return;
  }
  try {
    const response = await fetch(config.attendanceTodayUrl, {
      cache: "no-store",
    });
    const data = await readJson(response);
    if (!response.ok) {
      throw new Error(data.error || "Could not load attendance stats.");
    }
    renderAttendance(data.attendance);

    if (data.attendance && Array.isArray(data.attendance.present)) {
      presentNames = new Set(
        data.attendance.present
          .map((person) => (person && person.name ? String(person.name) : ""))
          .filter(Boolean),
      );
    }
  } catch (error) {
    setMarkStatus(error.message || "Could not load attendance.");
  }
}

async function markAttendance(name, confidence) {
  if (!name || !config.attendanceMarkUrl) {
    return;
  }

  try {
    const response = await fetch(config.attendanceMarkUrl, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ label: name, confidence }),
    });
    const data = await readJson(response);
    if (!response.ok) {
      throw new Error(data.error || "Could not mark attendance.");
    }
    setMarkStatus(`Attendance marked for ${name}.`);
    renderAttendance(data.attendance);
  } catch (error) {
    setMarkStatus(error.message || "Could not mark attendance.");
  }
}

async function unmarkAttendance(name) {
  if (!name || !config.attendanceUnmarkUrl) {
    return;
  }

  try {
    const response = await fetch(config.attendanceUnmarkUrl, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ label: name }),
    });
    const data = await readJson(response);
    if (!response.ok) {
      throw new Error(data.error || "Could not unmark attendance.");
    }
    setMarkStatus(`Marked ${name} as absent.`);
    renderAttendance(data.attendance);
  } catch (error) {
    setMarkStatus(error.message || "Could not unmark attendance.");
  }
}

async function fetchLatest() {
  if (!config.latestUrl) {
    return;
  }

  try {
    const response = await fetch(config.latestUrl, { cache: "no-store" });
    const data = await readJson(response);
    if (!response.ok) {
      throw new Error(data.error || "Could not load live prediction.");
    }

    const prediction = data.prediction || {};
    handlePredictionUpdate(prediction);
    setHint(data.running ? "Camera live." : "Camera is paused.");
  } catch (error) {
    setHint(error.message || "Could not read live prediction.");
  }
}

function startPolling() {
  // Populate the UI immediately with the current prediction so there is no
  // blank state while waiting for the first SSE message.
  fetchLatest();
  startPredictionStream();
  if (!attendanceTimer) {
    attendanceTimer = setInterval(fetchAttendance, 2500);
  }
  fetchAttendance();
}

function stopPolling() {
  stopPredictionStream();
  if (attendanceTimer) {
    clearInterval(attendanceTimer);
    attendanceTimer = null;
  }
}

async function startCamera() {
  if (!config.cameraStartUrl || !config.videoFeedUrl) {
    return;
  }
  cameraToggle.disabled = true;
  try {
    const response = await fetch(config.cameraStartUrl, { method: "POST" });
    const data = await readJson(response);
    if (!response.ok) {
      throw new Error(data.error || "Could not start camera.");
    }
    cameraRunning = true;
    cameraVideo.src = `${config.videoFeedUrl}?t=${Date.now()}`;
    cameraToggle.textContent = "Close Camera";
    setHint("Backend camera active. Running low-latency predictions.");
    startPolling();
  } catch (error) {
    setHint(error.message || "Could not start camera.");
  } finally {
    cameraToggle.disabled = false;
  }
}

async function stopCamera() {
  if (!config.cameraStopUrl) {
    return;
  }
  cameraToggle.disabled = true;
  try {
    await fetch(config.cameraStopUrl, { method: "POST" });
  } catch (_) {
    // Ignore stop errors and still reset UI state.
  }
  cameraRunning = false;
  cameraVideo.removeAttribute("src");
  cameraVideo.src = "";
  clearOverlay();
  cameraToggle.textContent = "Open Camera";
  setHint("Camera paused.");
  stopPolling();
  cameraToggle.disabled = false;
}

async function syncModelPreference() {
  const preferred = localStorage.getItem(MODEL_PREF_KEY);
  if (!preferred || !SUPPORTED_MODELS.includes(preferred)) {
    return;
  }

  try {
    const healthResponse = await fetch("/api/health", { cache: "no-store" });
    const healthData = await readJson(healthResponse);
    const backendModel = String(healthData.model_type || "").toLowerCase();

    if (healthResponse.ok && backendModel === preferred) {
      return;
    }

    await fetch("/api/model/switch", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ model_type: preferred }),
    });
  } catch (_) {
    // Ignore failures and continue with backend default.
  }
}

if (cameraToggle) {
  cameraToggle.addEventListener("click", async () => {
    if (cameraRunning) {
      await stopCamera();
    } else {
      await startCamera();
    }
  });
}

window.addEventListener("beforeunload", () => {
  stopPolling();
});

setHint("Starting backend stream...");
setMarkStatus("Ready.");
syncModelPreference().then(() => {
  startCamera();
});
