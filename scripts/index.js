const cameraVideo = document.getElementById("cameraVideo");
const cameraToggle = document.getElementById("cameraToggle");
const markNow = document.getElementById("markNow");
const cameraHint = document.getElementById("cameraHint");
const markStatus = document.getElementById("markStatus");
const todayDate = document.getElementById("todayDate");
const identifiedName = document.getElementById("identifiedName");
const identifiedStatus = document.getElementById("identifiedStatus");
const presentList = document.getElementById("presentList");
const absentList = document.getElementById("absentList");

const config = window.APP_CONFIG || {};
let cameraRunning = false;
let latestTimer = null;
let attendanceTimer = null;
let lastPrediction = null;
let markBusy = false;
const MODEL_PREF_KEY = "attsystem_selected_model";
const SUPPORTED_MODELS = ["cbir_method1", "cbir_method2"];

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

function canManuallyMark(prediction) {
  return Boolean(
    prediction &&
    prediction.status === "ok" &&
    prediction.accepted &&
    prediction.raw_name &&
    prediction.raw_name !== "unknown",
  );
}

function updateMarkButtonState(prediction) {
  if (!markNow) {
    return;
  }

  const allowed = canManuallyMark(prediction);
  markNow.disabled = !allowed || markBusy;
  markNow.textContent = allowed
    ? "Mark Verified Presence"
    : "Waiting for Verification";
}

function renderAttendance(attendance) {
  if (!attendance) {
    return;
  }

  setText(todayDate, attendance.date || new Date().toISOString().slice(0, 10));

  if (presentList) {
    presentList.innerHTML = "";
    const present = Array.isArray(attendance.present) ? attendance.present : [];
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
  } catch (error) {
    setMarkStatus(error.message || "Could not load attendance.");
  }
}

async function markAttendance(name, confidence) {
  if (!name || !config.attendanceMarkUrl) {
    return;
  }

  markBusy = true;
  updateMarkButtonState(lastPrediction);
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
  } finally {
    markBusy = false;
    updateMarkButtonState(lastPrediction);
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
    lastPrediction = prediction;

    const name = prediction.name || "Waiting...";
    const status = prediction.message || "No prediction yet.";

    setText(identifiedName, name);
    setText(identifiedStatus, status);

    setHint(data.running ? "Camera live." : "Camera is paused.");

    updateMarkButtonState(prediction);
  } catch (error) {
    setHint(error.message || "Could not read live prediction.");
  }
}

function startPolling() {
  if (!latestTimer) {
    latestTimer = setInterval(fetchLatest, 450);
  }
  if (!attendanceTimer) {
    attendanceTimer = setInterval(fetchAttendance, 2500);
  }
  fetchLatest();
  fetchAttendance();
}

function stopPolling() {
  if (latestTimer) {
    clearInterval(latestTimer);
    latestTimer = null;
  }
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

if (markNow) {
  markNow.addEventListener("click", async () => {
    if (!canManuallyMark(lastPrediction)) {
      setMarkStatus("Verify the identity on screen before marking attendance.");
      return;
    }
    await markAttendance(lastPrediction.raw_name, lastPrediction.confidence);
  });
}

window.addEventListener("beforeunload", () => {
  stopPolling();
});

setHint("Starting backend stream...");
setMarkStatus("Ready.");
updateMarkButtonState(lastPrediction);
syncModelPreference().then(() => {
  startCamera();
});
