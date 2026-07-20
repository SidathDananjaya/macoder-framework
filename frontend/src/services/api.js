const API_BASE = "http://127.0.0.1:8000/api";

export async function getSessionSummary() {
  const response = await fetch(`${API_BASE}/session-summary`);

  return response.json();
}

export async function getSessionData() {
  const response = await fetch(`${API_BASE}/session-data`);

  return response.json();
}

// Phase 7 - persist the current recording as a flat CSV.
export async function exportSessionCsv() {
  const response = await fetch(`${API_BASE}/session/export`, {
    method: "POST"
  });

  return response.json();
}

// Phase 7 - persist the current recording as structured JSON (report input).
export async function exportSessionJson() {
  const response = await fetch(`${API_BASE}/session/export/json`, {
    method: "POST"
  });

  return response.json();
}

// Phase 8 - generate the interview report for the current live recording.
export async function getSessionReport() {
  const response = await fetch(`${API_BASE}/session/report`);

  return response.json();
}

// Phase 8 - persist the current session's report to disk.
export async function exportSessionReport() {
  const response = await fetch(`${API_BASE}/session/report/export`, {
    method: "POST"
  });

  return response.json();
}

// Phase 9 - local LLM (Ollama) interpretation of the current live recording.
export async function getSessionInterpretation() {
  const response = await fetch(`${API_BASE}/session/interpret`, {
    method: "POST"
  });

  return response.json();
}
