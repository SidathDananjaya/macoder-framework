const API_BASE = "http://127.0.0.1:8000/api";

export async function getSessionSummary() {
  const response = await fetch(`${API_BASE}/session-summary`);

  return response.json();
}

export async function getSessionData() {
  const response = await fetch(`${API_BASE}/session-data`);

  return response.json();
}
