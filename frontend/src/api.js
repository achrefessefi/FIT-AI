// frontend/src/api.js
const API_BASE = import.meta.env.VITE_API_BASE || "http://localhost:8000";

export async function fetchDashboard() {
  const res = await fetch(`${API_BASE}/api/dashboard`);
  if (!res.ok) throw new Error(`Failed to fetch /api/dashboard: ${res.status}`);
  return res.json();
}
export async function fetchPlan() {
  const res = await fetch(`${API_BASE}/api/plan`);
  if (!res.ok) throw new Error("plan fetch failed");
  return res.json();
}