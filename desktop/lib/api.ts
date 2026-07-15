// Single source of truth for the backend origin — every fetch in this app
// (chat, dashboard, token) must import this rather than hardcoding the URL.
export const API_BASE = "http://127.0.0.1:5000";
