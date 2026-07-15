import { useEffect, useState, useCallback } from 'react'
import './App.css'

const API_BASE = 'http://127.0.0.1:5000'
const TOKEN_KEY = 'aios_token'

const RESOURCES = [
  { key: 'jobs', title: 'Jobs', path: '/dashboard/jobs', columns: [
    { key: 'company', label: 'Company' },
    { key: 'role', label: 'Role' },
    { key: 'resume_used', label: 'Resume', fmt: v => v || '—' },
    { key: 'status', label: 'Status' },
    { key: 'applied_at', label: 'Applied' },
    { key: 'follow_up_at', label: 'Follow-up', fmt: v => v || '—' },
  ] },
  { key: 'leads', title: 'Leads', path: '/dashboard/leads', columns: [
    { key: 'studio_name', label: 'Studio' },
    { key: 'location', label: 'Location', fmt: v => v || '—' },
    { key: 'outreach_sent', label: 'Outreach', fmt: v => v ? '✓' : '—' },
    { key: 'response', label: 'Response', fmt: v => v || '—' },
    { key: 'created_at', label: 'Created' },
  ] },
  { key: 'fitness', title: 'Fitness', path: '/dashboard/fitness', columns: [
    { key: 'activity', label: 'Activity' },
    { key: 'duration_min', label: 'Duration (min)', fmt: v => v ?? '—' },
    { key: 'notes', label: 'Notes', fmt: v => v || '—' },
    { key: 'logged_at', label: 'Logged' },
  ] },
  { key: 'study', title: 'Study', path: '/dashboard/study', columns: [
    { key: 'topic', label: 'Topic' },
    { key: 'duration_min', label: 'Duration (min)', fmt: v => v ?? '—' },
    { key: 'notes', label: 'Notes', fmt: v => v || '—' },
    { key: 'logged_at', label: 'Logged' },
  ] },
  { key: 'habits', title: 'Habits', path: '/dashboard/habits', columns: [
    { key: 'name', label: 'Habit' },
    { key: 'logged_at', label: 'Date' },
    { key: 'done', label: 'Done', fmt: v => v ? '✓' : '—' },
  ] },
]

const STAT_CARDS = [
  { key: 'jobs_applied', label: 'Jobs Applied' },
  { key: 'leads', label: 'Leads' },
  { key: 'study_sessions_this_week', label: 'Study (7d)' },
  { key: 'habit_streak', label: 'Habit Streak' },
  { key: 'pending_reminders', label: 'Pending Reminders' },
]

function useToken() {
  const [token, setToken] = useState(() => localStorage.getItem(TOKEN_KEY) || '')

  useEffect(() => {
    if (token) return
    let cancelled = false
    fetch(`${API_BASE}/token`)
      .then(r => r.json())
      .then(d => {
        if (!cancelled && d.token) {
          localStorage.setItem(TOKEN_KEY, d.token)
          setToken(d.token)
        }
      })
      .catch(() => {}) // manual entry still available below
    return () => { cancelled = true }
  }, [token])

  const save = useCallback((t) => {
    localStorage.setItem(TOKEN_KEY, t)
    setToken(t)
  }, [])

  return [token, save]
}

function Table({ columns, rows }) {
  if (rows === null) return <div className="loading-state">Loading</div>
  if (rows.error) return <div className="error-state">Error: {rows.error}</div>
  if (rows.length === 0) return <div className="empty-state">Nothing here yet.</div>
  return (
    <table>
      <thead>
        <tr>{columns.map(c => <th key={c.key}>{c.label}</th>)}</tr>
      </thead>
      <tbody>
        {rows.map((row, i) => (
          <tr key={row.id ?? i}>
            {columns.map(c => (
              <td key={c.key}>{c.fmt ? c.fmt(row[c.key]) : (row[c.key] ?? '—')}</td>
            ))}
          </tr>
        ))}
      </tbody>
    </table>
  )
}

function App() {
  const [token, saveToken] = useToken()
  const [tokenInput, setTokenInput] = useState('')
  const [overview, setOverview] = useState(null) // null = loading, {error} = failed
  const [data, setData] = useState({}) // { [resourceKey]: rows | null | {error} }
  const [reloadTick, setReloadTick] = useState(0)

  const authedFetch = useCallback(async (path) => {
    const res = await fetch(`${API_BASE}${path}`, {
      headers: { Authorization: `Bearer ${token}` },
    })
    if (!res.ok) throw new Error(`HTTP ${res.status}`)
    return res.json()
  }, [token])

  useEffect(() => {
    if (!token) return

    setOverview(null)
    authedFetch('/dashboard/overview')
      .then(setOverview)
      .catch(e => setOverview({ error: e.message }))

    for (const r of RESOURCES) {
      setData(prev => ({ ...prev, [r.key]: null }))
      authedFetch(r.path)
        .then(rows => setData(prev => ({ ...prev, [r.key]: rows })))
        .catch(e => setData(prev => ({ ...prev, [r.key]: { error: e.message } })))
    }
  }, [token, reloadTick, authedFetch])

  return (
    <div className="shell">
      <div className="topbar">
        <h1>AIOS Dashboard</h1>
        <div className="token-box">
          {token ? (
            <button type="button" onClick={() => setReloadTick(t => t + 1)}>
              Refresh
            </button>
          ) : (
            <>
              <input
                placeholder="paste JWT token"
                value={tokenInput}
                onChange={e => setTokenInput(e.target.value)}
              />
              <button type="button" onClick={() => saveToken(tokenInput.trim())}>
                Use token
              </button>
            </>
          )}
        </div>
      </div>

      {!token && (
        <div className="empty-state">
          Fetching a token automatically — or paste one above.
        </div>
      )}

      {token && (
        <>
          <div className="stat-row">
            {STAT_CARDS.map(s => (
              <div className="stat-card" key={s.key}>
                <div className="label">{s.label}</div>
                <div className="value">
                  {overview === null
                    ? '…'
                    : overview.error
                      ? '—'
                      : overview[s.key]}
                </div>
              </div>
            ))}
          </div>

          {RESOURCES.map(r => (
            <section className="panel" key={r.key}>
              <h2>{r.title}</h2>
              <div className="panel-body">
                <Table columns={r.columns} rows={data[r.key] ?? null} />
              </div>
            </section>
          ))}
        </>
      )}
    </div>
  )
}

export default App
