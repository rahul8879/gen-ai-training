import React, { useMemo, useState } from 'react'
import { createRoot } from 'react-dom/client'
import axios from 'axios'

const API_BASE = '/api'

function ChatPane({ title, endpoint }) {
  const [messages, setMessages] = useState([])
  const [input, setInput] = useState('')
  const [loading, setLoading] = useState(false)

  const send = async () => {
    const trimmed = input.trim()
    if (!trimmed) return
    const newMsgs = [...messages, { role: 'user', content: trimmed }]
    setMessages(newMsgs)
    setInput('')
    setLoading(true)
    try {
      const res = await axios.post(`${API_BASE}${endpoint}`, { messages: newMsgs })
      const msg = res.data?.message
      if (msg) setMessages([...newMsgs, msg])
    } catch (e) {
      setMessages([...newMsgs, { role: 'assistant', content: `Error: ${e}` }])
    } finally {
      setLoading(false)
    }
  }

  return (
    <div style={{ display: 'flex', flexDirection: 'column', height: '100%' }}>
      <div style={{ padding: '8px 12px', borderBottom: '1px solid #ddd', background: '#fafafa' }}>
        <strong>{title}</strong>
      </div>
      <div style={{ flex: 1, padding: 12, overflow: 'auto' }}>
        {messages.map((m, i) => (
          <div key={i} style={{ margin: '6px 0' }}>
            <div style={{ fontSize: 12, color: '#666' }}>{m.role}</div>
            <div style={{ whiteSpace: 'pre-wrap' }}>{m.content}</div>
          </div>
        ))}
        {loading && <div style={{ color: '#999' }}>Thinking…</div>}
      </div>
      <div style={{ display: 'flex', gap: 8, padding: 12, borderTop: '1px solid #eee' }}>
        <input value={input} onChange={(e)=>setInput(e.target.value)} onKeyDown={(e)=>{ if(e.key==='Enter') send() }}
               placeholder="Type your message…" style={{ flex: 1, padding: 8, border: '1px solid #ccc', borderRadius: 4 }}/>
        <button onClick={send} disabled={loading} style={{ padding: '8px 14px' }}>Send</button>
      </div>
    </div>
  )
}

function App() {
  const tabs = useMemo(() => ([
    { key: 'agent', label: 'LangGraph Agent', endpoint: '/agent/chat' },
    { key: 'retail', label: 'Retail Agent', endpoint: '/retail/chat' }
  ]), [])
  const [active, setActive] = useState(tabs[0].key)
  const current = tabs.find(t => t.key === active)

  return (
    <div style={{ height: '100vh', display: 'grid', gridTemplateRows: '48px 1fr' }}>
      <div style={{ display: 'flex', gap: 8, alignItems: 'center', padding: '0 12px', borderBottom: '1px solid #ddd' }}>
        <div style={{ fontWeight: 700 }}>GenAI Playground</div>
        <div style={{ marginLeft: 16, display: 'flex', gap: 6 }}>
          {tabs.map(t => (
            <button key={t.key} onClick={()=>setActive(t.key)}
              style={{ padding: '6px 10px', border: '1px solid #ccc', borderBottom: active===t.key?'2px solid #333':'1px solid #ccc', background: active===t.key?'#fff':'#f7f7f7', cursor: 'pointer' }}>
              {t.label}
            </button>
          ))}
        </div>
      </div>
      <div>
        <ChatPane title={current.label} endpoint={current.endpoint} />
      </div>
    </div>
  )
}

createRoot(document.getElementById('root')).render(<App />)

