/* Simple Node API server exposing agent endpoints */
const path = require('path');
const fs = require('fs');
const express = require('express');
const cors = require('cors');
const dotenv = require('dotenv');

// Load .env from repo root if present
const rootEnv = path.resolve(__dirname, '../../.env');
if (fs.existsSync(rootEnv)) dotenv.config({ path: rootEnv });
else dotenv.config();

const { OpenAI } = require('openai');
const openai = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });

const app = express();
app.use(cors());
app.use(express.json({ limit: '1mb' }));

const MODEL = process.env.LLM_MODEL || 'gpt-4o-mini';

// ---------- Utilities ----------
function readJSON(p) {
  return JSON.parse(fs.readFileSync(p, 'utf-8'));
}

function loadCSV(filePath) {
  if (!fs.existsSync(filePath)) return [];
  const text = fs.readFileSync(filePath, 'utf-8').trim();
  const [headerLine, ...lines] = text.split(/\r?\n/);
  const headers = headerLine.split(',');
  return lines.map((line) => {
    const vals = line.split(',');
    const obj = {};
    headers.forEach((h, i) => (obj[h] = vals[i]));
    return obj;
  });
}

function toNumber(x) {
  const n = Number(x);
  return Number.isFinite(n) ? n : 0;
}

// ---------- General tools (calculator, FAQ) ----------
const FAQ_PATH = path.resolve(__dirname, '../../05_GenAI/langgraph_agent/data/faq.json');

function safeEval(expression) {
  // Very limited safe eval: only arithmetic ops
  if (!/^[0-9+\-*/().,%\s]*$/.test(expression)) throw new Error('Unsupported characters');
  // eslint-disable-next-line no-new-func
  return Function(`return (${expression})`)();
}

const generalTools = [
  {
    type: 'function',
    function: {
      name: 'calculator',
      description: 'Evaluate a basic arithmetic expression like (2+3*4)/5.',
      parameters: {
        type: 'object',
        properties: { expression: { type: 'string' } },
        required: ['expression']
      }
    }
  },
  {
    type: 'function',
    function: {
      name: 'faq_lookup',
      description: 'Answer from a curated internal FAQ knowledge base.',
      parameters: {
        type: 'object',
        properties: { question: { type: 'string' } },
        required: ['question']
      }
    }
  }
];

async function runGeneralTool(name, args) {
  if (name === 'calculator') {
    return String(safeEval(String(args.expression || '0')));
  }
  if (name === 'faq_lookup') {
    if (!fs.existsSync(FAQ_PATH)) return 'FAQ not available';
    const faqs = readJSON(FAQ_PATH);
    const q = String(args.question || '').toLowerCase();
    const score = (text) => {
      const t = new Set(text.toLowerCase().split(/\W+/).filter((w) => w.length > 2));
      const qset = new Set(q.split(/\W+/).filter((w) => w.length > 2));
      const inter = [...qset].filter((w) => t.has(w)).length;
      return qset.size ? inter / qset.size : 0;
    };
    let best = null, bestScore = 0;
    for (const item of faqs) {
      const s = Math.max(score(item.q || ''), score(item.a || ''));
      if (s > bestScore) { bestScore = s; best = item; }
    }
    if (best && bestScore >= 0.2) return `MatchScore=${bestScore.toFixed(2)}\nQ: ${best.q}\nA: ${best.a}`;
    return 'No good match found in FAQ.';
  }
  throw new Error(`Unknown tool: ${name}`);
}

// ---------- Retail tools ----------
const SALES_CSV = path.resolve(__dirname, '../../05_GenAI/retail_agent/data/sales.csv');
const INV_CSV = path.resolve(__dirname, '../../05_GenAI/retail_agent/data/inventory.csv');

function loadSales() {
  const rows = loadCSV(SALES_CSV);
  return rows.map((r) => ({
    date: new Date(r.date),
    order_id: r.order_id,
    sku: r.sku,
    category: r.category,
    unit_price: toNumber(r.unit_price),
    quantity: toNumber(r.quantity),
    revenue: toNumber(r.unit_price) * toNumber(r.quantity)
  }));
}

function loadInventory() {
  const rows = loadCSV(INV_CSV);
  return rows.map((r) => ({
    sku: r.sku,
    category: r.category,
    unit_price: toNumber(r.unit_price),
    on_hand: toNumber(r.on_hand),
    reorder_point: toNumber(r.reorder_point)
  }));
}

const retailTools = [
  ...generalTools,
  {
    type: 'function',
    function: {
      name: 'retail_sales_summary',
      description: 'Summarize sales in a date range with revenue, units, top SKUs/categories.',
      parameters: {
        type: 'object',
        properties: {
          start: { type: 'string', description: 'YYYY-MM-DD' },
          end: { type: 'string', description: 'YYYY-MM-DD' },
          top_n: { type: 'number', default: 5 }
        }
      }
    }
  },
  {
    type: 'function',
    function: {
      name: 'retail_inventory_status',
      description: 'Return low-stock items (on_hand <= reorder_point).',
      parameters: { type: 'object', properties: {} }
    }
  },
  {
    type: 'function',
    function: {
      name: 'retail_price_optimize',
      description: 'Suggest price within ±10% that maximizes revenue using simple elasticity.',
      parameters: {
        type: 'object',
        properties: {
          skus: { type: 'array', items: { type: 'string' } },
          elasticity: { type: 'number', default: -1.2 }
        }
      }
    }
  },
  {
    type: 'function',
    function: {
      name: 'retail_markdown_report',
      description: 'Compose a concise markdown report from provided findings JSON.',
      parameters: { type: 'object', additionalProperties: true }
    }
  }
];

function groupBy(arr, key) {
  return arr.reduce((acc, item) => {
    const k = item[key];
    acc[k] = acc[k] || [];
    acc[k].push(item);
    return acc;
  }, {});
}

async function runRetailTool(name, args) {
  if (name === 'retail_sales_summary') {
    const top_n = Number(args.top_n || 5);
    const start = args.start ? new Date(args.start) : null;
    const end = args.end ? new Date(args.end) : null;
    let rows = loadSales();
    if (start) rows = rows.filter((r) => r.date >= start);
    if (end) rows = rows.filter((r) => r.date <= end);
    const totals = {
      orders: new Set(rows.map((r) => r.order_id)).size,
      units: rows.reduce((s, r) => s + r.quantity, 0),
      revenue: rows.reduce((s, r) => s + r.revenue, 0)
    };
    const bySku = Object.entries(groupBy(rows, 'sku')).map(([sku, items]) => ({
      sku, revenue: items.reduce((s, r) => s + r.revenue, 0), quantity: items.reduce((s, r) => s + r.quantity, 0)
    })).sort((a, b) => b.revenue - a.revenue).slice(0, top_n);
    const byCat = Object.entries(groupBy(rows, 'category')).map(([category, items]) => ({
      category, revenue: items.reduce((s, r) => s + r.revenue, 0), quantity: items.reduce((s, r) => s + r.quantity, 0)
    })).sort((a, b) => b.revenue - a.revenue).slice(0, top_n);
    return JSON.stringify({ totals, top_skus: bySku, top_categories: byCat });
  }
  if (name === 'retail_inventory_status') {
    const inv = loadInventory();
    const low = inv.filter((r) => r.on_hand <= r.reorder_point);
    return JSON.stringify({ low_stock: low, total_skus: inv.length, low_count: low.length });
  }
  if (name === 'retail_price_optimize') {
    const inv = loadInventory();
    const sales = loadSales();
    let sel = Array.isArray(args.skus) && args.skus.length ? new Set(args.skus) : new Set(inv.sort((a,b)=>b.unit_price-a.unit_price).slice(0,5).map(r=>r.sku));
    const elasticity = typeof args.elasticity === 'number' ? args.elasticity : -1.2;
    function avgQty(sku){
      const rows = sales.filter(r=>r.sku===sku);
      if(!rows.length) return 5;
      return rows.reduce((s,r)=>s+r.quantity,0)/rows.length;
    }
    const results = [];
    for (const row of inv.filter(r=>sel.has(r.sku))) {
      const p0 = row.unit_price; const q0 = Math.max(1, avgQty(row.sku));
      let best = { price: p0, revenue: p0*q0 };
      for (let i=0;i<=20;i++){
        const p = 0.9*p0 + (i/20)*(0.2*p0);
        const q = q0 * Math.pow(p/p0, elasticity);
        const r = p*q; if (r>best.revenue) best={price:p, revenue:r};
      }
      results.push({ sku: row.sku, current_price: p0, suggested_price: Number(best.price.toFixed(2)), revenue_baseline: Number((p0*q0).toFixed(2)), revenue_suggested: Number(best.revenue.toFixed(2)), delta: Number((best.revenue - p0*q0).toFixed(2)) });
    }
    return JSON.stringify({ pricing: results, assumptions: { elasticity, band: '+/-10%' } });
  }
  if (name === 'retail_markdown_report') {
    const p = args || {};
    const lines = ['# Retail Summary Report'];
    if (p.totals) {
      lines.push('## Overview');
      lines.push(`- Orders: ${p.totals.orders}`);
      lines.push(`- Units: ${p.totals.units}`);
      lines.push(`- Revenue: $${Number(p.totals.revenue||0).toFixed(2)}`);
      lines.push('');
    }
    if (Array.isArray(p.top_skus)) {
      lines.push('## Top SKUs');
      p.top_skus.slice(0,10).forEach(r=>lines.push(`- ${r.sku}: $${Number(r.revenue).toFixed(2)} | units=${r.quantity}`));
      lines.push('');
    }
    if (Array.isArray(p.top_categories)) {
      lines.push('## Top Categories');
      p.top_categories.slice(0,10).forEach(r=>lines.push(`- ${r.category}: $${Number(r.revenue).toFixed(2)} | units=${r.quantity}`));
      lines.push('');
    }
    if (Array.isArray(p.low_stock)) {
      lines.push('## Low Stock Alerts');
      p.low_stock.slice(0,15).forEach(r=>lines.push(`- ${r.sku} (on_hand=${r.on_hand}, ROP=${r.reorder_point})`));
      lines.push('');
    }
    if (Array.isArray(p.pricing)) {
      lines.push('## Pricing Suggestions');
      p.pricing.slice(0,10).forEach(r=>lines.push(`- ${r.sku}: ${r.current_price} -> ${r.suggested_price} | ΔRev=$${Number(r.delta).toFixed(2)}`));
      if (p.assumptions) lines.push(`Assumptions: elasticity=${p.assumptions.elasticity} within ${p.assumptions.band}`);
      lines.push('');
    }
    if (lines.length === 1) lines.push('(No findings)');
    return lines.join('\n');
  }
  // Fallback to general tools
  return runGeneralTool(name, args);
}

// ---------- Tool loop runner ----------
async function toolLoop(messages, tools, runner) {
  let toolMessages = [];
  for (let iter = 0; iter < 6; iter++) {
    const resp = await openai.chat.completions.create({
      model: MODEL,
      messages: [...messages, ...toolMessages],
      tools,
      tool_choice: 'auto'
    });
    const msg = resp.choices[0].message;
    if (msg.tool_calls && msg.tool_calls.length) {
      const newToolMsgs = [];
      for (const call of msg.tool_calls) {
        const args = call.function?.arguments ? JSON.parse(call.function.arguments) : {};
        const result = await runner(call.function.name, args);
        newToolMsgs.push({ role: 'tool', tool_call_id: call.id, name: call.function.name, content: String(result) });
      }
      toolMessages.push({ role: 'assistant', content: msg.content || '', tool_calls: msg.tool_calls });
      toolMessages.push(...newToolMsgs);
      continue;
    }
    return msg;
  }
  return { role: 'assistant', content: 'Stopped after max tool iterations.' };
}

// ---------- Routes ----------
app.post('/api/agent/chat', async (req, res) => {
  try {
    const { messages = [] } = req.body || {};
    const sys = { role: 'system', content: 'You are a pragmatic AI assistant. Use tools when helpful. Always finish with "Final Answer:".' };
    const msg = await toolLoop([sys, ...messages], generalTools, runGeneralTool);
    res.json({ message: msg });
  } catch (e) {
    res.status(500).json({ error: String(e) });
  }
});

app.post('/api/retail/chat', async (req, res) => {
  try {
    const { messages = [] } = req.body || {};
    const sys = { role: 'system', content: 'You are a senior retail analytics assistant. Use sales, inventory, and pricing tools; then return a concise markdown report with Final Answer.' };
    const msg = await toolLoop([sys, ...messages], retailTools, runRetailTool);
    res.json({ message: msg });
  } catch (e) {
    res.status(500).json({ error: String(e) });
  }
});

const PORT = process.env.PORT || 3001;
app.listen(PORT, () => {
  console.log(`API server on http://localhost:${PORT}`);
});

