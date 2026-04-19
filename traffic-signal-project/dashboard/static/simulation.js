/* simulation.js — Multi-Intersection Grid UI */
"use strict";

const PHASE_COLORS = {
  0: { name: "NS Grn", ns: "#22c55e", ew: "#ef4444", badge: "#22c55e", badgeBg: "rgba(34,197,94,0.15)", badgeBorder: "rgba(34,197,94,0.35)" },
  1: { name: "NS Yel", ns: "#eab308", ew: "#ef4444", badge: "#eab308", badgeBg: "rgba(234,179,8,0.15)", badgeBorder: "rgba(234,179,8,0.35)" },
  2: { name: "EW Grn", ns: "#ef4444", ew: "#3b82f6", badge: "#3b82f6", badgeBg: "rgba(59,130,246,0.15)", badgeBorder: "rgba(59,130,246,0.35)" },
  3: { name: "EW Yel", ns: "#ef4444", ew: "#f97316", badge: "#f97316", badgeBg: "rgba(249,115,22,0.15)", badgeBorder: "rgba(249,115,22,0.35)" },
};

let state = {
  step: 0, episode: 1, avg_queue: 0, avg_wait: 0,
  transit_east: 0, transit_west: 0,
  intersections: [
    { phase: 0, phase_timer: 30, queues: [0,0,0,0], waiting_times: [0,0,0,0], vehicle_counts: [0,0,0,0], action: 0, action_name: "NS Green" },
    { phase: 0, phase_timer: 30, queues: [0,0,0,0], waiting_times: [0,0,0,0], vehicle_counts: [0,0,0,0], action: 0, action_name: "NS Green" }
  ],
  reward_history: [], queue_history: [],
  paused: false, running: false, max_steps: 1000,
};

const canvas = document.getElementById("intersection-canvas");
const ctx = canvas.getContext("2d");

const rCanvas = document.getElementById("reward-chart");
const qCanvas = document.getElementById("queue-chart");
[rCanvas, qCanvas].forEach(c => { c.width = c.offsetWidth || 300; c.height = 110; });

const es = new EventSource("/stream");
es.onmessage = (e) => {
  const d = JSON.parse(e.data);
  Object.assign(state, d);
  updateUI();
};
es.onerror = () => { document.getElementById("status-text").textContent = "Disconnected (Retrying...)"; };
fetch("/state").then(r => r.json()).then(d => { Object.assign(state, d); updateUI(); });

function updateUI() {
  const dot  = document.getElementById("status-dot");
  const stxt = document.getElementById("status-text");
  if (!state.running)     { dot.className = "status-dot";         stxt.textContent = "Idle"; }
  else if (state.paused)  { dot.className = "status-dot paused";  stxt.textContent = "Paused"; }
  else                    { dot.className = "status-dot running";  stxt.textContent = "Running"; }

  const i1 = state.intersections[0] || state.intersections[0];
  const i2 = state.intersections[1] || state.intersections[0];

  // Header Badge (Show I1 & I2 phase)
  const pc1 = PHASE_COLORS[i1.phase] || PHASE_COLORS[0];
  const pc2 = PHASE_COLORS[i2.phase] || PHASE_COLORS[0];
  const pb = document.getElementById("phase-badge");
  pb.textContent = `I1: ${pc1.name}  |  I2: ${pc2.name}`;
  pb.style.color = pc1.badge; pb.style.background = "rgba(255,255,255,0.05)"; pb.style.borderColor = "rgba(255,255,255,0.2)";

  // Global KPIs
  document.getElementById("step-counter").textContent = `Step ${state.step} / ${state.max_steps}  ·  Ep ${state.episode}`;
  const epReward = state.reward_history.length > 0 ? state.reward_history[state.reward_history.length-1] : 0;
  document.getElementById("kpi-reward").textContent  = epReward.toFixed(4);
  document.getElementById("kpi-queue").textContent   = (state.avg_queue || 0).toFixed(2);
  document.getElementById("kpi-wait").textContent    = (state.avg_wait || 0).toFixed(2) + "s";
  document.getElementById("kpi-episode").textContent = state.episode;

  document.getElementById("ai-action").textContent = `I1: ${i1.action_name}  |  I2: ${i2.action_name}`;

  document.getElementById("q-N").textContent = Math.round(i1.queues[0] + i2.queues[0]);
  document.getElementById("q-S").textContent = Math.round(i1.queues[1] + i2.queues[1]);
  document.getElementById("q-E").textContent = Math.round(i2.queues[2]);
  document.getElementById("q-W").textContent = Math.round(i1.queues[3]);

  // Combine queues for charts (Max visual is 60 now for sum)
  const maxQ = 60, maxW = 240;
  setBar("qb-N", "qbv-N", i1.queues[0] + i2.queues[0], maxQ);
  setBar("qb-S", "qbv-S", i1.queues[1] + i2.queues[1], maxQ);
  setBar("qb-E", "qbv-E", i2.queues[2], maxQ);
  setBar("qb-W", "qbv-W", i1.queues[3], maxQ);

  setBar("wb-N", "wbv-N", i1.waiting_times[0] + i2.waiting_times[0], maxW);
  setBar("wb-S", "wbv-S", i1.waiting_times[1] + i2.waiting_times[1], maxW);
  setBar("wb-E", "wbv-E", i2.waiting_times[2], maxW);
  setBar("wb-W", "wbv-W", i1.waiting_times[3], maxW);

  // Timer bar uses purely I1 phase timer
  const maxDur = 50;
  document.getElementById("timer-bar").style.width = Math.min(100, (i1.phase_timer / maxDur) * 100) + "%";
  document.getElementById("timer-val").textContent = i1.phase_timer + "s";

  drawGrid(state);
  drawMiniChart(rCanvas, state.reward_history,  "#3b82f6", "#1e40af");
  drawMiniChart(qCanvas, state.queue_history,   "#f97316", "#92400e");
}

function setBar(fillId, valId, value, max) {
  const pct = Math.min(100, (value / max) * 100);
  const el = document.getElementById(fillId);
  if (el) el.style.width = pct + "%";
  const label = document.getElementById(valId);
  if (label) label.textContent  = Math.round(value);
}

function roundRect(ctx, x, y, w, h, r) {
  ctx.beginPath();
  ctx.moveTo(x + r, y);
  ctx.arcTo(x + w, y, x + w, y + h, r);
  ctx.arcTo(x + w, y + h, x, y + h, r);
  ctx.arcTo(x, y + h, x, y, r);
  ctx.arcTo(x, y, x + w, y, r);
  ctx.closePath();
}

function drawGrid(s) {
  const W = canvas.width, H = canvas.height;
  const CY = H / 2;
  const ROAD_W = 80;
  const CX1 = W * 0.25 + 20;  // Intersection 1 Center
  const CX2 = W * 0.75 - 20;  // Intersection 2 Center

  ctx.clearRect(0, 0, W, H);
  const bg = ctx.createLinearGradient(0, 0, W, 0);
  bg.addColorStop(0, "#0f1f35"); bg.addColorStop(0.5, "#060d18"); bg.addColorStop(1, "#0f1f35");
  ctx.fillStyle = bg;
  ctx.fillRect(0, 0, W, H);

  // Draw Horizontal Main Street
  ctx.fillStyle = "#1a2535";
  ctx.fillRect(0, CY - ROAD_W/2, W, ROAD_W);

  // Draw Vertical Streets
  ctx.fillRect(CX1 - ROAD_W/2, 0, ROAD_W, H);
  ctx.fillRect(CX2 - ROAD_W/2, 0, ROAD_W, H);

  // Intersection Boxes
  ctx.fillStyle = "#1d2d42";
  ctx.fillRect(CX1 - ROAD_W/2, CY - ROAD_W/2, ROAD_W, ROAD_W);
  ctx.fillRect(CX2 - ROAD_W/2, CY - ROAD_W/2, ROAD_W, ROAD_W);

  // Centre Lines
  ctx.setLineDash([14, 10]);
  ctx.strokeStyle = "rgba(255,255,255,0.15)";
  ctx.lineWidth = 2;
  // Horiz
  ctx.beginPath(); ctx.moveTo(0, CY); ctx.lineTo(CX1 - ROAD_W/2, CY); ctx.stroke();
  ctx.beginPath(); ctx.moveTo(CX1 + ROAD_W/2, CY); ctx.lineTo(CX2 - ROAD_W/2, CY); ctx.stroke();
  ctx.beginPath(); ctx.moveTo(CX2 + ROAD_W/2, CY); ctx.lineTo(W, CY); ctx.stroke();
  // Verts
  ctx.beginPath(); ctx.moveTo(CX1, 0); ctx.lineTo(CX1, CY - ROAD_W/2); ctx.stroke();
  ctx.beginPath(); ctx.moveTo(CX1, CY + ROAD_W/2); ctx.lineTo(CX1, H); ctx.stroke();
  ctx.beginPath(); ctx.moveTo(CX2, 0); ctx.lineTo(CX2, CY - ROAD_W/2); ctx.stroke();
  ctx.beginPath(); ctx.moveTo(CX2, CY + ROAD_W/2); ctx.lineTo(CX2, H); ctx.stroke();
  ctx.setLineDash([]);

  // Loop through intersections to draw local cues
  for(let idx = 0; idx < 2; idx++) {
    const cx = idx === 0 ? CX1 : CX2;
    const iState = s.intersections[idx] || s.intersections[0];
    const pc = PHASE_COLORS[iState.phase] || PHASE_COLORS[0];
    const queues = iState.queues || [0,0,0,0];

    // Traffic Lights (Inline with lanes)
    const TL = [
      { x: cx - 22, y: CY - ROAD_W/2 - 36, color: pc.ns },  // North
      { x: cx - 22, y: CY + ROAD_W/2 + 18, color: pc.ns },  // South
      { x: cx + ROAD_W/2 + 18, y: CY - 18, color: pc.ew },  // East
      { x: cx - ROAD_W/2 - 36, y: CY - 18, color: pc.ew },  // West
    ];
    TL.forEach(tl => {
      ctx.fillStyle = "#0a0f1a"; ctx.strokeStyle = "#1e3a5f"; ctx.lineWidth = 1.5;
      roundRect(ctx, tl.x - 2, tl.y - 2, 18, 18, 4);
      ctx.fill(); ctx.stroke();
      ctx.globalAlpha = 0.35;
      ctx.fillStyle = tl.color; ctx.fillRect(tl.x - 8, tl.y - 8, 32, 32);
      ctx.globalAlpha = 1.0;
      ctx.beginPath(); ctx.arc(tl.x + 7, tl.y + 7, 7, 0, Math.PI * 2);
      ctx.fillStyle = tl.color; ctx.shadowColor = tl.color; ctx.shadowBlur = 14;
      ctx.fill(); ctx.shadowBlur = 0;
    });

    // Car Queues
    drawCarQueue(ctx, cx - 14, CY - ROAD_W/2 - 18, 0, -1, queues[0], pc.ns); // N
    drawCarQueue(ctx, cx - 14, CY + ROAD_W/2 + 18, 0, +1, queues[1], pc.ew); // S

    if (idx === 1) {
      // East arrival for Inter 2 (outer edge)
      drawCarQueue(ctx, cx + ROAD_W/2 + 18, CY - 10, +1, 0, queues[2], pc.ew);
    }
    if (idx === 0) {
      // West arrival for Inter 1 (outer edge)
      drawCarQueue(ctx, cx - ROAD_W/2 - 18, CY - 10, -1, 0, queues[3], pc.ns);
    }
    
    // Labels
    ctx.font = "bold 14px Inter, sans-serif"; ctx.fillStyle = "rgba(255,255,255,0.7)";
    ctx.textAlign = "center"; ctx.fillText("Intersection " + (idx+1), cx, CY + ROAD_W/2 + 70);
  }

  // --- CONNECTING CORRIDOR TRANSIT CARS ---
  // Transit East applies to cars leaving I1 going to I2.
  if (s.transit_east > 0) {
     drawCarQueue(ctx, CX1 + ROAD_W/2 + 10, CY - 10, +1, 0, s.transit_east, "#aaaaaa"); // Draw between I1 and I2, moving East
     // Wait, if they are moving East, they are on the bottom half of the road? No, moving East is right, so right lane = bottom lane.
     // In right-hand traffic, driving East means you are on the SOUTH half of the East-West road.
  }
  // Transit West applies to cars leaving I2 going to I1.
  if (s.transit_west > 0) {
     drawCarQueue(ctx, CX2 - ROAD_W/2 - 10, CY - 24, -1, 0, s.transit_west, "#aaaaaa"); // Draw between I1 and I2, moving West
  }
}

// Fixed connecting corridor transit car placement
function drawGridFixedTransit(s) {
    originalDrawGrid(s);
    // Overlay transits
    const W = canvas.width, H = canvas.height;
    const CY = H / 2, ROAD_W = 80;
    const CX1 = W * 0.25 + 20, CX2 = W * 0.75 - 20;
    
    // Cars going East (I1 -> I2) are on the bottom half of the road: y = CY + 14
    if (s.transit_east > 0) {
       drawCarQueue(ctx, CX1 + ROAD_W/2 + 10, CY + 14, +1, 0, s.transit_east, "#93c5fd");
    }
    // Cars going West (I2 -> I1) are on the top half of the road: y = CY - 24
    if (s.transit_west > 0) {
       drawCarQueue(ctx, CX2 - ROAD_W/2 - 10, CY - 24, -1, 0, s.transit_west, "#fdba74");
    }
}
// Substitute original drawGrid with the fix
const originalDrawGrid = drawGrid;
drawGrid = drawGridFixedTransit;

function drawCarQueue(ctx, startX, startY, dx, dy, countF, phaseColor) {
  const count = Math.min(Math.round(countF), 18);
  const CAR_W = 20, CAR_H = 12, GAP = 4;
  const step = (dy !== 0) ? CAR_H + GAP : CAR_W + GAP;
  for (let i = 0; i < count; i++) {
    const cx = startX + dx * i * step;
    const cy = startY + dy * i * step;
    ctx.save();
    ctx.shadowColor = phaseColor === "#f97316" ? "#f97316" : "#3b82f6";
    ctx.shadowBlur = 4;
    
    const carW = dy !== 0 ? CAR_W : CAR_H;
    const carH = dy !== 0 ? CAR_H : CAR_W;

    ctx.fillStyle = phaseColor;
    if (phaseColor !== "#93c5fd" && phaseColor !== "#fdba74") {
        if (phaseColor === "#f97316") {
            ctx.fillStyle = i < 3 ? "#f97316" : "#c2410c";
        } else if (phaseColor === "#22c55e" || phaseColor === "#3b82f6") {
            ctx.fillStyle = i === 0 ? "#60a5fa" : (i < 3 ? "#3b82f6" : "#1d4ed8");
        }
    }

    roundRect(ctx, cx, cy, carW, carH, 3);
    ctx.fill();
    ctx.fillStyle = "rgba(186,230,253,0.6)";
    if (dy !== 0) ctx.fillRect(cx + 3, cy + 2, carW - 6, carH - 6);
    else ctx.fillRect(cx + 2, cy + 3, carW - 6, carH - 6);
    ctx.shadowBlur = 0;
    ctx.restore();
  }
}

function drawMiniChart(cv, data, lineColor, fillColor) {
  const W = cv.width = cv.offsetWidth || 300;
  const H = cv.height;
  const ct = cv.getContext("2d");
  ct.clearRect(0, 0, W, H);
  if (!data || data.length < 2) return;

  const pts = data.slice(-60);
  const minV = Math.min(...pts);
  const maxV = Math.max(...pts);
  const range = maxV - minV || 1;
  const PAD = { t: 10, b: 20, l: 40, r: 10 };
  const toX = (i)   => PAD.l + (i / (pts.length - 1)) * (W - PAD.l - PAD.r);
  const toY = (v) => PAD.t + (H - PAD.t - PAD.b) - ((v - minV) / range) * (H - PAD.t - PAD.b);

  ct.beginPath(); ct.moveTo(toX(0), toY(pts[0]));
  pts.forEach((v, i) => ct.lineTo(toX(i), toY(v)));
  ct.strokeStyle = lineColor; ct.lineWidth = 2; ct.stroke();
}

function startSim() { fetch("/start", { method: "POST" }); }
function pauseSim() { fetch("/pause", { method: "POST" }); }
function resetSim() { fetch("/reset", { method: "POST" }); }
function setSpeed(val) { fetch("/speed", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ speed: parseInt(val) }) }); }

window.addEventListener("load", () => {
  setTimeout(startSim, 600);
});
