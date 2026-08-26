let gaugeChart, perFamilyChart, trainingCurveChart;

// Honest per-scan indicator of which LLM provider (if any) actually
// generated Agent 1's reasoning text -- so an "AI-powered" claim is only
// ever made when it's true for that specific scan, not implied
// unconditionally by the presence of reasoning text.
function reasoningSourceBadge(source) {
    const labels = {
        groq: ["🤖 Groq Live", "#A6E3A1"],
        ollama: ["🖥️ Ollama Live (local)", "#A6E3A1"],
        template: ["📋 Template Fallback", "#F9AE6B"],
    };
    const entry = labels[source];
    if (!entry) return "";
    const [label, color] = entry;
    return `<span style="display:inline-block;font-size:11px;padding:2px 8px;border-radius:4px;margin-bottom:6px;background:${color}22;color:${color};border:1px solid ${color}66;">${label}</span><br>`;
}

function initCharts() {
    Chart.defaults.color = '#94a3b8';
    Chart.defaults.font.family = 'Inter, Segoe UI';

    const ctxGauge = document.getElementById('gaugeChart').getContext('2d');
    gaugeChart = new Chart(ctxGauge, {
        type: 'doughnut',
        data: {
            labels: ['AI Confidence', 'Doubt'],
            datasets: [{
                data: [0, 100],
                backgroundColor: ['#334155', 'transparent'],
                borderWidth: 0,
                circumference: 180,
                rotation: 270
            }]
        },
        options: { cutout: '75%', plugins: { legend: { display: false } }, responsive: true, maintainAspectRatio: false }
    });
}

function renderPerFamilyChart(perClass) {
    const canvas = document.getElementById('perFamilyChart');
    if (!canvas || !perClass) return;
    if (perFamilyChart) perFamilyChart.destroy();

    perFamilyChart = new Chart(canvas.getContext('2d'), {
        type: 'bar',
        data: {
            labels: perClass.map(c => c.label),
            datasets: [{
                label: 'F1 Score (%)',
                data: perClass.map(c => c.f1_score),
                backgroundColor: perClass.map(c => c.label === 'benign' ? 'rgba(74, 222, 128, 0.6)' : 'rgba(192, 132, 252, 0.6)'),
                borderColor: perClass.map(c => c.label === 'benign' ? '#4ade80' : '#c084fc'),
                borderWidth: 1,
                borderRadius: 4
            }]
        },
        options: {
            indexAxis: 'y',
            plugins: { legend: { display: false } },
            scales: {
                x: { min: 0, max: 100, grid: { color: '#1e293b' }, ticks: { color: '#94a3b8' } },
                y: { grid: { display: false }, ticks: { color: '#cbd5e1' } }
            },
            responsive: true, maintainAspectRatio: false
        }
    });
}

function renderTrainingCurveChart(history) {
    const canvas = document.getElementById('trainingCurveChart');
    if (!canvas || !history || !history.length) return;
    if (trainingCurveChart) trainingCurveChart.destroy();

    trainingCurveChart = new Chart(canvas.getContext('2d'), {
        type: 'line',
        data: {
            labels: history.map(h => h.epoch),
            datasets: [
                {
                    label: 'Val Accuracy (%)',
                    data: history.map(h => h.val_accuracy),
                    borderColor: '#22d3ee',
                    backgroundColor: 'rgba(34, 211, 238, 0.1)',
                    tension: 0.3,
                    pointRadius: 0
                },
                {
                    label: 'Val F1 Macro (%)',
                    data: history.map(h => h.val_f1_macro),
                    borderColor: '#4ade80',
                    backgroundColor: 'rgba(74, 222, 128, 0.1)',
                    tension: 0.3,
                    pointRadius: 0
                }
            ]
        },
        options: {
            plugins: { legend: { position: 'top', labels: { color: '#cbd5e1' } } },
            scales: {
                x: { title: { display: true, text: 'Epoch', color: '#94a3b8' }, grid: { display: false }, ticks: { color: '#94a3b8' } },
                y: { min: 0, max: 100, grid: { color: '#1e293b' }, ticks: { color: '#94a3b8' } }
            },
            responsive: true, maintainAspectRatio: false
        }
    });
}

// Global scope logic router
window.onload = async function () {
    const pageType = document.body.getAttribute('data-page');
    
    if (pageType === "static") {
        initCharts();
        await loadOverview();
        await loadTrainingCurve();
        await loadRecentScans();

    } else if (pageType === "live") {
        document.querySelectorAll('.pipeline-tracker .step').forEach(el => {
            el.addEventListener('click', () => showStepDetail(el.dataset.step));
            // Keyboard users get tabindex="0" in the HTML; a bare tabindex
            // doesn't make Enter/Space activate a <div> the way it would a
            // real <button>, so wire that up explicitly.
            el.addEventListener('keydown', (e) => {
                if (e.key === 'Enter' || e.key === ' ') {
                    e.preventDefault();
                    showStepDetail(el.dataset.step);
                }
            });
        });

        const uploader = document.getElementById('fileUploader');
        if (uploader) {
            uploader.addEventListener('change', function (e) {
                const file = e.target.files[0];
                if (!file) return;
                const reader = new FileReader();
                reader.onload = function (evt) {
                    document.getElementById('payloadInput').value = evt.target.result;
                    document.getElementById('jsonDump').innerText = "[*] File '" + file.name + "' successfully loaded into memory.\n[*] Ready for Analysis.";
                };
                reader.readAsText(file);
            });
        }
        
        // Restore session state if available -- shown as a small dismissible
        // banner above the (still visible) upload form, not a separate
        // screen state that hides the form until explicitly cleared.
        const sessionStore = sessionStorage.getItem('scanData');
        if (sessionStore) {
            try {
                const data = JSON.parse(sessionStore);
                document.getElementById('jsonDump').innerText = "Restored previous analysis from session. Ready for new upload if needed.";

                // Make arena visible
                const arena = document.getElementById('live-arena');
                if (arena) {
                    arena.style.opacity = "1";
                    arena.style.pointerEvents = "auto";
                }

                // Re-populate dashboard
                updateDashboard(data.formattedData, data.gnnConf);
                restoreNavigationButtons(data.formattedData);

                const banner = document.getElementById('restored-banner');
                if (banner) banner.style.display = 'block';

                restoreSidebar();

            } catch (e) {
                console.error("Failed to restore session state", e);
            }
        }

    } else if (pageType === "metrics") {
        loadMetricsReport(); // Load the detailed comparison from API
        restoreSidebar();
    }
};

async function loadOverview() {
    const summaryEl = document.getElementById('overview_summary');
    const emptyEl = document.getElementById('overview_empty_state');
    try {
        const res = await fetch('/api/report');
        const data = await res.json();

        if (!data.success) {
            if (summaryEl) summaryEl.innerText = data.detail || 'No dataset evaluation available yet.';
            if (emptyEl) emptyEl.style.display = 'block';
            return;
        }

        const mc = data.gnn_model_multiclass;
        const base = data.baseline_model;
        const gnn = data.gnn_model;

        const totalSamples = mc ? mc.dataset_size : data.dataset_size;
        const accuracyPct = mc ? mc.accuracy : gnn.accuracy;
        const correct = Math.round(totalSamples * (accuracyPct / 100));
        const missed = totalSamples - correct;

        if (summaryEl) {
            summaryEl.innerText = `Evaluated on ${totalSamples} held-out samples the model never trained on. ` +
                `${correct} correctly classified, ${missed} missed (${accuracyPct.toFixed(2)}% accuracy).`;
        }

        const setText = (id, text) => { const el = document.getElementById(id); if (el) el.innerText = text; };
        setText('ov_total', totalSamples);
        setText('ov_correct', correct);
        setText('ov_missed', missed);
        setText('ov_completeness', mc ? mc.macro_f1.toFixed(2) + '%' : 'N/A (retrain to compute)');

        setText('ov_base_name', base.name);
        setText('ov_base_acc', base.accuracy.toFixed(2) + '%');
        setText('ov_base_pre', base.precision.toFixed(2) + '%');
        setText('ov_base_rec', base.recall.toFixed(2) + '%');
        setText('ov_base_note', base.is_measured ? '' : (base.note || 'Reference value, not measured on this dataset.'));

        setText('ov_gnn_name', gnn.name);
        setText('ov_gnn_acc', gnn.accuracy.toFixed(2) + '%');
        setText('ov_gnn_pre', gnn.precision.toFixed(2) + '%');
        setText('ov_gnn_rec', gnn.recall.toFixed(2) + '%');

        if (gaugeChart) {
            gaugeChart.data.datasets[0].data = [accuracyPct, 100 - accuracyPct];
            gaugeChart.data.datasets[0].backgroundColor = ['#4ade80', 'transparent'];
            gaugeChart.update();
        }

        if (mc) {
            renderPerFamilyChart(mc.per_class);
        }
    } catch (e) {
        console.error('Failed to load overview report', e);
        if (summaryEl) summaryEl.innerText = 'Failed to load dataset evaluation -- is the server running?';
        if (emptyEl) emptyEl.style.display = 'block';
    }
}

async function loadRecentScans() {
    const container = document.getElementById('recentScansTable');
    if (!container) return;
    try {
        const res = await fetch('/api/scans/recent?limit=10');
        const data = await res.json();
        const scans = data.scans || [];
        if (!scans.length) {
            container.innerHTML = '<p style="color:#64748b; font-size:0.9rem;">No scans yet -- run one from the Live Analysis page.</p>';
            return;
        }
        const escapeHtml = (str) => String(str).replace(/[&<>"']/g, (c) => (
            { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]
        ));
        const rows = scans.map(s => {
            const label = s.fused_diagnosis || '-';
            const color = label === 'Malicious' ? 'var(--neon-red)' : (label === 'Benign' ? 'var(--neon-green)' : '#94a3b8');
            const when = s.timestamp ? new Date(s.timestamp).toLocaleString() : '-';
            const conf = s.fused_confidence != null ? (s.fused_confidence * 100).toFixed(1) + '%' : '-';
            const rawSource = s.source_label || s.input_type || '-';
            const source = escapeHtml(rawSource);
            return `<tr>
                <td>${when}</td>
                <td><span class="source-cell" title="${source}">${source}</span></td>
                <td style="color:${color}; font-weight:bold;">${label}</td>
                <td>${conf}</td>
                <td>${s.agent2_triggered ? '⚠️ Yes' : '-'}</td>
            </tr>`;
        }).join('');
        container.innerHTML = `<table class="metrics-table">
            <thead><tr style="color:#94a3b8; text-align:left; font-size:0.8rem; text-transform:uppercase;">
                <th style="padding:8px 0;">When</th><th>Source</th><th>Verdict</th><th>Confidence</th><th>Agent 2</th>
            </tr></thead>
            <tbody>${rows}</tbody>
        </table>`;
    } catch (e) {
        container.innerHTML = '<p style="color:#64748b; font-size:0.9rem;">Scan history unavailable.</p>';
    }
}

async function loadTrainingCurve() {
    try {
        const res = await fetch('/api/training_history');
        if (!res.ok) return;
        const data = await res.json();
        renderTrainingCurveChart(data.history);
    } catch (e) {
        console.log('No training history available yet.');
    }
}

function showStepDetail(stepKey) {
    const panel = document.getElementById('stepDetailPanel');
    const titleEl = document.getElementById('stepDetailTitle');
    const contentEl = document.getElementById('stepDetailContent');
    if (!panel || !contentEl) return;

    document.querySelectorAll('.pipeline-tracker .step').forEach(el => el.classList.remove('selected'));
    const clicked = document.querySelector(`.pipeline-tracker .step[data-step="${stepKey}"]`);
    if (clicked) clicked.classList.add('selected');

    const sessionStore = sessionStorage.getItem('scanData');
    if (!sessionStore) {
        panel.style.display = 'block';
        titleEl.innerText = 'No analysis yet';
        contentEl.innerHTML = '<em>Submit an API request on the left first -- this panel shows real data from that specific scan, not a canned example.</em>';
        return;
    }

    const data = JSON.parse(sessionStore).formattedData;
    panel.style.display = 'block';

    if (stepKey === 'input') {
        titleEl.innerText = '1. What you submitted';
        const meta = data.input_meta || {};
        contentEl.innerHTML = `
            <div class="stat-row"><span>Input type</span><strong>${meta.input_type || '-'}</strong></div>
            <div class="stat-row"><span>Source</span><strong>${meta.filename || 'pasted text'}</strong></div>
            <div class="stat-row"><span>Size</span><strong>${meta.content_length || 0} bytes/chars</strong></div>
            <div style="margin-top:10px;"><span style="color:#94a3b8;">Preview:</span><pre style="margin-top:6px; max-height:120px;">${(meta.content_preview || '').replace(/</g, '&lt;')}</pre></div>
        `;

    } else if (stepKey === 'adapter') {
        titleEl.innerText = '2. Feature extraction (real signal found in your input)';
        const sig = data.path_gnn.signal_summary || {};
        contentEl.innerHTML = `
            <p style="color:#94a3b8; margin-top:0;">UniversalInputAdapter scans the raw input for API-like tokens, network indicators, PE header markers, and computes Shannon entropy -- this is what it actually found:</p>
            <div class="stat-row"><span>Tokens scanned</span><strong>${sig.token_count ?? '-'}</strong></div>
            <div class="stat-row"><span>API-like tokens</span><strong>${sig.api_count ?? 0}</strong></div>
            <div class="stat-row"><span>Network indicators</span><strong>${sig.network_count ?? 0}</strong></div>
            <div class="stat-row"><span>Header markers</span><strong>${sig.header_count ?? 0}</strong></div>
            <div class="stat-row"><span>Entropy</span><strong>${sig.entropy ?? '-'}</strong></div>
            <div class="stat-row"><span>Suspicious terms matched</span><strong>${sig.suspicious_count ?? 0}</strong></div>
            ${sig.suspicious_terms && sig.suspicious_terms.length ? `<div style="margin-top:8px; color:#f87171;">${sig.suspicious_terms.join(', ')}</div>` : ''}
        `;

    } else if (stepKey === 'graph') {
        titleEl.innerText = '3. The 4-node graph built from your input';
        const completeness = data.completeness_raw || 0;
        const filled = data.nodes_filled_detail || { header: false, entropy: false, api: false, network: false };
        contentEl.innerHTML = `
            <p style="color:#94a3b8; margin-top:0;">Every request becomes this same fixed 4-node structure (matching <code>core/graph_constructor.py</code>) -- only which nodes carry real signal vs. zero-padding changes per input:</p>
            ${buildGraphDiagramSVG(filled)}
            <div class="stat-row" style="margin-top:10px;"><span>Feature completeness</span><strong>${Math.round(completeness * 100)}% (${Object.values(filled).filter(Boolean).length}/4 nodes with real signal)</strong></div>
        `;

    } else if (stepKey === 'gnn') {
        titleEl.innerText = '4. GNN attention + PAC-X findings';
        const attn = data.path_gnn.attention_weights || [0, 0, 0, 0];
        const labels = ['Header', 'Entropy', 'API', 'Network'];
        const attnObj = { header: attn[0] || 0, entropy: attn[1] || 0, api: attn[2] || 0, network: attn[3] || 0 };
        const attnRows = labels.map((l, i) => `
            <div class="stat-row"><span>${l} attention</span><strong>${((attn[i] || 0) * 100).toFixed(0)}%</strong></div>
        `).join('');
        const findings = (data.path_pacx.findings && data.path_pacx.findings.length)
            ? data.path_pacx.findings.map(f => `<li>${f}</li>`).join('')
            : '<li>No heuristic findings triggered</li>';
        contentEl.innerHTML = `
            <p style="color:#94a3b8; margin-top:0;">GNN attention overlaid on the real graph -- larger/brighter node = more weight in the final decision for this sample:</p>
            ${buildGraphDiagramSVG(null, attnObj)}
            ${attnRows}
            <p style="color:#94a3b8; margin-top:14px;">PAC-X: what the heuristic path actually flagged.</p>
            <ul style="margin:6px 0 0 18px; padding:0;">${findings}</ul>
        `;

    } else if (stepKey === 'decision') {
        titleEl.innerText = '5. How the final verdict was reached';
        const fusion = data.decision_fusion || {};
        contentEl.innerHTML = `
            <div class="stat-row"><span>PAC-X said</span><strong>${data.path_pacx.prediction}</strong></div>
            <div class="stat-row"><span>GNN said</span><strong>${data.path_gnn.prediction}</strong></div>
            <div class="stat-row"><span>Agreement</span><strong>${fusion.agreement ? 'Yes -- scores blended' : 'No -- arbitrated by trust score'}</strong></div>
            <div class="stat-row"><span>Final verdict</span><strong>${fusion.final_label || '-'}</strong></div>
            <div class="stat-row"><span>Final confidence</span><strong>${((fusion.confidence || 0) * 100).toFixed(1)}%</strong></div>
        `;
    }
}

function buildGraphDiagramSVG(filled, attention) {
    // Real topology from core/graph_constructor.py -- NOT fully connected:
    // Header<->Entropy, Header<->API, API<->Network, Entropy<->API.
    // `attention` (optional) is {header, entropy, api, network} in [0,1] from
    // the real GNN forward pass -- when present, node size/glow and edge
    // thickness scale with it instead of just showing fill status.
    const pos = { header: [90, 30], entropy: [180, 90], api: [90, 150], network: [0, 90] };
    const edges = [['header', 'entropy'], ['header', 'api'], ['api', 'network'], ['entropy', 'api']];
    const labelFor = { header: 'Header', entropy: 'Entropy', api: 'API', network: 'Network' };
    const attn = attention || null;

    const edgeLines = edges.map(([a, b]) => {
        const w = attn ? 2 + 6 * ((attn[a] || 0) + (attn[b] || 0)) / 2 : 2;
        const opacity = attn ? 0.35 + 0.65 * ((attn[a] || 0) + (attn[b] || 0)) / 2 : 1;
        return `<line x1="${pos[a][0]}" y1="${pos[a][1]}" x2="${pos[b][0]}" y2="${pos[b][1]}" stroke="${attn ? '#22d3ee' : '#475569'}" stroke-width="${w.toFixed(1)}" opacity="${opacity.toFixed(2)}"/>`;
    }).join('');

    const nodeCircles = Object.keys(pos).map(k => {
        const a = attn ? (attn[k] || 0) : null;
        const radius = attn ? (16 + a * 18) : 22;
        const color = attn ? '#c084fc' : (filled[k] ? '#4ade80' : '#334155');
        const opacity = attn ? (0.4 + a * 0.6) : (filled[k] ? 0.85 : 0.35);
        const glow = attn && a > 0.3 ? `filter="drop-shadow(0 0 ${(a * 8).toFixed(0)}px ${color})"` : '';
        const sublabel = attn ? `<text x="${pos[k][0]}" y="${pos[k][1] + 16}" text-anchor="middle" font-size="8" fill="#cbd5e1">${(a * 100).toFixed(0)}%</text>` : '';
        return `
        <circle cx="${pos[k][0]}" cy="${pos[k][1]}" r="${radius.toFixed(1)}" fill="${color}" opacity="${opacity.toFixed(2)}" stroke="#0f172a" stroke-width="2" ${glow}/>
        <text x="${pos[k][0]}" y="${pos[k][1] + 4}" text-anchor="middle" font-size="10" fill="#0f172a" font-weight="bold">${labelFor[k]}</text>
        ${sublabel}
        `;
    }).join('');

    return `<svg viewBox="-20 -10 220 210" style="width:100%; max-width:280px; height:auto; display:block; margin:12px auto;">${edgeLines}${nodeCircles}</svg>`;
}

function clearSession() {
    sessionStorage.removeItem('scanData');
    location.reload();
}

function restoreNavigationButtons(formattedData) {
    const btn1 = document.getElementById('nav_btn_1');
    const btn1Text = document.getElementById('btn1_text');

    if (btn1) btn1.style.display = 'block';

    if (btn1Text && formattedData?.agent1?.better_model) {
        const bestModel = formattedData.agent1.better_model === 'GNN' ? 'GNN (Agentic-PACX)' : 'PAC-X';
        btn1Text.innerText = 'Know more why ' + bestModel + ' is best';
    }
}

function restoreSidebar() {
    const sessionStore = sessionStorage.getItem('scanData');
    if (sessionStore) {
        try {
            const data = JSON.parse(sessionStore);
            const agent1Decision = data.formattedData.agent1;
            
            // Set pipeline to completed state
            const steps = ['step-input', 'step-adapter', 'step-graph', 'step-gnn', 'step-decision'];
            for (let i = 0; i < 4; i++) {
                const stepEl = document.getElementById(steps[i]);
                if (stepEl) stepEl.className = "step active";
            }
            
            const stepDecision = document.getElementById('step-decision');
            if (stepDecision) {
                if (data.formattedData.agent2.triggered) {
                    stepDecision.className = "step escalated";
                    stepDecision.innerHTML = "5. Zero-Day Detected → Agent 2 Active";
                } else {
                    stepDecision.className = "step trusted";
                    stepDecision.innerHTML = "5. Verified - " + agent1Decision.better_model + " Trusted";
                }
            }
            
            const dump = document.getElementById('jsonDump');
            if (dump) {
                dump.innerHTML = "<b>Recommendation:</b> " + data.formattedData.agent1.recommendation + 
                "<br><br><b>Agent 2 Status:</b> " + (data.formattedData.agent2.triggered ? "⚠️ ACTIVATED" : "✓ Standby");
            }
            
        } catch (e) {
            console.error("Failed to restore sidebar", e);
        }
    }
}

// ============================================
// LIVE ANALYSIS LOGIC (runs only on dynamic page)
// ============================================
// Remembers the last batch response so a drill-down can return to it
// without re-running all N rows through the pipeline.
let lastBatchResult = null;

async function runAnalysis(rowIndex = null) {
    // Get input from UI
    const inputPayload = document.getElementById('payloadInput').value;
    const fileInput = document.getElementById('fileUploader');

    // Validate input
    if (!inputPayload && (!fileInput.files || !fileInput.files[0])) {
        document.getElementById('jsonDump').innerText = "[-] Error: Please upload a file or enter text.";
        return;
    }

    // Reset UI for new analysis
    document.getElementById('m_total').innerText = "Analyzing...";
    document.getElementById('agent1_reasoning').innerText = "Processing Dual-Pathway analysis...";

    // Show animation
    await executePipelineAnimation();

    // Make arena visible
    const arena = document.getElementById('live-arena');
    if (arena) {
        arena.style.opacity = "1";
        arena.style.pointerEvents = "auto";
    }

    // Evaluate Data Upload via Real GNN API
    const formData = new FormData();
    const inputType = document.getElementById('inputType').value;
    formData.append('input_type', inputType);

    if (fileInput.files && fileInput.files[0]) {
        formData.append('file', fileInput.files[0]);
    } else {
        formData.append('text', inputPayload);
    }

    // Drilling into a single row of a multi-row upload -- the backend returns
    // the full single-sample shape for just that row.
    if (rowIndex !== null) {
        formData.append('row_index', rowIndex);
    }

    try {
        const response = await fetch('/api/analyze/live', {
            method: 'POST',
            body: formData
        });
        const result = await response.json();

        if (!result.success) throw new Error(result.detail || "Inference failed");

        console.log("Inference Result:", result);

        if (result.batch) {
            lastBatchResult = result;
            renderBatchResults(result);
            return;
        }

        // A normal single-sample result -- make sure a previous batch run
        // isn't still hiding these panels / showing its own table.
        const singlePanels = document.getElementById('singleResultPanels');
        if (singlePanels) singlePanels.style.display = 'contents';
        const batchPanel = document.getElementById('batchResultsPanel');
        if (batchPanel) batchPanel.style.display = 'none';

        // Format backend dual-path response
        const neuralConf = result.path_gnn.confidence;
        const pacxConf = result.path_pacx.confidence;
        const agent1Decision = result.agent1_decision;
        const fusion = result.decision_fusion || {
            final_label: (agent1Decision.better_model === "GNN" ? result.path_gnn.diagnosis : result.path_pacx.diagnosis),
            final_score: (agent1Decision.better_model === "GNN" ? result.path_gnn.confidence : result.path_pacx.confidence),
            weights: { pacx: 0.4, gnn: 0.6 }
        };

        const formattedData = {
            metrics: {
                total: 1,
                trusted: fusion.final_label === "Benign" ? 1 : 0,
                escalated: fusion.final_label !== "Benign" ? 1 : 0
            },
            decision_fusion: fusion,
            path_pacx: {
                prediction: result.path_pacx.diagnosis,
                confidence: pacxConf,
                accuracy: result.path_pacx.accuracy,
                precision: result.path_pacx.precision,
                recall: result.path_pacx.recall,
                findings: result.path_pacx.findings,
                evidence_score: result.path_pacx.evidence_score,
                breakdown: result.path_pacx.breakdown,
                method: "Prospect-Aspect-Context"
            },
            path_gnn: {
                prediction: result.path_gnn.diagnosis,
                confidence: neuralConf,
                accuracy: result.path_gnn.accuracy,
                precision: result.path_gnn.precision,
                recall: result.path_gnn.recall,
                completeness: result.path_gnn.completeness,
                attention_weights: result.path_gnn.attention_weights,
                predicted_family: result.path_gnn.predicted_family,
                family_confidence: result.path_gnn.family_confidence,
                raw_confidence: result.path_gnn.raw_confidence,
                binary_scores: result.path_gnn.binary_scores,
                model_scores: result.path_gnn.model_scores,
                centroid_scores: result.path_gnn.centroid_scores,
                evidence_score: result.path_gnn.evidence_score,
                signal_summary: result.path_gnn.signal_summary,
                top_classes: result.path_gnn.top_classes,
                centroid_top_classes: result.path_gnn.centroid_top_classes,
                method: "Graph Attention Network"
            },
            agent1: {
                better_model: agent1Decision.better_model,
                reasoning: agent1Decision.reasoning,
                reasoning_source: agent1Decision.reasoning_source,
                agreement: agent1Decision.prediction_agreement,
                trust_scores: agent1Decision.trust_scores,
                recommendation: result.agent1_recommendation
            },
            agent2: result.agent2,
            gnn_report: result.gnn_report,
            completeness_raw: result.completeness,
            nodes_filled_detail: result.nodes_filled_detail,
            input_meta: {
                input_type: inputType,
                filename: (fileInput.files && fileInput.files[0]) ? fileInput.files[0].name : null,
                content_preview: (inputPayload || '').slice(0, 200),
                content_length: (fileInput.files && fileInput.files[0]) ? fileInput.files[0].size : (inputPayload || '').length,
            },
        };

        updateDashboard(formattedData, neuralConf);
        restoreNavigationButtons(formattedData);

        restoreSidebar();
        
        // Save to session storage for multi-page routing
        sessionStorage.setItem('scanData', JSON.stringify({formattedData, gnnConf: neuralConf}));
        
    } catch (error) {
        console.error("Analysis Error:", error);
        document.getElementById('m_total').innerText = "0";
        document.getElementById('jsonDump').innerText = "[-] Error: " + error.message;
    }
}

/**
 * Colour for a fused verdict. "Suspicious" is its own amber state: the sample
 * was NOT cleared, but nothing was positively identified either -- showing it
 * green would hide a likely zero-day, showing it red would claim a detection
 * the system did not actually make.
 */
function verdictColor(label) {
    if (label === 'Malicious') return 'var(--neon-red)';
    if (label === 'Suspicious') return 'var(--verdict-amber)';
    return 'var(--neon-green)';
}

function renderBatchResults(result) {
    const s = result.summary;
    const escapeHtml = (str) => String(str).replace(/[&<>"']/g, (c) => (
        { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]
    ));

    // Metric ribbon reflects the whole batch, not a single sample.
    document.getElementById('m_total').innerText = s.n_total;
    document.getElementById('m_trusted').innerText = s.benign_count;
    document.getElementById('m_escalated').innerText = s.malicious_count;
    document.getElementById('m_comp').innerText = (s.avg_confidence * 100).toFixed(0) + '%';

    // Single-sample panels (Decision Fusion verdict, PAC-X/GNN showdown,
    // Agent 1 reasoning, structural graph...) don't make sense for N
    // results at once -- swap them for the batch table instead.
    const singlePanels = document.getElementById('singleResultPanels');
    if (singlePanels) singlePanels.style.display = 'none';
    const stepDetail = document.getElementById('stepDetailPanel');
    if (stepDetail) stepDetail.style.display = 'none';

    const batchPanel = document.getElementById('batchResultsPanel');
    batchPanel.style.display = 'block';

    let summaryHtml = `Analyzed all <strong>${s.n_total}</strong> rows -- `
        + `<span style="color:var(--neon-green);">${s.benign_count} Benign</span>, `
        + `<span style="color:var(--verdict-amber);">${s.suspicious_count || 0} Suspicious</span>, `
        + `<span style="color:var(--neon-red);">${s.malicious_count} Malicious</span>, `
        + `<strong>${s.agent2_triggered_count}</strong> flagged low-confidence (added to Agent 2's retrain pool -- `
        + `retraining still requires an explicit click, not automatic).`;
    if (s.n_failed > 0) {
        summaryHtml += ` <span style="color:var(--neon-red);">${s.n_failed} row(s) failed to parse.</span>`;
    }
    if (s.truncated) {
        summaryHtml += ` <em>File had more rows than the ${s.n_total} processed -- truncated for this run.</em>`;
    }
    document.getElementById('batchSummaryLine').innerHTML = summaryHtml;

    const rows = result.results.map(r => {
        if (!r.success) {
            return `<tr><td>${r.row_index + 1}</td><td colspan="5" style="color:var(--neon-red);">${escapeHtml(r.error || 'Failed')}</td></tr>`;
        }
        const color = verdictColor(r.fused_label);
        return `<tr class="batch-row" data-row-index="${r.row_index}" tabindex="0"
                    role="button" title="Click to see the full PAC-X / GNN / Fusion breakdown for this row">
            <td>${r.row_index + 1}</td>
            <td>${escapeHtml(r.pacx_label)}</td>
            <td>${escapeHtml(r.gnn_label)}${r.gnn_family && r.gnn_family !== 'benign' ? ' (' + escapeHtml(r.gnn_family) + ')' : ''}</td>
            <td style="color:${color}; font-weight:bold;">${escapeHtml(r.fused_label)}</td>
            <td>${(r.fused_confidence * 100).toFixed(1)}%</td>
            <td>${r.agent2_triggered ? '⚠️ Yes' : '-'}</td>
        </tr>`;
    }).join('');
    const tbody = document.getElementById('batchResultsBody');
    tbody.innerHTML = rows;

    // Click (or keyboard-activate) a row to drill into that single sample.
    tbody.querySelectorAll('tr.batch-row').forEach(tr => {
        const open = () => drillIntoRow(Number(tr.dataset.rowIndex));
        tr.addEventListener('click', open);
        tr.addEventListener('keydown', (e) => {
            if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); open(); }
        });
    });

    document.getElementById('jsonDump').innerText =
        `Batch analysis complete: ${s.n_total} rows, ${s.malicious_count} malicious, ${s.agent2_triggered_count} triggered Agent 2.\n`
        + `Click any row for its full PAC-X / GNN / Fusion breakdown.`;

    sessionStorage.removeItem('scanData');
}

/**
 * Re-analyzes ONE row of the current multi-row upload and shows the normal
 * single-sample panels for it. The batch view deliberately hides those panels
 * (they can't represent N samples at once), so this is how a single row's
 * detail -- including Agent 1's LLM reasoning, which batch mode skips -- is
 * recovered. Re-sends the same file with a row_index rather than caching N
 * full analyses client-side.
 */
async function drillIntoRow(rowIndex) {
    showRetrainStatus(`Loading full breakdown for row ${rowIndex + 1}...`);
    await runAnalysis(rowIndex);

    const backBtn = document.getElementById('backToBatchBtn');
    if (backBtn) backBtn.style.display = 'inline-block';

    const banner = document.getElementById('drillDownBanner');
    if (banner) {
        banner.style.display = 'block';
        banner.innerText = `Showing row ${rowIndex + 1} of the uploaded batch.`;
    }
}

/** Returns to the batch table without re-running every row. */
function backToBatchResults() {
    if (!lastBatchResult) return;
    renderBatchResults(lastBatchResult);
    const backBtn = document.getElementById('backToBatchBtn');
    if (backBtn) backBtn.style.display = 'none';
    const banner = document.getElementById('drillDownBanner');
    if (banner) banner.style.display = 'none';
}

function updateDashboard(data, gnnConf) {
    const clipped = (v) => Math.max(0, Math.min(1, Number(v || 0)));
    const displayGnnConf = clipped(data?.path_gnn?.confidence);
    const fusionConf = clipped(data?.decision_fusion?.confidence);
    const fusionScore = clipped(data?.decision_fusion?.final_score);
    const agent2Active = Boolean(data?.agent2?.triggered);

    // 1. Update Top Ribbon Metrics (with null checks)
    const m_total = document.getElementById('m_total');
    if (m_total) {
        m_total.innerText = "1";
        document.getElementById('m_trusted').innerText = data.metrics.trusted;
        document.getElementById('m_escalated').innerText = data.metrics.escalated;
        document.getElementById('m_comp').innerText = (data.completeness_raw * 100).toFixed(0) + "%";

        // 2. Update Side-by-Side Diagnosis
        document.getElementById('pacx_pred').innerText = data.path_pacx.prediction;
        document.getElementById('pacx_conf').innerText = (data.path_pacx.confidence * 100).toFixed(1) + "% Confidence";

        document.getElementById('gnn_pred').innerText = data.path_gnn.prediction;
        document.getElementById('gnn_conf').innerText = (displayGnnConf * 100).toFixed(1) + "% Confidence";
        const agent1Status = document.getElementById('pacx_status');
        if (agent1Status) {
            agent1Status.innerText = data.path_pacx.prediction === "Malicious" ? "Heuristic Alert" : "Heuristic Clear";
            agent1Status.className = data.path_pacx.prediction === "Malicious" ? "status-badge badge-escalated" : "status-badge badge-trusted";
        }

        const agent1Reasoning = document.getElementById('agent1_reasoning');
        if (agent1Reasoning) {
            const familyLine = data.path_gnn.predicted_family
                ? `<div style="margin-top:8px;color:#94a3b8;">Top GNN family: <b>${data.path_gnn.predicted_family}</b> (${(data.path_gnn.family_confidence * 100).toFixed(1)}%)</div>`
                : "";
            agent1Reasoning.innerHTML = `${reasoningSourceBadge(data.agent1.reasoning_source)}${data.agent1.reasoning}${familyLine}`;
        }

        // GNN structural forensic report (attention-weight based XAI narrative) --
        // computed by the backend on every scan but previously never rendered anywhere.
        const gnnForensicCard = document.getElementById('gnn_forensic_card');
        const gnnForensicReport = document.getElementById('gnn_forensic_report');
        if (gnnForensicCard && gnnForensicReport && data.gnn_report) {
            gnnForensicReport.innerHTML = data.gnn_report;
            gnnForensicCard.style.display = "block";
        }

        const fusionLabelEl = document.getElementById('fusion_label');
        const fusionConfEl = document.getElementById('fusion_confidence');
        const fusionArbitrationEl = document.getElementById('fusion_arbitration');
        const fusionWPacxEl = document.getElementById('fusion_w_pacx');
        const fusionWGnnEl = document.getElementById('fusion_w_gnn');
        const fusionInputsEl = document.getElementById('fusion_inputs');
        const fusionData = data.decision_fusion || {};
        const fusionWeights = fusionData.weights || {};
        const fusionInputs = fusionData.inputs || {};
        if (fusionLabelEl) {
            fusionLabelEl.innerText = fusionData.final_label || "-";
            fusionLabelEl.style.color = verdictColor(fusionData.final_label);
        }
        if (fusionConfEl) fusionConfEl.innerText = (fusionConf * 100).toFixed(1) + "%";

        // Explain an escalated verdict in plain language, right under it --
        // "Suspicious" is meaningless to a reader without the reason.
        const escalationEl = document.getElementById('fusion_escalation');
        if (escalationEl) {
            if (fusionData.escalated) {
                escalationEl.innerText = fusionData.escalation_reason
                    || "Not cleared: confidence too low to call this benign.";
                escalationEl.style.display = 'block';
            } else {
                escalationEl.style.display = 'none';
            }
        }
        if (fusionArbitrationEl) fusionArbitrationEl.innerText = describeArbitration(fusionData);
        if (fusionWPacxEl) fusionWPacxEl.innerText = ((fusionWeights.pacx || 0) * 100).toFixed(1) + "%";
        if (fusionWGnnEl) fusionWGnnEl.innerText = ((fusionWeights.gnn || 0) * 100).toFixed(1) + "%";
        if (fusionInputsEl) {
            fusionInputsEl.innerText =
                `PAC-X score ${(clipped(fusionInputs.pacx_malicious_score) * 100).toFixed(1)}% | ` +
                `GNN score ${(clipped(fusionInputs.gnn_malicious_score) * 100).toFixed(1)}% | ` +
                `Completeness ${(clipped(fusionInputs.gnn_completeness) * 100).toFixed(0)}% | ` +
                `Certainty ${(clipped(fusionInputs.gnn_model_certainty) * 100).toFixed(1)}% | ` +
                `Final score ${(fusionScore * 100).toFixed(1)}%`;
        }

        const ag2Status = document.getElementById('gnn_status');
        const board = document.getElementById('gnn_board');
        if (agent2Active) {
            ag2Status.innerText = "⚠️ ZERO-DAY SUSPECTED";
            ag2Status.className = "status-badge badge-escalated";
            board.classList.add("alert");
        } else {
            ag2Status.innerText = "✓ VERIFIED ANALYSIS";
            ag2Status.className = "status-badge badge-trusted";
            board.classList.remove("alert");
        }
    }

    // 3. Update Visualizations (Only if charts are initialized on this page)
    if (document.getElementById('sunburstChart')) {
        // Plotly Sunburst
        const labels = ['Analysis', 'GNN Prediction', 'PAC-X Prediction'];
        const parents = ['', 'Analysis', 'Analysis'];
        const values = [100, displayGnnConf * 100, clipped(data.path_pacx.confidence) * 100];
        
        // Add GNN Features (Radar Data)
        const gnnFeatures = ['File Headings', 'File Complexity', 'Programs Opened', 'Network & IP'];
        const attention = data.path_gnn.attention_weights || [0.25, 0.25, 0.25, 0.25];
        for (let i = 0; i < gnnFeatures.length; i++) {
            labels.push(gnnFeatures[i]);
            parents.push('GNN Prediction');
            values.push((attention[i] * displayGnnConf) * 100);
        }
        
        // Add PAC-X Findings
        const pacxBreakdown = data.path_pacx.breakdown || {};
        const pacxEntries = Object.entries(pacxBreakdown).filter(([, value]) => value > 0);
        if (pacxEntries.length > 0) {
            for (const [key, value] of pacxEntries) {
                labels.push(`PAC-X ${key}`);
                parents.push('PAC-X Prediction');
                values.push(value * data.path_pacx.confidence * 100);
            }
        } else {
            const pacxFindings = data.path_pacx.findings || ["Header", "Entropy", "API", "Network"];
            for (let i = 0; i < pacxFindings.length; i++) {
                labels.push(pacxFindings[i]);
                parents.push('PAC-X Prediction');
                values.push((data.path_pacx.confidence * 100) / pacxFindings.length);
            }
        }

        const sunburstData = [{
            type: "sunburst",
            labels: labels,
            parents: parents,
            values: values,
            outsidetextfont: {size: 14, color: "#cbd5e1"},
            leaf: {opacity: 0.8},
            marker: {line: {width: 2, color: '#0f172a'}, colors: ['#0f172a', '#4ade80', '#22d3ee', '#1e293b', '#1e293b', '#1e293b', '#1e293b', '#c084fc', '#c084fc', '#c084fc', '#c084fc']}
        }];

        const layout = {
            margin: {l: 0, r: 0, b: 0, t: 0},
            paper_bgcolor: 'transparent',
            sunburstcolorway: ["#4ade80","#22d3ee"],
            font: {family: 'Inter, Segoe UI', color: '#cbd5e1'}
        };

        Plotly.newPlot('sunburstChart', sunburstData, layout, {responsive: true});
        
        const loader = document.getElementById('visualLoader');
        const content = document.getElementById('visualContent');
        if(loader) loader.style.display = 'none';
        if(content) content.style.display = 'flex';
    }
}

async function executePipelineAnimation() {
    const steps = ['step-input', 'step-adapter', 'step-graph', 'step-gnn', 'step-decision'];
    for (let step of steps) { document.getElementById(step).className = "step"; } // reset
    for (let i = 0; i < 4; i++) {
        document.getElementById(steps[i]).className = "step active";
        await sleep(400);
    }
}

function showRetrainStatus(message) {
    const panel = document.getElementById('jsonDump');
    if (panel) panel.innerText = message;
}

async function triggerRetraining() {
    const btn = document.getElementById('retrainBtn') || document.getElementById('retrainBtnLive');
    const originalText = btn.innerHTML;

    // Show what's actually IN the pool before retraining on it. Retraining is
    // never automatic -- a human reviewing this breakdown is the whole point.
    let pool = null;
    try {
        const poolResp = await fetch('/api/retrain/pool');
        pool = await poolResp.json();
    } catch (e) {
        // Fall through to the server-side check, which is authoritative.
    }

    if (pool && pool.blocked) {
        // Report to the STATUS panel, not just alert(). Chrome suppresses
        // repeated alert() dialogs ("prevent this page from creating more
        // dialogs"), which made a refusal look like the button was dead --
        // the click was correctly blocked but said so invisibly.
        showRetrainStatus("[-] RETRAINING BLOCKED\n\n" + pool.reason
              + "\n\nRemedy: clear data/retrain_pool.pt and rebuild it from "
              + "samples with verified family labels before retraining.");
        alert("🚫 RETRAINING BLOCKED\n\n" + pool.reason
              + "\n\nClear data/retrain_pool.pt and rebuild it from samples with "
              + "verified family labels before retraining.");
        return;
    }

    let poolSummary = "";
    if (pool && pool.size) {
        const breakdown = Object.entries(pool.counts)
            .map(([name, n]) => `  ${n} ${name}`).join("\n");
        poolSummary = `\n\nPool contents (${pool.size} samples):\n${breakdown}\n`;
    }

    if (!confirm(
        "This retrains Agent 2 on the current low-confidence sample pool and replaces "
        + "the live model when it finishes."
        + poolSummary
        + "\nThese labels are the model's own guesses on samples it was uncertain about, "
        + "not verified ground truth. Retraining on them can REDUCE real accuracy. Continue?"
    )) {
        return;
    }

    try {
        btn.innerHTML = "🧬 Evolving Model...";
        btn.disabled = true;
        btn.style.opacity = "0.5";

        const response = await fetch('/api/retrain', { method: 'POST' });
        const result = await response.json();

        if (response.status === 409) {
            const msg = "[-] RETRAINING BLOCKED\n\n" + (result.message || "Pool is unfit to train on.")
                  + (result.remedy ? "\n\nRemedy: " + result.remedy : "");
            showRetrainStatus(msg);
            alert(msg);
        } else if (result.success && result.queued) {
            // Retraining now runs in the background and reloads the model
            // when it finishes -- this response only confirms it *started*,
            // it hasn't evolved yet. Reloading the page immediately would
            // just show the pre-retrain model.
            const msg = "[*] AGENT 2 QUEUED: retraining started in the background.\n"
                  + "This takes a minute or two -- refresh the page afterwards to see the updated model.";
            showRetrainStatus(msg);
            alert(msg);
        } else if (result.success) {
            showRetrainStatus("[+] AGENT 2: retraining complete.");
            alert("✅ AGENT 2 SUCCESS: The model has evolved and studied the new patterns.");
            location.reload();
        } else {
            const msg = "[-] Retraining error: " + (result.log || result.message || "Unknown error");
            showRetrainStatus(msg);
            alert(msg);
        }
    } catch (error) {
        const msg = "[-] Agent 2 offline: " + error.message;
        showRetrainStatus(msg);
        alert(msg);
    } finally {
        btn.innerHTML = originalText;
        btn.disabled = false;
        btn.style.opacity = "1";
    }
}

async function loadMetricsReport() {
    try {
        const clipped = (v) => Math.max(0, Math.min(1, Number(v || 0)));
        document.getElementById('jsonDump').innerText = "Fetching detailed model evaluation metrics...";
            const sessionData = sessionStorage.getItem('scanData');
            if (sessionData) {
                try {
                    const scan = JSON.parse(sessionData);
                    const data = scan.formattedData;
                    document.getElementById('liveComparisonEmpty').style.display = 'none';
                    document.getElementById('liveComparisonBlock').style.display = 'block';
                    document.getElementById('live_pacx_diag').innerText = data.path_pacx.prediction;
                    document.getElementById('live_pacx_conf').innerText = (data.path_pacx.confidence * 100).toFixed(1) + '%';
                    document.getElementById('live_pacx_evidence').innerText = ((data.path_pacx.evidence_score || 0) * 100).toFixed(1) + '%';
                    document.getElementById('live_pacx_findings').innerText = (data.path_pacx.findings && data.path_pacx.findings.length)
                        ? data.path_pacx.findings.join(' | ')
                        : 'No PAC-X findings triggered.';

                    document.getElementById('live_gnn_diag').innerText = data.path_gnn.prediction;
                    document.getElementById('live_gnn_conf').innerText = (clipped(data.path_gnn.confidence) * 100).toFixed(1) + '%';
                    document.getElementById('live_gnn_family').innerText = data.path_gnn.predicted_family || '-';
                    document.getElementById('live_gnn_completeness').innerText = (clipped(data.path_gnn.completeness) * 100).toFixed(0) + '%';
                    document.getElementById('live_gnn_topclasses').innerText = (data.path_gnn.top_classes || [])
                        .map(item => `${item.label} ${(item.confidence * 100).toFixed(1)}%`)
                        .join(' | ') || 'No top classes available.';

                    document.getElementById('live_better_model').innerText = data.agent1.better_model;
                    document.getElementById('live_agreement').innerText = data.agent1.agreement ? 'PAC-X and GNN agree' : 'PAC-X and GNN disagree';
                    const trustScores = data.agent1.trust_scores || {};
                    document.getElementById('live_trust_scores').innerText = `PAC-X ${(trustScores.pacx || 0).toFixed(2)} / GNN ${(trustScores.gnn || 0).toFixed(2)}`;
                    document.getElementById('live_reasoning').innerHTML = `${reasoningSourceBadge(data.agent1.reasoning_source)}${data.agent1.reasoning}`;
                    const fusionData = data.decision_fusion || {
                        final_label: data.path_gnn?.prediction || data.path_pacx?.prediction || "-",
                        confidence: data.path_gnn?.confidence ?? data.path_pacx?.confidence ?? 0,
                        final_score: data.path_gnn?.confidence ?? data.path_pacx?.confidence ?? 0,
                        weights: { pacx: 0.4, gnn: 0.6 },
                        inputs: {
                            pacx_malicious_score: data.path_pacx?.confidence ?? 0,
                            gnn_malicious_score: data.path_gnn?.confidence ?? 0,
                            gnn_completeness: data.path_gnn?.completeness ?? 0,
                        }
                    };
                    const fusionWeights = fusionData.weights || {};
                    const fusionInputs = fusionData.inputs || {};
                    const liveFusionLabel = document.getElementById('live_fusion_label');
                    const liveFusionConf = document.getElementById('live_fusion_conf');
                    const liveFusionArbitration = document.getElementById('live_fusion_arbitration');
                    const liveFusionWeights = document.getElementById('live_fusion_weights');
                    const liveFusionInputs = document.getElementById('live_fusion_inputs');
                    if (liveFusionLabel) liveFusionLabel.innerText = fusionData.final_label || '-';
                    if (liveFusionConf) liveFusionConf.innerText = (clipped(fusionData.confidence) * 100).toFixed(1) + '%';
                    if (liveFusionArbitration) liveFusionArbitration.innerText = describeArbitration(fusionData);
                    if (liveFusionWeights) liveFusionWeights.innerText = `PAC-X ${(clipped(fusionWeights.pacx) * 100).toFixed(1)}% / GNN ${(clipped(fusionWeights.gnn) * 100).toFixed(1)}%`;
                    if (liveFusionInputs) {
                        liveFusionInputs.innerText =
                            `PAC-X score ${(clipped(fusionInputs.pacx_malicious_score) * 100).toFixed(1)}%, ` +
                            `GNN score ${(clipped(fusionInputs.gnn_malicious_score) * 100).toFixed(1)}%, ` +
                            `Completeness ${(clipped(fusionInputs.gnn_completeness) * 100).toFixed(0)}%`;
                    }
                    document.getElementById('jsonDump').innerText = `Loaded current scan comparison from session.\nTrusted model: ${data.agent1.better_model}\nFusion: ${fusionData.final_label || '-'} (${((clipped(fusionData.confidence) || 0) * 100).toFixed(1)}%)\nGNN family: ${data.path_gnn.predicted_family || '-'}`;
                } catch (e) {
                    console.error('Failed to populate live comparison block', e);
                }
            }
            const res = await fetch('/api/report');
            const data = await res.json();
            
            if (data.success) {
                const base = data.baseline_model;
                document.getElementById('rep_base_title').innerText = base.name;
                document.getElementById('rep_base_acc').innerText = base.accuracy.toFixed(2) + '%';
                document.getElementById('rep_base_pre').innerText = base.precision.toFixed(2) + '%';
                document.getElementById('rep_base_rec').innerText = base.recall.toFixed(2) + '%';
                document.getElementById('rep_base_f1').innerText = base.f1_score.toFixed(2) + '%';
                
                // Helper to render heatmap
                function renderMatrix(prefix, matrix, baseColor) {
                    const row1_total = matrix.tn + matrix.fp;
                    const row2_total = matrix.fn + matrix.tp;
                    
                    const tn_norm = row1_total ? (matrix.tn / row1_total) : 0;
                    const fp_norm = row1_total ? (matrix.fp / row1_total) : 0;
                    const fn_norm = row2_total ? (matrix.fn / row2_total) : 0;
                    const tp_norm = row2_total ? (matrix.tp / row2_total) : 0;
                    
                    document.getElementById(prefix + 'tn_norm').innerText = tn_norm.toFixed(2);
                    document.getElementById(prefix + 'tn').innerText = matrix.tn;
                    document.getElementById(prefix + 'fp_norm').innerText = fp_norm.toFixed(2);
                    document.getElementById(prefix + 'fp').innerText = matrix.fp;
                    document.getElementById(prefix + 'fn_norm').innerText = fn_norm.toFixed(2);
                    document.getElementById(prefix + 'fn').innerText = matrix.fn;
                    document.getElementById(prefix + 'tp_norm').innerText = tp_norm.toFixed(2);
                    document.getElementById(prefix + 'tp').innerText = matrix.tp;
                    
                    document.getElementById(prefix + 'tn_cell').style.backgroundColor = `rgba(${baseColor}, ${tn_norm * 0.8 + 0.1})`;
                    document.getElementById(prefix + 'fp_cell').style.backgroundColor = `rgba(${baseColor}, ${fp_norm * 0.8 + 0.1})`;
                    document.getElementById(prefix + 'fn_cell').style.backgroundColor = `rgba(${baseColor}, ${fn_norm * 0.8 + 0.1})`;
                    document.getElementById(prefix + 'tp_cell').style.backgroundColor = `rgba(${baseColor}, ${tp_norm * 0.8 + 0.1})`;
                    
                    document.getElementById(prefix + 'tn_cell').style.color = tn_norm > 0.5 ? '#fff' : '#cbd5e1';
                    document.getElementById(prefix + 'fp_cell').style.color = fp_norm > 0.5 ? '#fff' : '#cbd5e1';
                    document.getElementById(prefix + 'fn_cell').style.color = fn_norm > 0.5 ? '#fff' : '#cbd5e1';
                    document.getElementById(prefix + 'tp_cell').style.color = tp_norm > 0.5 ? '#fff' : '#cbd5e1';
                }

                renderMatrix('rep_base_', base.confusion_matrix, '34, 211, 238'); // Cyan theme

                const gnn = data.gnn_model;
                document.getElementById('rep_gnn_title').innerText = gnn.name;
                document.getElementById('rep_gnn_acc').innerText = gnn.accuracy.toFixed(2) + '%';
                document.getElementById('rep_gnn_pre').innerText = gnn.precision.toFixed(2) + '%';
                document.getElementById('rep_gnn_rec').innerText = gnn.recall.toFixed(2) + '%';
                document.getElementById('rep_gnn_f1').innerText = gnn.f1_score.toFixed(2) + '%';
                
                renderMatrix('rep_gnn_', gnn.confusion_matrix, '192, 132, 252'); // Purple theme

                renderMulticlassReport(data.gnn_model_multiclass);
                await loadAblationReport();

                document.getElementById('reportLoader').style.display = 'none';
                document.getElementById('reportContent').style.display = 'block';
                
                const ctx = document.getElementById('metricsComparisonChart').getContext('2d');
                if (window.metricsChartInstance) {
                    window.metricsChartInstance.destroy();
                }
                window.metricsChartInstance = new Chart(ctx, {
                    type: 'bar',
                    data: {
                        labels: ['Accuracy', 'Precision', 'Recall', 'F1-Score'],
                        datasets: [
                            {
                                label: base.name,
                                data: [base.accuracy, base.precision, base.recall, base.f1_score],
                                backgroundColor: 'rgba(34, 211, 238, 0.6)',
                                borderColor: '#22d3ee',
                                borderWidth: 1,
                                borderRadius: 4
                            },
                            {
                                label: gnn.name,
                                data: [gnn.accuracy, gnn.precision, gnn.recall, gnn.f1_score],
                                backgroundColor: 'rgba(74, 222, 128, 0.6)',
                                borderColor: '#4ade80',
                                borderWidth: 1,
                                borderRadius: 4
                            }
                        ]
                    },
                    options: {
                        responsive: true,
                        maintainAspectRatio: false,
                        plugins: {
                            legend: { position: 'top', labels: { color: '#cbd5e1', font: {family: 'Inter'} } }
                        },
                        scales: {
                            y: {
                                beginAtZero: false,
                                min: 60,
                                max: 100,
                                grid: { color: 'rgba(255, 255, 255, 0.05)' },
                                ticks: { color: '#94a3b8' }
                            },
                            x: {
                                grid: { display: false },
                                ticks: { color: '#94a3b8' }
                            }
                        }
                    }
                });
            } else {
                document.getElementById('reportLoader').innerText =
                    data.detail || 'No dataset report available yet -- run agentic_pacx/gnn_classification.py to train the model first.';
            }
        } catch (e) {
            document.getElementById('reportLoader').innerText = 'Failed to load detailed report. Server offline.';
            console.error(e);
        }
}

async function loadAblationReport() {
    const emptyEl = document.getElementById('ablationEmpty');
    const blockEl = document.getElementById('ablationBlock');
    try {
        const res = await fetch('/api/ablation');
        if (!res.ok) {
            if (emptyEl) emptyEl.style.display = 'block';
            if (blockEl) blockEl.style.display = 'none';
            return;
        }
        const data = await res.json();
        if (emptyEl) emptyEl.style.display = 'none';
        if (blockEl) blockEl.style.display = 'block';

        document.getElementById('ablation_note').innerText = data.note || '';

        const rows = [
            ['GNN-only', data.gnn_only],
            ['PAC-X-only', data.pacx_only],
            ['Fused (with arbitration)', data.fused],
        ];
        const tbody = document.getElementById('ablation_body');
        tbody.innerHTML = '';
        for (const [name, m] of rows) {
            const tr = document.createElement('tr');
            tr.style.borderBottom = '1px solid rgba(255,255,255,0.05)';
            tr.innerHTML = `<td style="padding:6px 0;">${name}</td><td>${m.accuracy.toFixed(2)}%</td><td>${m.precision.toFixed(2)}%</td><td>${m.recall.toFixed(2)}%</td><td>${m.f1_score.toFixed(2)}%</td>`;
            tbody.appendChild(tr);
        }

        const ctx = document.getElementById('ablationChart').getContext('2d');
        if (window.ablationChartInstance) window.ablationChartInstance.destroy();
        window.ablationChartInstance = new Chart(ctx, {
            type: 'bar',
            data: {
                labels: ['Accuracy', 'Precision', 'Recall', 'F1-Score'],
                datasets: rows.map(([name, m], i) => ({
                    label: name,
                    data: [m.accuracy, m.precision, m.recall, m.f1_score],
                    backgroundColor: [
                        'rgba(34, 211, 238, 0.6)',
                        'rgba(248, 113, 113, 0.6)',
                        'rgba(74, 222, 128, 0.6)',
                    ][i],
                    borderColor: ['#22d3ee', '#f87171', '#4ade80'][i],
                    borderWidth: 1,
                    borderRadius: 4,
                })),
            },
            options: {
                responsive: true, maintainAspectRatio: false,
                plugins: { legend: { position: 'top', labels: { color: '#cbd5e1' } } },
                scales: {
                    y: { min: 0, max: 100, grid: { color: 'rgba(255,255,255,0.05)' }, ticks: { color: '#94a3b8' } },
                    x: { grid: { display: false }, ticks: { color: '#94a3b8' } },
                },
            },
        });
    } catch (e) {
        console.error('Failed to load ablation report', e);
        if (emptyEl) emptyEl.style.display = 'block';
        if (blockEl) blockEl.style.display = 'none';
    }
}

function renderMulticlassReport(mc) {
    const emptyEl = document.getElementById('multiclassEmpty');
    const blockEl = document.getElementById('multiclassBlock');
    if (!mc) {
        if (emptyEl) emptyEl.style.display = 'block';
        if (blockEl) blockEl.style.display = 'none';
        return;
    }
    if (emptyEl) emptyEl.style.display = 'none';
    if (blockEl) blockEl.style.display = 'block';

    document.getElementById('mc_accuracy').innerText = mc.accuracy.toFixed(2) + '%';
    document.getElementById('mc_macro_p').innerText = mc.macro_precision.toFixed(2) + '%';
    document.getElementById('mc_macro_r').innerText = mc.macro_recall.toFixed(2) + '%';
    document.getElementById('mc_macro_f1').innerText = mc.macro_f1.toFixed(2) + '%';

    const tbody = document.getElementById('mc_per_class_body');
    tbody.innerHTML = '';
    for (const row of mc.per_class) {
        const tr = document.createElement('tr');
        tr.style.borderBottom = '1px solid rgba(255,255,255,0.05)';
        tr.innerHTML =
            `<td style="padding: 6px 0; text-transform: capitalize;">${row.label}</td>` +
            `<td>${row.precision.toFixed(1)}%</td>` +
            `<td>${row.recall.toFixed(1)}%</td>` +
            `<td>${row.f1_score.toFixed(1)}%</td>` +
            `<td>${row.support}</td>`;
        tbody.appendChild(tr);
    }

    // Confusion matrix heatmap, generated from real data (rows=true, cols=predicted)
    const labels = mc.class_labels;
    const matrix = mc.confusion_matrix;
    const maxVal = Math.max(1, ...matrix.flat());
    let html = '<table class="cm-table" style="min-width: 700px;"><thead><tr><th></th>';
    for (const label of labels) {
        html += `<th style="font-size:0.7rem; color:#94a3b8; padding:4px; writing-mode: vertical-rl; text-orientation: mixed;">${label}</th>`;
    }
    html += '</tr></thead><tbody>';
    for (let i = 0; i < labels.length; i++) {
        html += `<tr><td style="font-size:0.75rem; color:#94a3b8; text-align:right; padding-right:8px; white-space:nowrap;">${labels[i]}</td>`;
        for (let j = 0; j < labels.length; j++) {
            const val = matrix[i][j];
            const intensity = val / maxVal;
            const bg = i === j
                ? `rgba(74, 222, 128, ${0.15 + intensity * 0.7})`
                : `rgba(248, 113, 113, ${val > 0 ? 0.15 + intensity * 0.7 : 0.03})`;
            html += `<td class="cm-cell" style="background:${bg}; padding:8px 4px; font-size:0.8rem;">${val}</td>`;
        }
        html += '</tr>';
    }
    html += '</tbody></table>';
    document.getElementById('mc_heatmap_wrapper').innerHTML = html;
}

function describeArbitration(fusionData) {
    const arbitration = fusionData.arbitration || '';
    if (fusionData.agreement) {
        return 'PAC-X and GNN agreed -- scores blended';
    }
    if (arbitration.startsWith('arbitrated_by_trust:')) {
        const winner = arbitration.split(':')[1] === 'gnn' ? 'GNN' : 'PAC-X';
        return `Disagreed -- deferred to ${winner} (higher trust score)`;
    }
    return 'Blended (fallback)';
}

function sleep(ms) { return new Promise(r => setTimeout(r, ms)); }
