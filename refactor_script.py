import os

path = 'c:/Users/THIS PC/Downloads/PACX-main/web/static/js/script.js'
with open(path, 'r', encoding='utf-8') as f:
    js_code = f.read()

# 1. Update runAnalysis to save sessionStorage and enable buttons
old_runAnalysis_end = '''        document.getElementById('agent1_reasoning').innerHTML = agent1Decision.reasoning;
        document.getElementById('jsonDump').innerHTML = "<b>Recommendation:</b> " + result.agent1_recommendation + "<br><br><b>Agent 2 Status:</b> " + (result.agent2.triggered ? "⚠️ ACTIVATED" : "✓ Standby");
    } catch (error) {'''

new_runAnalysis_end = '''        document.getElementById('agent1_reasoning').innerHTML = agent1Decision.reasoning;
        document.getElementById('jsonDump').innerHTML = "<b>Recommendation:</b> " + result.agent1_recommendation + "<br><br><b>Agent 2 Status:</b> " + (result.agent2.triggered ? "⚠️ ACTIVATED" : "✓ Standby");
        
        // Save to session storage for multi-page routing
        sessionStorage.setItem('scanData', JSON.stringify({formattedData, gnnConf}));
        
        // Update Button 1 text based on which model performed better
        const bestModel = agent1Decision.better_model === 'GNN' ? 'GNN (Agentic-PACX)' : 'Alternative Baseline Model';
        const btn1Text = document.getElementById('btn1_text');
        if (btn1Text) btn1Text.innerText = 'Know more why ' + bestModel + ' is best';
        
        // Unhide navigation buttons
        const btn1 = document.getElementById('nav_btn_1');
        const btn2 = document.getElementById('nav_btn_2');
        if (btn1) btn1.style.display = 'block';
        if (btn2) btn2.style.display = 'block';
        
    } catch (error) {'''

js_code = js_code.replace(old_runAnalysis_end, new_runAnalysis_end)

# 2. Update window.onload
old_onload = '''window.onload = async function () {
    const pageType = document.body.getAttribute('data-page');
    if (pageType === "static") {
        initCharts(true);
        try {
            const res = await fetch('/api/results');
            const data = await res.json();
            // Update charts with real aggregated data if available
            if (data.showdown) {
                const gnn = data.showdown.metrics["GNN (Agentic-PACX)"];
                gaugeChart.data.datasets[0].data = [gnn.Accuracy, 100 - gnn.Accuracy];
                gaugeChart.update();
            }
        } catch (e) { console.log("Baseline load..."); }
    } else {
        initCharts(false);
        // Bind File Upload Reader natively on Live Dashboard
        const uploader = document.getElementById('fileUploader');
        if (uploader) {
            uploader.addEventListener('change', function (e) {
                const file = e.target.files[0];
                if (!file) return;
                const reader = new FileReader();
                reader.onload = function (evt) {
                    document.getElementById('payloadInput').value = evt.target.result;
                    document.getElementById('jsonDump').innerText = "[*] File '" + file.name + "' successfully loaded into memory.\\n[*] Ready for Analysis.";
                };
                reader.readAsText(file);
            });
        }
    }
};'''

new_onload = '''window.onload = async function () {
    const pageType = document.body.getAttribute('data-page');
    
    if (pageType === "static") {
        initCharts(true);
        try {
            const res = await fetch('/api/results');
            const data = await res.json();
            if (data.showdown) {
                const gnn = data.showdown.metrics["GNN (Agentic-PACX)"];
                gaugeChart.data.datasets[0].data = [gnn.Accuracy, 100 - gnn.Accuracy];
                gaugeChart.update();
            }
        } catch (e) { console.log("Baseline load..."); }
        
    } else if (pageType === "live") {
        const uploader = document.getElementById('fileUploader');
        if (uploader) {
            uploader.addEventListener('change', function (e) {
                const file = e.target.files[0];
                if (!file) return;
                const reader = new FileReader();
                reader.onload = function (evt) {
                    document.getElementById('payloadInput').value = evt.target.result;
                    document.getElementById('jsonDump').innerText = "[*] File '" + file.name + "' successfully loaded into memory.\\n[*] Ready for Analysis.";
                };
                reader.readAsText(file);
            });
        }
        
    } else if (pageType === "metrics") {
        loadMetricsReport(); // Load the detailed comparison from API
        
    } else if (pageType === "visuals") {
        initCharts(false); // Setup empty graphs
        const sessionStore = sessionStorage.getItem('scanData');
        if (sessionStore) {
            try {
                const data = JSON.parse(sessionStore);
                document.getElementById('jsonDump').innerText = "Rendering visual graphs from recent analysis scan...";
                updateDashboard(data.formattedData, data.gnnConf); // Fill the graphs with the stored data
            } catch (e) {
                document.getElementById('jsonDump').innerText = "Error reading session data.";
            }
        } else {
            document.getElementById('jsonDump').innerText = "No recent analysis found. Please run a scan on the Live Dashboard.";
            document.getElementById('visualLoader').innerText = "Waiting for data...";
        }
    }
};'''

js_code = js_code.replace(old_onload, new_onload)


# 3. Rename toggleAnalysisReport to loadMetricsReport and remove the display toggle
old_toggle = '''async function toggleAnalysisReport() {
    const container = document.getElementById('detailedReportContainer');
    const chevron = document.getElementById('reportChevron');
    
    if (container.style.display === 'none' || container.style.display === '') {
        container.style.display = 'block';
        chevron.innerText = '▲';
        
        try {'''

new_toggle = '''async function loadMetricsReport() {
    try {
        document.getElementById('jsonDump').innerText = "Fetching detailed model evaluation metrics...";'''

js_code = js_code.replace(old_toggle, new_toggle)

# Remove the else block of the old toggle function
old_else_block = '''        } catch (e) {
            document.getElementById('reportLoader').innerText = "Failed to load detailed report. Server offline.";
            console.error(e);
        }
    } else {
        container.style.display = 'none';
        chevron.innerText = '▼';
    }
}'''

new_else_block = '''        } catch (e) {
        document.getElementById('reportLoader').innerText = "Failed to load detailed report. Server offline.";
        console.error(e);
    }
}'''

js_code = js_code.replace(old_else_block, new_else_block)


# 4. Safely update Visualizations
old_update_dash_charts = '''    // 3. Update Visualizations
    // Gauge
    gaugeChart.data.datasets[0].data = [gnnConf * 100, (1 - gnnConf) * 100];
    gaugeChart.data.datasets[0].backgroundColor = [gnnConf < 0.6 ? '#f87171' : '#4ade80', '#1e2d3d'];
    gaugeChart.update();

    // Radar (GNN Attention)
    radarChart.data.datasets[0].data = data.path_gnn.attention_weights || [0.25, 0.25, 0.25, 0.25];
    radarChart.update();

    // Bar (PAC-X Behavioral Findings)
    const findings = data.path_pacx.findings || ["Header", "Entropy", "API", "Network"];
    barChart.data.labels = findings.slice(0, 4).map(f => f.split(' ')[0]);
    barChart.data.datasets[0].data = findings.slice(0, 4).map((_, i) => 100 - (i * 20));
    barChart.update();'''

new_update_dash_charts = '''    // 3. Update Visualizations (Only if charts are initialized on this page)
    if (typeof gaugeChart !== 'undefined' && document.getElementById('gaugeChart')) {
        gaugeChart.data.datasets[0].data = [gnnConf * 100, (1 - gnnConf) * 100];
        gaugeChart.data.datasets[0].backgroundColor = [gnnConf < 0.6 ? '#f87171' : '#4ade80', '#1e2d3d'];
        gaugeChart.update();
        
        const gLabel = document.getElementById('gauge_confidence');
        if(gLabel) {
            gLabel.innerText = (gnnConf * 100).toFixed(1) + '%';
            gLabel.style.color = gnnConf < 0.6 ? '#f87171' : '#4ade80';
        }
        
        const loader = document.getElementById('visualLoader');
        const content = document.getElementById('visualContent');
        if(loader) loader.style.display = 'none';
        if(content) content.style.display = 'grid';
    }

    if (typeof radarChart !== 'undefined' && document.getElementById('radarChart')) {
        radarChart.data.datasets[0].data = data.path_gnn.attention_weights || [0.25, 0.25, 0.25, 0.25];
        radarChart.update();
    }

    if (typeof barChart !== 'undefined' && document.getElementById('barChart')) {
        const findings = data.path_pacx.findings || ["Header", "Entropy", "API", "Network"];
        barChart.data.labels = findings.slice(0, 4).map(f => f.split(' ')[0]);
        barChart.data.datasets[0].data = findings.slice(0, 4).map((_, i) => 100 - (i * 20));
        barChart.update();
    }'''

js_code = js_code.replace(old_update_dash_charts, new_update_dash_charts)

with open(path, 'w', encoding='utf-8') as f:
    f.write(js_code)
print("SUCCESS!")
