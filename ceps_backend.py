<!DOCTYPE html>
<html lang="sk">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>Avalanche Trade – Dispečing (ČEPS + OBSyd)</title>
  <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
  <script src="https://cdn.jsdelivr.net/npm/chartjs-plugin-annotation@3.0.1/dist/chartjs-plugin-annotation.min.js"></script>
  <style>
    body { font-family: system-ui, sans-serif; background: #0b1220; color: #e5e7eb; margin: 0; padding: 20px; }
    h1 { margin-top: 0; margin-bottom: 5px; font-size: 24px; }
    .subtitle { color: #9ca3af; margin-bottom: 20px; font-size: 14px; }
    
    .dashboard {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(500px, 1fr));
      gap: 20px;
      margin-bottom: 20px;
    }
    
    .card { 
      background: #111827; 
      border-radius: 10px; 
      padding: 16px; 
      border: 1px solid #1f2937;
      display: flex;
      flex-direction: column;
    }
    
    .card-header {
      display: flex;
      justify-content: space-between;
      align-items: flex-start;
      margin-bottom: 15px;
      padding-bottom: 10px;
      border-bottom: 1px solid #1f2937;
    }
    
    .header-titles {
      display: flex;
      flex-direction: column;
      gap: 4px;
    }
    
    .card h2 { margin: 0; font-size: 16px; font-weight: 600; }
    .status { font-size: 12px; color: #6b7280; font-style: italic; }
    
    .latest-data {
      text-align: right;
    }
    
    .latest-value {
      font-size: 20px;
      font-weight: bold;
      line-height: 1.2;
    }

    .svr-container {
      display: flex;
      flex-direction: column;
      align-items: flex-end;
      gap: 6px;
    }

    .svr-main-val {
      font-size: 20px;
      font-weight: bold;
      line-height: 1;
    }

    .svr-grid {
      display: flex;
      flex-direction: column;
      gap: 2px;
      text-align: right;
    }

    .svr-row {
      display: flex;
      justify-content: flex-end;
      gap: 8px;
      font-size: 11px;
      font-weight: 600;
    }
    
    .latest-time {
      font-size: 11px;
      color: #9ca3af;
      margin-top: 2px;
      text-align: right;
    }
    
    .chart-container {
      position: relative;
      height: 250px;
      width: 100%;
    }

    /* Legenda a ovládacie prvky pre OBSyd */
    .obsyd-controls {
      display: flex;
      gap: 10px;
      align-items: center;
    }
    .obsyd-select {
      background: #1f2937;
      color: #e5e7eb;
      border: 1px solid #374151;
      border-radius: 4px;
      padding: 4px 8px;
      font-size: 12px;
      outline: none;
    }
    .metric-legend {
      font-size: 11px;
      color: #9ca3af;
      margin-top: 8px;
      padding: 6px 10px;
      background: rgba(31, 41, 55, 0.4);
      border-radius: 4px;
      border-left: 3px solid #38bdf8;
    }

    /* Tabuľka */
    .vdt-table-container {
      margin-top: 15px;
      max-height: 200px;
      overflow-y: auto;
      border: 1px solid #1f2937;
      border-radius: 6px;
    }
    .vdt-table {
      width: 100%;
      border-collapse: collapse;
      font-size: 13px;
      text-align: left;
    }
    .vdt-table th, .vdt-table td {
      padding: 8px 12px;
      border-bottom: 1px solid #1f2937;
    }
    .vdt-table th {
      background: #1f2937;
      color: #9ca3af;
      position: sticky;
      top: 0;
    }
    .vdt-table tr:hover {
      background: rgba(56, 189, 248, 0.05);
    }
    
    @media (max-width: 600px) {
      .dashboard { grid-template-columns: 1fr; }
    }
  </style>
</head>
<body>
  <h1>Avalanche Trade Dispečing</h1>
  <div class="subtitle">Live real-time energetický prehľad (ČEPS + OBSYD API)</div>

  <!-- RAD 1: GRAFY SVR A SYSTEMOVEJ ODCHÝLKY -->
  <div class="dashboard">
    <div class="card">
      <div class="card-header">
        <div class="header-titles">
            <h2>Aktivace SVR v ČR</h2>
            <span class="status" id="status-aktivace-svr">Čakám na dáta...</span>
        </div>
        <div class="latest-data">
            <div class="svr-container">
              <div class="svr-main-val" id="val-aktivace-svr-main">-- MW</div>
              <div class="svr-grid" id="val-aktivace-svr-grid">
                <div class="svr-row">
                  <span style="color: #cc3f44;">aFRR+: --</span>
                  <span style="color: #a3a6a9;">aFRR-: --</span>
                  <span style="color: #facc15;">mFRR+: --</span>
                </div>
                <div class="svr-row">
                  <span style="color: #38bdf8;">mFRR-: --</span>
                  <span style="color: #4ade80;">mFRR5: --</span>
                </div>
              </div>
            </div>
            <div class="latest-time" id="time-aktivace-svr"></div>
        </div>
      </div>
      <div class="chart-container"><canvas id="chart-aktivace-svr"></canvas></div>
    </div>

    <div class="card">
      <div class="card-header">
        <div class="header-titles">
            <h2>Systémová odchýlka ČR</h2>
            <span class="status" id="status-systemova-odchylka">Čakám na dáta...</span>
        </div>
        <div class="latest-data">
            <div class="latest-value" id="val-systemova-odchylka">--</div>
            <div class="latest-time" id="time-systemova-odchylka"></div>
        </div>
      </div>
      <div class="chart-container"><canvas id="chart-systemova-odchylka"></canvas></div>
    </div>
  </div>

  <!-- RAD 2: CENOVÉ GRAFY -->
  <div class="dashboard">
    <div class="card">
      <div class="card-header">
        <div class="header-titles">
            <h2>Odhadovaná cena odchýlky</h2>
            <span class="status" id="status-odchylka">Čakám na dáta...</span>
        </div>
        <div class="latest-data">
            <div class="latest-value" id="val-odchylka">--</div>
            <div class="latest-time" id="time-odchylka"></div>
        </div>
      </div>
      <div class="chart-container"><canvas id="chart-odchylka"></canvas></div>
    </div>

    <div class="card">
      <div class="card-header">
        <div class="header-titles">
            <h2>Aktuální cena RE v EUR</h2>
            <span class="status" id="status-cena-re">Čakám na dáta...</span>
        </div>
        <div class="latest-data">
            <div class="svr-container">
              <div class="svr-main-val" id="val-cena-re-main">-- €/MWh</div>
              <div class="svr-grid" id="val-cena-re-grid">
                <div class="svr-row">
                  <span style="color: #cc3f44;">aFRR: --</span>
                  <span style="color: #a3a6a9;">mFRR+: --</span>
                </div>
                <div class="svr-row">
                  <span style="color: #facc15;">mFRR-: --</span>
                  <span style="color: #38bdf8;">mFRR5: --</span>
                </div>
              </div>
            </div>
            <div class="latest-time" id="time-cena-re"></div>
        </div>
      </div>
      <div class="chart-container"><canvas id="chart-cena-re"></canvas></div>
    </div>
  </div>

  <!-- RAD 3: OBSYD API SEKCIA S LEGENDOU -->
  <div class="dashboard">
    <div class="card" style="grid-column: 1 / -1;">
      <div class="card-header">
        <div class="header-titles">
            <h2>OBSYD – Európske energetické dáta</h2>
            <span class="status" id="status-obsyd">Čakám na dáta...</span>
        </div>
        <div class="obsyd-controls">
            <select id="obsyd-metric-select" class="obsyd-select" onchange="loadObsydData()">
              <option value="dayahead">Day-ahead ceny (Hodinové)</option>
              <option value="dayahead-qh">Day-ahead ceny (15-min)</option>
              <option value="load">Skutočná spotreba (Load)</option>
              <option value="genmix">Generation Mix</option>
              <option value="flows">Cross-border Flows</option>
            </select>
            <select id="obsyd-zone-select" class="obsyd-select" onchange="loadObsydData()">
              <option value="CZ">CZ</option>
              <option value="SK">SK</option>
              <option value="DE_LU">DE_LU</option>
              <option value="FR">FR</option>
              <option value="AT">AT</option>
            </select>
            <div class="latest-data" style="margin-left: 15px;">
                <div class="latest-value" id="val-obsyd">--</div>
                <div class="latest-time" id="time-obsyd"></div>
            </div>
        </div>
      </div>
      <div class="chart-container"><canvas id="chart-obsyd"></canvas></div>
      <div class="metric-legend" id="obsyd-legend">Legenda: Načítavam popis metriky...</div>
      
      <!-- Tabuľka pre OBSyd dáta -->
      <div class="vdt-table-container">
        <table class="vdt-table">
          <thead>
            <tr>
              <th>Čas / Záznam</th>
              <th>Hodnota</th>
            </tr>
          </thead>
          <tbody id="obsyd-table-body">
            <!-- Dynamicky naplnené cez JS -->
          </tbody>
        </table>
      </div>
    </div>
  </div>

  <script>
    const BASE_BACKEND = "https://tatrameteo-api-5.onrender.com";
    
    const charts = {};

    const metricsConfig = {
      'aktivace-svr': { canvasId: 'chart-aktivace-svr', statusId: 'status-aktivace-svr', color: '#10b981', label: 'MW', type: 'bar' },
      'systemova-odchylka': { canvasId: 'chart-systemova-odchylka', statusId: 'status-systemova-odchylka', color: '#10b981', label: 'MW', type: 'line' },
      'odchylka': { canvasId: 'chart-odchylka', statusId: 'status-odchylka', color: '#3b82f6', label: 'Kč / MWh', type: 'line' },
      'cena-re': { canvasId: 'chart-cena-re', statusId: 'status-cena-re', color: '#f59e0b', label: 'EUR / MWh', type: 'bar' }
    };

    const obsydDescriptions = {
      'dayahead': 'Day-ahead ceny (SDAC): Hodinové clearingové ceny elektriny pre zvolenú bidding zónu v EUR/MWh.',
      'dayahead-qh': 'Day-ahead ceny (QH): 15-minútové detailné clearingové ceny pre trh s denným vopred dohodnutým dodaním.',
      'load': 'Skutočná spotreba (Load): Reálne zaťaženie a celkový dopyt po elektrine v danej prenosovej sústave (MW).',
      'genmix': 'Generation Mix: Rozloženie aktuálnej výroby elektriny podľa jednotlivých technológií (jadro, obnoviteľné zdroje, fosílne palivá).',
      'flows': 'Cross-border Flows: Fyzické alebo obchodné medzinárodné cezhraničné toky elektriny medzi prenosovými sústavami.'
    };

    async function loadMetric(metric) {
      const config = metricsConfig[metric];
      const statusEl = document.getElementById(config.statusId);
      
      statusEl.textContent = "Aktualizujem...";
      statusEl.style.color = "#6b7280";

      try {
        const res = await fetch(`${BASE_BACKEND}/api/data/${metric}`);
        if (!res.ok) throw new Error(`Chyba API ${res.status}`);
        
        const data = await res.json();
        let items = data.items || (Array.isArray(data) ? data : (data.fallback || []));
        const seriesMap = data.series || {};
        const seriesKeys = Object.keys(seriesMap);
        
        const threeHoursAgo = Date.now() - (3 * 60 * 60 * 1000);
        let filteredItems = items.filter(d => {
            const rawDate = d.date || d.time;
            if (!rawDate) return false;
            return new Date(rawDate).getTime() >= threeHoursAgo;
        });
        if (filteredItems.length === 0) filteredItems = items; 
        items = filteredItems;
        
        const labels = items.map(d => {
            const rawDate = d.date || d.time;
            if (rawDate) {
                const dateObj = new Date(rawDate);
                if (!isNaN(dateObj)) return dateObj.toLocaleTimeString('sk-SK', { hour: '2-digit', minute: '2-digit' });
            }
            return rawDate || "";
        });

        const datasets = [];

        if (metric === 'aktivace-svr' && seriesKeys.length > 0) {
            seriesKeys.forEach(key => {
                const seriesName = seriesMap[key];
                let color = '#10b981';
                if (seriesName.includes('aFRR+')) color = '#cc3f44';
                else if (seriesName.includes('aFRR-')) color = '#a3a6a9';
                else if (seriesName.includes('mFRR+')) color = '#facc15';
                else if (seriesName.includes('mFRR-')) color = '#38bdf8';
                else if (seriesName.includes('mFRR5')) color = '#4ade80';

                datasets.push({
                    label: seriesName,
                    data: items.map(d => parseFloat(d[key] || 0)),
                    backgroundColor: color,
                    borderColor: color,
                    borderWidth: 0,
                    pointRadius: 0,
                    fill: false,
                    tension: 0.2
                });
            });
        } else if (metric === 'cena-re' && seriesKeys.length > 0) {
            seriesKeys.forEach(key => {
                const seriesName = seriesMap[key];
                let color = '#f59e0b';
                if (seriesName.includes('aFRR')) color = '#cc3f44';
                else if (seriesName.includes('mFRR+')) color = '#a3a6a9';
                else if (seriesName.includes('mFRR-')) color = '#facc15';
                else if (seriesName.includes('mFRR5')) color = '#38bdf8';

                datasets.push({
                    label: seriesName,
                    data: items.map(d => parseFloat(d[key] || 0)),
                    backgroundColor: color,
                    borderColor: color,
                    borderWidth: 0,
                    pointRadius: 0,
                    fill: true,
                    tension: 0.2
                });
            });
        } else {
            const values = items.map(d => {
                let val = d.value1 || d.value2 || d.value3 || d.value || Object.entries(d).find(([k, v]) => k !== 'date' && k !== 'time' && !isNaN(parseFloat(v)))?.[1];
                return parseFloat(val || 0);
            });
            
            let datasetConfig = {
                label: config.label,
                data: values,
                borderWidth: 2,
                pointRadius: 0,
                tension: 0.2,
                fill: false,
                borderColor: config.color
            };

            if (metric === 'systemova-odchylka') {
                datasetConfig.borderColor = '#10b981';
                datasetConfig.segment = {
                    borderColor: ctx => ctx.p0.parsed.y < 0 || ctx.p1.parsed.y < 0 ? '#ef4444' : '#10b981'
                };
            }
            datasets.push(datasetConfig);
        }

        if (labels.length > 0) {
            let lastIndex = labels.length - 1;
            const now = new Date();
            const timeWithSecondsStr = now.toLocaleTimeString('sk-SK', { hour: '2-digit', minute: '2-digit', second: '2-digit' });

            if (metric === 'aktivace-svr') {
                let netSvr = 0;
                datasets.forEach(ds => { netSvr += (ds.data[lastIndex] || 0); });
                const mainValEl = document.getElementById(`val-aktivace-svr-main`);
                if (mainValEl) {
                    mainValEl.textContent = netSvr.toLocaleString('sk-SK', { minimumFractionDigits: 1, maximumFractionDigits: 1 }) + " MW";
                    mainValEl.style.color = netSvr >= 0 ? '#10b981' : '#ef4444';
                }
            } else if (metric === 'cena-re' && seriesKeys.length > 0) {
                let totalRe = 0;
                datasets.forEach(ds => { totalRe += (ds.data[lastIndex] || 0); });
                const mainValEl = document.getElementById(`val-cena-re-main`);
                if (mainValEl) mainValEl.textContent = totalRe.toLocaleString('sk-SK', { minimumFractionDigits: 1, maximumFractionDigits: 1 }) + " €/MWh";
            } else {
                const valEl = document.getElementById(`val-${metric}`);
                let val = datasets[0].data[lastIndex];
                if (valEl) {
                    let unit = metric === 'systemova-odchylka' ? ' MW' : (metric === 'odchylka' ? ' Kč' : ' €');
                    valEl.innerHTML = `${val.toLocaleString('sk-SK', { minimumFractionDigits: 2, maximumFractionDigits: 2 })} ${unit} <span style="font-size: 12px; color: #9ca3af;">(${timeWithSecondsStr})</span>`;
                    valEl.style.color = metric === 'systemova-odchylka' ? (val >= 0 ? '#10b981' : '#ef4444') : config.color;
                }
            }
        }

        updateChart(metric, labels, datasets);
        statusEl.textContent = "Pripojené";
        statusEl.style.color = "#10b981"; 
      } catch (err) {
        statusEl.textContent = "Chyba načítania";
        statusEl.style.color = "#ef4444"; 
      }
    }

    // --- OBSYD INTEGRÁCIA S LEGENDOU ---
    async function loadObsydData() {
        const metric = document.getElementById('obsyd-metric-select').value;
        const zone = document.getElementById('obsyd-zone-select').value;
        const statusEl = document.getElementById('status-obsyd');
        const valEl = document.getElementById('val-obsyd');
        const tableBody = document.getElementById('obsyd-table-body');
        const legendEl = document.getElementById('obsyd-legend');
        
        statusEl.textContent = "Aktualizujem...";
        statusEl.style.color = "#6b7280";
        legendEl.textContent = obsydDescriptions[metric] || 'Európske energetické údaje z oficiálnych registrov.';

        try {
            const res = await fetch(`${BASE_BACKEND}/api/obsyd/${metric}/${zone}`);
            if (!res.ok) throw new Error(`Chyba API ${res.status}`);
            
            const data = await res.json();
            if (!Array.isArray(data) || data.length === 0) throw new Error("Žiadne dáta");

            const labels = data.map(d => {
                const raw = d.time || d.date || d.timestamp || Object.values(d)[0];
                const dt = new Date(raw);
                return !isNaN(dt) ? dt.toLocaleTimeString('sk-SK', { hour: '2-digit', minute: '2-digit' }) : (raw || "");
            });

            let valueKey = Object.keys(data[0]).find(k => k !== 'time' && k !== 'date' && typeof data[0][k] === 'number') || Object.keys(data[0])[1];
            const values = data.map(d => parseFloat(d[valueKey] || Object.values(d).find(v => !isNaN(parseFloat(v))) || 0));

            const datasets = [{
                label: `${metric.toUpperCase()} (${zone})`,
                data: values,
                borderColor: '#38bdf8',
                backgroundColor: 'rgba(56, 189, 248, 0.1)',
                borderWidth: 2,
                pointRadius: 0,
                fill: true,
                tension: 0.2
            }];

            let lastVal = values[values.length - 1] || 0;
            valEl.textContent = lastVal.toLocaleString('sk-SK', { minimumFractionDigits: 2, maximumFractionDigits: 2 });
            valEl.style.color = '#38bdf8';

            let tableRows = '';
            data.forEach((d, idx) => {
                let timeStr = d.time || d.date || labels[idx];
                let valStr = values[idx].toLocaleString('sk-SK', { minimumFractionDigits: 2, maximumFractionDigits: 2 });
                tableRows += `<tr><td>${timeStr}</td><td style="color: #38bdf8; font-weight: bold;">${valStr}</td></tr>`;
            });
            tableBody.innerHTML = tableRows;

            updateChart('obsyd', labels, datasets);
            statusEl.textContent = "Pripojené";
            statusEl.style.color = "#10b981";
        } catch (err) {
            statusEl.textContent = "Chyba načítania";
            statusEl.style.color = "#ef4444";
            console.error("OBSyd Error:", err);
        }
    }

    function updateChart(metric, labels, datasets) {
      const isObsyd = (metric === 'obsyd');
      const config = isObsyd ? { type: 'line', canvasId: 'chart-obsyd' } : metricsConfig[metric];
      const ctx = document.getElementById(config.canvasId).getContext("2d");
      
      const isStacked = (!isObsyd && config.type === 'bar');

      let pluginsConfig = {
          legend: { 
              display: isStacked, 
              position: 'bottom',
              labels: { color: '#9ca3af', usePointStyle: true, boxWidth: 10, font: { size: 10 } }
          }
      };

      if (metric === 'systemova-odchylka') {
          pluginsConfig.annotation = {
              annotations: {
                  zeroLine: {
                      type: 'line',
                      yMin: 0,
                      yMax: 0,
                      borderColor: '#e5e7eb',
                      borderWidth: 1.5,
                      borderDash: [4, 4]
                  }
              }
          };
      }

      if (!charts[metric]) {
        charts[metric] = new Chart(ctx, {
          type: config.type,
          data: {
            labels: labels,
            datasets: datasets
          },
          options: {
              responsive: true,
              maintainAspectRatio: false,
              plugins: pluginsConfig,
              scales: {
                  x: { 
                      stacked: isStacked, 
                      ticks: { 
                          color: '#9ca3af', 
                          maxTicksLimit: 8, 
                          autoSkip: true 
                      }, 
                      grid: { color: '#374151', drawBorder: false } 
                  },
                  y: { 
                      stacked: isStacked, 
                      ticks: { color: '#9ca3af' }, 
                      grid: { color: '#374151', drawBorder: false } 
                  }
              }
          }
        });
      } else {
        charts[metric].data.labels = labels;
        charts[metric].data.datasets = datasets;
        charts[metric].options.plugins = pluginsConfig;
        charts[metric].update('none'); 
      }
    }

    async function loadAllMetrics() {
        for (const metric of Object.keys(metricsConfig)) {
            await loadMetric(metric); 
            await new Promise(resolve => setTimeout(resolve, 500));
        }
        await loadObsydData();
    }

    loadAllMetrics();
    setInterval(loadAllMetrics, 30000); 
  </script>
</body>
</html>
