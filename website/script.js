// Pong RL Laboratory: Dashboard Telemetry and Replay Pipeline

document.addEventListener('DOMContentLoaded', () => {
    // ─── STATE VARIABLES FOR TELEMETRY POLLING ───
    let autoRefreshActive = true;
    let refreshInterval = null;
    const REFRESH_RATE_MS = 5000;
    
    let chartInstance = null;
    let currentChartTab = 'hitrate'; // 'hitrate' | 'reward' | 'qtable'
    let historyData = [];
    
    // Select Polling Elements
    const refreshToggle = document.getElementById('refresh-toggle'); // Note: if not present in new HTML, we guard it
    const lastUpdateLabel = document.getElementById('last-update');
    
    const valEpisode = document.getElementById('val-episode');
    const valHitrate = document.getElementById('val-hitrate');
    const valEpsilon = document.getElementById('val-epsilon');
    const valQtable = document.getElementById('val-qtable');
    const valReward = document.getElementById('val-reward');

    // Safe Polling Toggle Event Handler
    if (refreshToggle) {
        const refreshText = refreshToggle.querySelector('.toggle-text');
        refreshToggle.addEventListener('click', () => {
            autoRefreshActive = !autoRefreshActive;
            if (autoRefreshActive) {
                refreshToggle.classList.add('active');
                if (refreshText) refreshText.textContent = 'ON';
                startAutoRefresh();
                fetchData();
            } else {
                refreshToggle.classList.remove('active');
                if (refreshText) refreshText.textContent = 'OFF';
                stopAutoRefresh();
            }
        });
    }

    // Chart Tabs Switcher
    const chartTabBtns = document.querySelectorAll('.btn-chart-tab');
    chartTabBtns.forEach(btn => {
        btn.addEventListener('click', (e) => {
            chartTabBtns.forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            currentChartTab = btn.dataset.chart;
            updateChart();
        });
    });

    // Start Auto Refresh Loop
    function startAutoRefresh() {
        stopAutoRefresh();
        refreshInterval = setInterval(fetchData, REFRESH_RATE_MS);
    }

    // Stop Auto Refresh Loop
    function stopAutoRefresh() {
        if (refreshInterval) {
            clearInterval(refreshInterval);
            refreshInterval = null;
        }
    }

    // Fetch Stats & History from Python run
    async function fetchData() {
        const timestamp = new Date().getTime();
        try {
            const statsRes = await fetch(`data/stats.json?t=${timestamp}`);
            if (!statsRes.ok) throw new Error('stats.json not found');
            const stats = await statsRes.json();
            
            const historyRes = await fetch(`data/history.json?t=${timestamp}`);
            if (historyRes.ok) {
                historyData = await historyRes.json();
            }

            updateTelemetry(stats);
            updateGallery(stats);
            updateChart();
            
            const now = new Date();
            if (lastUpdateLabel) {
                lastUpdateLabel.textContent = `Sync: ${now.toLocaleTimeString()}`;
                lastUpdateLabel.style.color = 'var(--accent-blue)';
            }
        } catch (err) {
            console.warn('Waiting for local python training outputs...', err.message);
            if (lastUpdateLabel) {
                lastUpdateLabel.textContent = 'Standby';
                lastUpdateLabel.style.color = 'var(--text-dim)';
            }
        }
    }

    // Update Telemetry Panel
    function updateTelemetry(stats) {
        const episode = stats.episode || 0;
        const hitrate = stats.hit_rate || 0.0;
        const epsilon = stats.epsilon !== undefined ? stats.epsilon : 1.0;
        const qtable = stats.qtable_size || 0;
        const reward = stats.average_reward || 0.0;

        if (valEpisode) valEpisode.textContent = episode.toLocaleString();
        if (valHitrate) valHitrate.textContent = `${hitrate.toFixed(1)}%`;
        if (valEpsilon) valEpsilon.textContent = epsilon.toFixed(3);
        if (valQtable) valQtable.textContent = qtable.toLocaleString();
        if (valReward) valReward.textContent = (reward >= 0 ? '+' : '') + reward.toFixed(2);
    }

    // Update Replays Gallery
    function updateGallery(stats) {
        const episode = stats.episode || 0;
        const cards = document.querySelectorAll('.timeline-card');
        
        cards.forEach(card => {
            const cardEp = parseInt(card.dataset.episode, 10);
            const skeleton = card.querySelector('.replay-skeleton');
            const img = card.querySelector('.replay-gif');
            
            if (img && skeleton) {
                if (episode >= cardEp) {
                    if (!img.src || img.src.includes('undefined') || img.src === '') {
                        const src = img.dataset.src;
                        img.src = `${src}?t=${new Date().getTime()}`;
                        img.onload = () => {
                            skeleton.style.display = 'none';
                            img.style.display = 'block';
                        };
                        img.onerror = () => {
                            const text = skeleton.querySelector('p');
                            if (text) text.textContent = 'Loading...';
                        };
                    } else {
                        skeleton.style.display = 'none';
                        img.style.display = 'block';
                    }
                } else {
                    skeleton.style.display = 'flex';
                    img.style.display = 'none';
                    const text = skeleton.querySelector('p');
                    if (text) text.textContent = `Unlocks at Episode ${cardEp}`;
                }
            }
        });
    }

    // Initialize or Update Chart.js
    function updateChart() {
        if (!historyData || historyData.length === 0) return;
        
        const labels = historyData.map(d => d.episode);
        let datasetLabel = '';
        let datasetData = [];
        let borderColor = '';
        let backgroundColor = '';
        
        switch (currentChartTab) {
            case 'hitrate':
                datasetLabel = 'Hit Rate (%)';
                datasetData = historyData.map(d => d.hit_rate);
                borderColor = '#fafafa';
                backgroundColor = 'rgba(250, 250, 250, 0.04)';
                break;
            case 'reward':
                datasetLabel = 'Average Reward';
                datasetData = historyData.map(d => d.average_reward);
                borderColor = '#3b82f6';
                backgroundColor = 'rgba(59, 130, 246, 0.04)';
                break;
            case 'qtable':
                datasetLabel = 'Q-Table Size';
                datasetData = historyData.map(d => d.qtable_size);
                borderColor = '#a1a1aa';
                backgroundColor = 'rgba(161, 161, 170, 0.04)';
                break;
        }
        
        const canvasEl = document.getElementById('telemetryChart');
        if (!canvasEl) return;
        const ctx = canvasEl.getContext('2d');
        
        if (chartInstance) {
            chartInstance.data.labels = labels;
            chartInstance.data.datasets[0].label = datasetLabel;
            chartInstance.data.datasets[0].data = datasetData;
            chartInstance.data.datasets[0].borderColor = borderColor;
            chartInstance.data.datasets[0].backgroundColor = backgroundColor;
            chartInstance.update();
        } else {
            chartInstance = new Chart(ctx, {
                type: 'line',
                data: {
                    labels: labels,
                    datasets: [{
                        label: datasetLabel,
                        data: datasetData,
                        borderColor: borderColor,
                        backgroundColor: backgroundColor,
                        borderWidth: 1.5,
                        tension: 0.15,
                        fill: true,
                        pointBackgroundColor: borderColor,
                        pointBorderColor: '#09090b',
                        pointHoverRadius: 4,
                        pointRadius: 2
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                        legend: {
                            display: true,
                            labels: {
                                color: '#a1a1aa',
                                font: { family: 'JetBrains Mono', size: 11 }
                            }
                        },
                        tooltip: {
                            mode: 'index',
                            intersect: false,
                            backgroundColor: '#18181b',
                            titleColor: '#fafafa',
                            bodyColor: '#fafafa',
                            borderColor: '#27272a',
                            borderWidth: 1,
                            titleFont: { family: 'JetBrains Mono' },
                            bodyFont: { family: 'JetBrains Mono' }
                        }
                    },
                    scales: {
                        x: {
                            grid: { color: '#27272a', drawBorder: false },
                            ticks: {
                                color: '#71717a',
                                font: { family: 'JetBrains Mono', size: 10 }
                            }
                        },
                        y: {
                            grid: { color: '#27272a', drawBorder: false },
                            ticks: {
                                color: '#71717a',
                                font: { family: 'JetBrains Mono', size: 10 }
                            }
                        }
                    }
                }
            });
        }
    }


    // ─── LOCAL IN-BROWSER Q-LEARNING ENVIRONMENT ───
    const simCanvas = document.getElementById('pongSimulator');
    if (simCanvas) {
        const simCtx = simCanvas.getContext('2d');
        
        // DOM Controls
        const btnPauseResume = document.getElementById('btn-pause-resume');
        const btnClearQTable = document.getElementById('btn-clear-qtable');
        const btnResetAgent = document.getElementById('btn-reset-agent');
        const simSpeedSelect = document.getElementById('sim-speed-select');
        const toggleGrid = document.getElementById('toggle-grid');
        
        // Brain displays
        const lblStateVector = document.getElementById('sim-state-vector');
        const lblAction = document.getElementById('sim-action');
        const lblReward = document.getElementById('sim-reward');
        const lblEpsilon = document.getElementById('sim-epsilon');
        
        // Active Q-value cells
        const qRowStay = document.getElementById('q-row-stay');
        const qRowUp = document.getElementById('q-row-up');
        const qRowDown = document.getElementById('q-row-down');
        
        // Local stats labels
        const lblLocalEpisode = document.getElementById('local-episode');
        const lblLocalHitrate = document.getElementById('local-hitrate');
        const lblLocalStates = document.getElementById('local-states-visited');
        const lblLocalReward = document.getElementById('local-total-reward');

        // Physics Entities
        let ball = { x: simCanvas.width / 2, y: simCanvas.height / 2, vx: 4.5, vy: 2.2, radius: 7 };
        let paddle = { x: simCanvas.width - 25, y: simCanvas.height / 2 - 27.5, width: 8, height: 55, speed: 7 };
        
        // Simulation State flags
        let isPaused = false;
        let cumulativeReward = 0.0;
        let lastFrameReward = 0.0;
        let chosenAction = 0; // 0 = STAY, 1 = UP, 2 = DOWN
        
        // Hit rate history (rolling queue of last 50 reflections)
        let hitHistory = [];
        
        // Persistent statistics
        let localEpisode = parseInt(localStorage.getItem('pong_rl_episode')) || 1;
        cumulativeReward = parseFloat(localStorage.getItem('pong_rl_cumreward')) || 0.0;
        
        // Real Q-Table Object loading/init
        let qTable = {};
        try {
            const savedQTable = localStorage.getItem('pong_rl_qtable');
            if (savedQTable) {
                qTable = JSON.parse(savedQTable);
            }
        } catch (e) {
            console.error('Failed to parse saved Q-Table, initializing empty.', e);
            qTable = {};
        }

        // Q-Learning Hyperparameters
        const alpha = 0.1;   // Learning rate
        const gamma = 0.95;  // Discount factor
        
        // Calculate epsilon based on decay curve from current episode
        function getEpsilon() {
            return Math.max(0.02, 0.6 * Math.pow(0.995, localEpisode - 1));
        }

        // Initialize state arrays if not exists
        function getQArray(stateKey) {
            if (!qTable[stateKey]) {
                qTable[stateKey] = [0.0, 0.0, 0.0]; // STAY, UP, DOWN
            }
            return qTable[stateKey];
        }

        // State formulation: (ball_y_zone, paddle_y_zone, vy_direction)
        function getStateKey() {
            // Discretize vertical workspace height (385 pixels) into 10 bins
            const ballYZone = Math.min(9, Math.max(0, Math.floor(ball.y / (simCanvas.height / 10))));
            
            // Discretize paddle center height into 10 bins
            const paddleCenterY = paddle.y + paddle.height / 2;
            const paddleYZone = Math.min(9, Math.max(0, Math.floor(paddleCenterY / (simCanvas.height / 10))));
            
            // Vertical direction: 0 = UP, 1 = DOWN, 2 = FLAT
            let vyDir = 2;
            if (ball.vy < -0.5) vyDir = 0;
            else if (ball.vy > 0.5) vyDir = 1;
            
            return `${ballYZone}_${paddleYZone}_${vyDir}`;
        }

        // Action Selection (Epsilon-greedy)
        function selectAction(stateKey) {
            const eps = getEpsilon();
            const qArr = getQArray(stateKey);
            
            if (Math.random() < eps) {
                // Explore: random action
                return Math.floor(Math.random() * 3);
            } else {
                // Exploit: maximize Q-value
                let maxIdx = 0;
                let maxVal = qArr[0];
                for (let i = 1; i < 3; i++) {
                    if (qArr[i] > maxVal) {
                        maxVal = qArr[i];
                        maxIdx = i;
                    }
                }
                return maxIdx;
            }
        }

        // Physics step execution and Reward return
        function stepPhysics(action) {
            let reward = 0.0;
            let hitOccurred = false;
            let missOccurred = false;

            // 1. Apply paddle action
            if (action === 1) { // UP
                paddle.y = Math.max(0, paddle.y - paddle.speed);
            } else if (action === 2) { // DOWN
                paddle.y = Math.min(simCanvas.height - paddle.height, paddle.y + paddle.speed);
            }

            // 2. Update ball coordinates
            ball.x += ball.vx;
            ball.y += ball.vy;

            // 3. Wall boundaries collision
            if (ball.y - ball.radius <= 0) {
                ball.y = ball.radius;
                ball.vy = -ball.vy;
            } else if (ball.y + ball.radius >= simCanvas.height) {
                ball.y = simCanvas.height - ball.radius;
                ball.vy = -ball.vy;
            }

            // Left wall bounce (no penalty, bounces back to right)
            if (ball.x - ball.radius <= 0) {
                ball.x = ball.radius;
                ball.vx = -ball.vx;
            }

            // 4. Paddle collision detection
            if (ball.vx > 0 &&
                ball.x + ball.radius >= paddle.x &&
                ball.x - ball.radius <= paddle.x + paddle.width &&
                ball.y >= paddle.y &&
                ball.y <= paddle.y + paddle.height) {
                
                // Reposition ball
                ball.x = paddle.x - ball.radius;
                ball.vx = -ball.vx;

                // Add minor velocity variation to prevent repeating state loops
                ball.vy += (Math.random() - 0.5) * 1.5;
                const sign = Math.sign(ball.vy) || 1;
                ball.vy = sign * Math.max(1.5, Math.min(5.5, Math.abs(ball.vy)));

                reward = 1.0;
                hitOccurred = true;
            } 
            // 5. Pass/Miss detection
            else if (ball.x - ball.radius > simCanvas.width) {
                reward = -10.0;
                missOccurred = true;
            }

            return { reward, hitOccurred, missOccurred };
        }

        // Q-table Persistence Save
        function persistState() {
            localStorage.setItem('pong_rl_qtable', JSON.stringify(qTable));
            localStorage.setItem('pong_rl_episode', localEpisode.toString());
            localStorage.setItem('pong_rl_cumreward', cumulativeReward.toString());
        }

        // Single Learning loop update
        function runLearningCycle() {
            // A. Get current state (s)
            const sKey = getStateKey();
            
            // B. Choose action (a)
            chosenAction = selectAction(sKey);

            // C. Execute step and receive reward (r)
            const result = stepPhysics(chosenAction);
            lastFrameReward = result.reward;
            cumulativeReward += result.reward;

            // D. Get next state (s')
            const nextSKey = getStateKey();

            // E. Perform Bellman Optimality Temporal Difference Update
            const qArrCurrent = getQArray(sKey);
            const qArrNext = getQArray(nextSKey);
            
            const maxNextQ = Math.max(qArrNext[0], qArrNext[1], qArrNext[2]);
            
            // Bellman formula
            qArrCurrent[chosenAction] = qArrCurrent[chosenAction] + alpha * (result.reward + gamma * maxNextQ - qArrCurrent[chosenAction]);

            // F. Handle Episode End resets and logs
            if (result.hitOccurred) {
                hitHistory.push(1);
                if (hitHistory.length > 50) hitHistory.shift();
            }

            if (result.missOccurred) {
                hitHistory.push(0);
                if (hitHistory.length > 50) hitHistory.shift();

                localEpisode++;
                persistState();

                // Reset Ball to center
                ball.x = simCanvas.width / 2;
                ball.y = simCanvas.height / 2;
                ball.vx = Math.random() > 0.5 ? 4.5 : -4.5;
                ball.vy = Math.random() > 0.5 ? 2.2 : -2.2;
            }
        }

        // Dynamic rendering loop
        function drawSimulationFrame() {
            // Clear Frame
            simCtx.fillStyle = '#050507';
            simCtx.fillRect(0, 0, simCanvas.width, simCanvas.height);

            const showGrid = toggleGrid ? toggleGrid.checked : true;
            
            // Current coordinates discretized
            const ballYZone = Math.min(9, Math.max(0, Math.floor(ball.y / (simCanvas.height / 10))));
            const paddleCenterY = paddle.y + paddle.height / 2;
            const paddleYZone = Math.min(9, Math.max(0, Math.floor(paddleCenterY / (simCanvas.height / 10))));
            
            const sKey = getStateKey();
            const qArr = getQArray(sKey);

            // 1. Draw Grid Overlay
            if (showGrid) {
                // Highlight rows
                simCtx.fillStyle = 'rgba(255, 255, 255, 0.03)';
                simCtx.fillRect(0, ballYZone * (simCanvas.height / 10), simCanvas.width, simCanvas.height / 10);
                
                simCtx.fillStyle = 'rgba(59, 130, 246, 0.04)';
                simCtx.fillRect(0, paddleYZone * (simCanvas.height / 10), simCanvas.width, simCanvas.height / 10);

                // Draw grid division borders (horizontal separators)
                simCtx.strokeStyle = '#1d1d22';
                simCtx.lineWidth = 1;
                for (let i = 1; i < 10; i++) {
                    simCtx.beginPath();
                    simCtx.moveTo(0, i * (simCanvas.height / 10));
                    simCtx.lineTo(simCanvas.width, i * (simCanvas.height / 10));
                    simCtx.stroke();
                }

                // Draw solid indicator margins on active zones
                simCtx.strokeStyle = 'rgba(255, 255, 255, 0.15)';
                simCtx.strokeRect(0, ballYZone * (simCanvas.height / 10), simCanvas.width, simCanvas.height / 10);
                
                simCtx.strokeStyle = 'rgba(59, 130, 246, 0.25)';
                simCtx.strokeRect(simCanvas.width - 15, paddleYZone * (simCanvas.height / 10), 15, simCanvas.height / 10);
            }

            // 2. Draw Entities
            // Ball
            simCtx.fillStyle = '#fafafa';
            simCtx.beginPath();
            simCtx.arc(ball.x, ball.y, ball.radius, 0, Math.PI * 2);
            simCtx.fill();
            simCtx.strokeStyle = '#09090b';
            simCtx.lineWidth = 1.5;
            simCtx.stroke();

            // Paddle
            simCtx.fillStyle = '#a1a1aa';
            simCtx.fillRect(paddle.x, paddle.y, paddle.width, paddle.height);
            simCtx.strokeStyle = '#fafafa';
            simCtx.lineWidth = 1;
            simCtx.strokeRect(paddle.x, paddle.y, paddle.width, paddle.height);

            // Vector arrow line ahead of ball
            simCtx.strokeStyle = 'rgba(250, 250, 250, 0.35)';
            simCtx.lineWidth = 1.5;
            simCtx.beginPath();
            simCtx.moveTo(ball.x, ball.y);
            const angle = Math.atan2(ball.vy, ball.vx);
            simCtx.lineTo(ball.x + Math.cos(angle) * 16, ball.y + Math.sin(angle) * 16);
            simCtx.stroke();

            // 3. Update Brain Panel Displays
            let vyDirText = 'FLAT';
            if (ball.vy < -0.5) vyDirText = 'UP';
            else if (ball.vy > 0.5) vyDirText = 'DOWN';
            
            if (lblStateVector) lblStateVector.textContent = `(${ballYZone}, ${paddleYZone}, ${vyDirText})`;
            
            const actionsTextMap = ['STAY', 'UP', 'DOWN'];
            if (lblAction) lblAction.textContent = actionsTextMap[chosenAction];
            if (lblReward) {
                lblReward.textContent = (lastFrameReward >= 0 ? '+' : '') + lastFrameReward.toFixed(2);
                if (lastFrameReward > 0) lblReward.style.color = '#3b82f6';
                else if (lastFrameReward < 0) lblReward.style.color = '#ef4444';
                else lblReward.style.color = 'var(--text-primary)';
            }
            if (lblEpsilon) lblEpsilon.textContent = getEpsilon().toFixed(3);

            // 4. Update Q-values table
            const qStay = qArr[0];
            const qUp = qArr[1];
            const qDown = qArr[2];
            
            const cellStay = qRowStay.querySelector('.q-val-cell');
            const cellUp = qRowUp.querySelector('.q-val-cell');
            const cellDown = qRowDown.querySelector('.q-val-cell');
            
            if (cellStay) cellStay.textContent = qStay.toFixed(4);
            if (cellUp) cellUp.textContent = qUp.toFixed(4);
            if (cellDown) cellDown.textContent = qDown.toFixed(4);

            // Highlight max value
            qRowStay.classList.remove('preferred');
            qRowUp.classList.remove('preferred');
            qRowDown.classList.remove('preferred');
            
            let preferredActionIdx = 0;
            let maxVal = qStay;
            if (qUp > maxVal) { maxVal = qUp; preferredActionIdx = 1; }
            if (qDown > maxVal) { maxVal = qDown; preferredActionIdx = 2; }
            
            // Only highlight preference if there has been some learning (values not all zero)
            if (qStay !== 0.0 || qUp !== 0.0 || qDown !== 0.0) {
                if (preferredActionIdx === 0) qRowStay.classList.add('preferred');
                else if (preferredActionIdx === 1) qRowUp.classList.add('preferred');
                else qRowDown.classList.add('preferred');
            }

            // Draw preference bar lengths
            const maxValAbs = Math.max(Math.abs(qStay), Math.abs(qUp), Math.abs(qDown), 0.001);
            
            // Softmax or simple ratio to show relative selection preference
            let minQ = Math.min(qStay, qUp, qDown);
            let maxQ = Math.max(qStay, qUp, qDown);
            let range = maxQ - minQ || 1;
            
            const pctStay = qStay === 0.0 && qUp === 0.0 && qDown === 0.0 ? 0 : ((qStay - minQ) / range) * 100;
            const pctUp = qStay === 0.0 && qUp === 0.0 && qDown === 0.0 ? 0 : ((qUp - minQ) / range) * 100;
            const pctDown = qStay === 0.0 && qUp === 0.0 && qDown === 0.0 ? 0 : ((qDown - minQ) / range) * 100;

            const fillStay = qRowStay.querySelector('.pref-fill');
            const fillUp = qRowUp.querySelector('.pref-fill');
            const fillDown = qRowDown.querySelector('.pref-fill');
            
            if (fillStay) fillStay.style.width = `${pctStay}%`;
            if (fillUp) fillUp.style.width = `${pctUp}%`;
            if (fillDown) fillDown.style.width = `${pctDown}%`;

            // 5. Update Statistics Panel
            if (lblLocalEpisode) lblLocalEpisode.textContent = localEpisode.toLocaleString();
            
            if (lblLocalHitrate) {
                if (hitHistory.length === 0) {
                    lblLocalHitrate.textContent = '0.0%';
                } else {
                    const hits = hitHistory.reduce((sum, current) => sum + current, 0);
                    const rate = (hits / hitHistory.length) * 100;
                    lblLocalHitrate.textContent = `${rate.toFixed(1)}%`;
                }
            }

            if (lblLocalStates) {
                const visitedCount = Object.keys(qTable).length;
                lblLocalStates.textContent = `${visitedCount} / 300`;
            }

            if (lblLocalReward) {
                lblLocalReward.textContent = (cumulativeReward >= 0 ? '+' : '') + cumulativeReward.toFixed(1);
            }
        }

        // Main animation logic coordinating steps per frame
        function simLoop() {
            if (!isPaused) {
                // Read speed multiplier
                const speed = simSpeedSelect ? parseInt(simSpeedSelect.value, 10) : 1;
                
                // Run physics and learning updates headless N times
                for (let i = 0; i < speed; i++) {
                    runLearningCycle();
                }
            }
            
            // Draw once per screen refresh frame
            drawSimulationFrame();
            requestAnimationFrame(simLoop);
        }

        // Pause/Resume listener
        if (btnPauseResume) {
            btnPauseResume.addEventListener('click', () => {
                isPaused = !isPaused;
                btnPauseResume.textContent = isPaused ? 'Resume Simulation' : 'Pause Simulation';
                btnPauseResume.classList.toggle('btn-primary', isPaused);
                btnPauseResume.classList.toggle('btn-secondary', !isPaused);
            });
        }

        // Clear Q-table listener
        if (btnClearQTable) {
            btnClearQTable.addEventListener('click', () => {
                if (confirm('Clear the browser agent\'s Q-table? Learning will start from scratch.')) {
                    qTable = {};
                    localEpisode = 1;
                    cumulativeReward = 0.0;
                    lastFrameReward = 0.0;
                    hitHistory = [];
                    persistState();
                    console.log('Q-Table cleared.');
                }
            });
        }

        // Reset Statistics listener
        if (btnResetAgent) {
            btnResetAgent.addEventListener('click', () => {
                localEpisode = 1;
                cumulativeReward = 0.0;
                lastFrameReward = 0.0;
                hitHistory = [];
                persistState();
                console.log('Statistics counter reset.');
            });
        }

        // Start Loop
        simLoop();
    }


    // Initial Poll load
    fetchData();
    startAutoRefresh();
});
