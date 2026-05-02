/* ============================================================
   MoodSense AI – app.js
   ============================================================ */

/* ── Neural Network Canvas Animation ─────────────────────── */
(function initNeuralCanvas() {
    const canvas = document.getElementById('neural-canvas');
    if (!canvas) return;
    const ctx = canvas.getContext('2d');

    let W, H, nodes = [], edges = [], animFrame;
    const NODE_COUNT = window.innerWidth < 600 ? 28 : 50;
    const EDGE_DIST  = window.innerWidth < 600 ? 130 : 160;
    let themeColor = { r: 91, g: 127, b: 255 };   // default blue-violet
    let targetColor = { ...themeColor };

    function resize() {
        W = canvas.width  = window.innerWidth;
        H = canvas.height = window.innerHeight;
    }

    function spawnNodes() {
        nodes = [];
        for (let i = 0; i < NODE_COUNT; i++) {
            nodes.push({
                x: Math.random() * W,
                y: Math.random() * H,
                vx: (Math.random() - 0.5) * 0.35,
                vy: (Math.random() - 0.5) * 0.35,
                r: Math.random() * 2.2 + 1,
                pulse: Math.random() * Math.PI * 2,
            });
        }
    }

    function lerpColor() {
        const s = 0.04;
        themeColor.r += (targetColor.r - themeColor.r) * s;
        themeColor.g += (targetColor.g - themeColor.g) * s;
        themeColor.b += (targetColor.b - themeColor.b) * s;
    }

    function colorStr(a) {
        const { r, g, b } = themeColor;
        return `rgba(${r|0},${g|0},${b|0},${a})`;
    }

    function draw() {
        lerpColor();
        ctx.clearRect(0, 0, W, H);

        // Update positions
        nodes.forEach(n => {
            n.x += n.vx; n.y += n.vy;
            n.pulse += 0.02;
            if (n.x < 0 || n.x > W) n.vx *= -1;
            if (n.y < 0 || n.y > H) n.vy *= -1;
        });

        // Draw edges
        for (let i = 0; i < nodes.length; i++) {
            for (let j = i + 1; j < nodes.length; j++) {
                const a = nodes[i], b = nodes[j];
                const dx = a.x - b.x, dy = a.y - b.y;
                const dist = Math.sqrt(dx * dx + dy * dy);
                if (dist < EDGE_DIST) {
                    const alpha = (1 - dist / EDGE_DIST) * 0.22;
                    ctx.beginPath();
                    ctx.strokeStyle = colorStr(alpha);
                    ctx.lineWidth = 0.8;
                    ctx.moveTo(a.x, a.y);
                    ctx.lineTo(b.x, b.y);
                    ctx.stroke();
                }
            }
        }

        // Draw nodes
        nodes.forEach(n => {
            const pulsedR = n.r + Math.sin(n.pulse) * 0.6;
            ctx.beginPath();
            ctx.arc(n.x, n.y, pulsedR, 0, Math.PI * 2);
            ctx.fillStyle = colorStr(0.55);
            ctx.fill();

            // Soft halo
            const grad = ctx.createRadialGradient(n.x, n.y, 0, n.x, n.y, pulsedR * 4);
            grad.addColorStop(0, colorStr(0.12));
            grad.addColorStop(1, colorStr(0));
            ctx.beginPath();
            ctx.arc(n.x, n.y, pulsedR * 4, 0, Math.PI * 2);
            ctx.fillStyle = grad;
            ctx.fill();
        });

        animFrame = requestAnimationFrame(draw);
    }

    // Expose setter to update color on mood change
    window.setNeuralColor = function(r, g, b) {
        targetColor = { r, g, b };
    };

    window.addEventListener('resize', () => { resize(); spawnNodes(); });
    resize();
    spawnNodes();
    draw();
})();


/* ── Main Application ────────────────────────────────────── */
document.addEventListener('DOMContentLoaded', () => {
    const messageInput  = document.getElementById('message-input');
    const predictBtn    = document.getElementById('predict-btn');
    const errorText     = document.getElementById('error-text');
    const confidenceVal = document.getElementById('confidence-value');
    const confidenceBar = document.getElementById('confidence-bar');
    const charCount     = document.getElementById('char-count');
    const moodDesc      = document.getElementById('mood-desc');
    const moodValue     = document.getElementById('mood-value');
    const emptyState    = document.getElementById('empty-state');
    const resultDisplay = document.getElementById('result-display');

    /* Mood data ──────────────────────────────────────────── */
    const moodMap = {
        joy: {
            emoji: '😊', theme: 'theme-happy',
            desc: 'Your message radiates warmth and pure positivity! ✨',
            neural: [255, 183, 0],
        },
        love: {
            emoji: '❤️', theme: 'theme-love',
            desc: 'Deep affection and connection sensed in your words. ❤️',
            neural: [255, 107, 157],
        },
        sadness: {
            emoji: '😢', theme: 'theme-sad',
            desc: 'A touch of melancholy detected. It\'s okay to feel. 💙',
            neural: [34, 184, 240],
        },
        anger: {
            emoji: '😡', theme: 'theme-angry',
            desc: 'High frustration levels detected. Take a deep breath. 🔥',
            neural: [255, 51, 51],
        },
        fear: {
            emoji: '😨', theme: 'theme-fear',
            desc: 'Anxiety or apprehension is present in your words. 🌑',
            neural: [155, 91, 255],
        },
        surprise: {
            emoji: '😯', theme: 'theme-surprise',
            desc: 'Wow! A wave of astonishment detected! 🌟',
            neural: [250, 204, 21],
        },
        neutral: {
            emoji: '😐', theme: '',
            desc: 'A calm, balanced emotional state detected.',
            neural: [91, 127, 255],
        },
    };

    /* Helpers ───────────────────────────────────────────── */
    function clearTheme() {
        Object.values(moodMap).forEach(m => {
            if (m.theme) document.body.classList.remove(m.theme);
        });
    }
    function setTheme(themeClass, neural) {
        clearTheme();
        if (themeClass) document.body.classList.add(themeClass);
        if (window.setNeuralColor && neural) window.setNeuralColor(...neural);
    }

    /* Character counter ─────────────────────────────────── */
    messageInput.addEventListener('input', () => {
        const len = messageInput.value.length;
        charCount.textContent = `${len} / 500`;
        if (messageInput.value.trim()) {
            errorText.classList.remove('show');
        }
    });

    /* Sample chips ──────────────────────────────────────── */
    document.querySelectorAll('.sample-chip').forEach(chip => {
        chip.addEventListener('click', () => {
            messageInput.value = chip.dataset.text;
            charCount.textContent = `${chip.dataset.text.length} / 500`;
            errorText.classList.remove('show');
            messageInput.focus();
        });
    });

    /* Predict click ─────────────────────────────────────── */
    predictBtn.addEventListener('click', async () => {
        const text = messageInput.value.trim();
        if (!text) {
            errorText.classList.add('show');
            messageInput.focus();
            return;
        }
        errorText.classList.remove('show');
        predictBtn.classList.add('loading');
        predictBtn.disabled = true;

        try {
            // Using /analyze for both local ML and Groq AI results
            const response = await callBackendPrediction(text);
            displayResult(response.mood, response.confidence, response);
        } catch (err) {
            console.error('Prediction error:', err);
            alert('Could not reach the AI backend. Please ensure the server is running.');
        } finally {
            predictBtn.classList.remove('loading');
            predictBtn.disabled = false;
        }
    });

    /* Typewriter effect ─────────────────────────────────── */
    function typewriterEffect(text, element) {
        element.innerHTML = '';
        let i = 0;
        const speed = 25; // ms

        function type() {
            if (i < text.length) {
                element.innerHTML += text.charAt(i);
                i++;
                setTimeout(type, speed);
            }
        }
        type();
    }

    /* Display result ─────────────────────────────────────── */
    function displayResult(mood, confidence, response) {
        const key  = mood.toLowerCase();
        const data = moodMap[key] || moodMap.neutral;
        const pct  = (confidence * 100).toFixed(1);

        // Text
        moodValue.textContent     = mood.charAt(0).toUpperCase() + mood.slice(1);
        confidenceVal.textContent = `${pct}%`;
        if (moodDesc) moodDesc.textContent = data.desc;

        // Theme
        setTheme(data.theme, data.neural);

        // Emoji pop
        const emojiEl = document.getElementById('mood-emoji');
        if (emojiEl) {
            const clone = emojiEl.cloneNode(true);
            // Prioritize AI provided emoji, then fallback to moodMap, then default to '😐'
            clone.textContent = response.ai_emoji || data.emoji || '😐';
            emojiEl.parentNode.replaceChild(clone, emojiEl);
        }

        // Emoji glow colour
        const emojiGlow = document.getElementById('emoji-glow');
        if (emojiGlow) {
            emojiGlow.style.background = '';   // let CSS var handle it
        }

        // Confidence bar
        confidenceBar.style.width = '0%';
        setTimeout(() => { confidenceBar.style.width = `${pct}%`; }, 60);

        // AI Insight (Groq)
        const aiTextEl = document.getElementById('ai-analysis-text');
        if (aiTextEl && response.ai_analysis) {
            typewriterEffect(response.ai_analysis, aiTextEl);
        }

        // Analysis Breakdown
        const analysisTokens = document.getElementById('analysis-tokens');
        if (analysisTokens && response.clean_text) {
            analysisTokens.innerHTML = '';
            response.clean_text.split(' ').forEach(token => {
                const span = document.createElement('span');
                span.className = 'token-chip';
                if (token.startsWith('not_')) span.classList.add('token-neg');
                span.textContent = token;
                analysisTokens.appendChild(span);
            });
        }

        // Show result, hide empty state
        emptyState.style.opacity = '0';
        emptyState.style.pointerEvents = 'none';
        resultDisplay.classList.remove('hidden');
    }

    /* Real ML Backend Call ───────────────────────────────── */
    async function callBackendPrediction(text) {
        const res = await fetch('/analyze', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ text }),
        });
        if (!res.ok) throw new Error(`Backend error: ${res.status}`);
        return res.json();
    }
});

