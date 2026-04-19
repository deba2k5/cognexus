/**
 * AWE Chrome Extension - Popup Script
 * ====================================
 * Handles tab switching, API calls, and results rendering.
 */

const API_URL = 'http://localhost:8000';

// ============================
// Initialization
// ============================

document.addEventListener('DOMContentLoaded', () => {
    initTabs();
    getCurrentTab();
    checkApiStatus();

    document.getElementById('btnExtract').addEventListener('click', runExtraction);
    document.getElementById('btnSecurity').addEventListener('click', runSecurityScan);
    document.getElementById('btnStructure').addEventListener('click', runStructureAnalysis);
});

// ============================
// Tab Navigation
// ============================

function initTabs() {
    document.querySelectorAll('.tab-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
            document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
            btn.classList.add('active');
            document.getElementById(`tab-${btn.dataset.tab}`).classList.add('active');
        });
    });
}

// ============================
// Get Current Tab URL
// ============================

async function getCurrentTab() {
    try {
        const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
        if (tab?.url) {
            document.getElementById('currentUrl').textContent = tab.url;
            document.getElementById('currentUrl').title = tab.url;
        }
    } catch {
        document.getElementById('currentUrl').textContent = 'Unable to get current tab URL';
    }
}

function getUrl() {
    return document.getElementById('currentUrl').textContent;
}

// ============================
// API Status
// ============================

async function checkApiStatus() {
    const badge = document.getElementById('apiStatus');
    try {
        const res = await fetch(`${API_URL}/health`);
        if (res.ok) {
            badge.className = 'status-badge online';
            badge.querySelector('.status-text').textContent = 'ONLINE';
        } else {
            throw new Error('not ok');
        }
    } catch {
        badge.className = 'status-badge offline';
        badge.querySelector('.status-text').textContent = 'OFFLINE';
    }
}

// ============================
// Loading State
// ============================

function showLoading(text = 'Processing...') {
    const overlay = document.getElementById('loadingOverlay');
    overlay.querySelector('.loading-text').textContent = text;
    overlay.classList.add('active');
}

function hideLoading() {
    document.getElementById('loadingOverlay').classList.remove('active');
}

// ============================
// Extraction
// ============================

async function runExtraction() {
    const url = getUrl();
    if (!url || url === 'Loading...') return;

    showLoading('Extracting with ToT...');
    const container = document.getElementById('extractResults');

    try {
        const res = await fetch(`${API_URL}/demo`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ url, quick_mode: true }),
        });
        const data = await res.json();

        if (data.status === 'error') {
            container.innerHTML = renderError(data.message);
        } else {
            let html = '';

            // Stats
            const stats = data.stats || {};
            html += `
        <div class="stats-grid" style="grid-template-columns: repeat(3, 1fr);">
          <div class="stat-card stat-pass">
            <div class="stat-value">${stats.items_extracted || 0}</div>
            <div class="stat-label">Items</div>
          </div>
          <div class="stat-card stat-info">
            <div class="stat-value">${((stats.duration_ms || 0) / 1000).toFixed(1)}s</div>
            <div class="stat-label">Duration</div>
          </div>
          <div class="stat-card stat-warning">
            <div class="stat-value">${stats.tot_enabled ? 'ToT' : 'Live'}</div>
            <div class="stat-label">Mode</div>
          </div>
        </div>
      `;

            // Data items
            if (data.data && data.data.length > 0) {
                html += `<div class="section-label">Extracted Data (${data.data.length} items)</div>`;
                data.data.forEach(item => {
                    html += '<div class="data-item">';
                    Object.entries(item).forEach(([key, value]) => {
                        const val = Array.isArray(value) ? value.join(', ') : String(value);
                        html += `
              <div class="data-field">
                <span class="data-key">${key}:</span>
                <span class="data-value">${val}</span>
              </div>
            `;
                    });
                    html += '</div>';
                });
            }

            container.innerHTML = html;
        }
    } catch (err) {
        container.innerHTML = renderError(err.message);
    }

    hideLoading();
}

// ============================
// Security Scan
// ============================

async function runSecurityScan() {
    const url = getUrl();
    if (!url || url === 'Loading...') return;

    showLoading('Running Security Scan...');
    const container = document.getElementById('securityResults');

    try {
        const res = await fetch(`${API_URL}/analyze/security`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ url, check_links: true }),
        });
        const data = await res.json();

        if (data.error) {
            container.innerHTML = renderError(data.error);
        } else {
            let html = '';

            // Score card
            const gradeClass = `grade-${(data.grade || 'f').toLowerCase()}`;
            html += `
        <div class="score-card">
          <div class="grade-badge ${gradeClass}">${data.grade || '?'}</div>
          <div class="score-info">
            <h3>Score: ${data.score}/100</h3>
            <p>Scanned in ${data.duration_seconds}s</p>
          </div>
        </div>
      `;

            // Summary
            const s = data.summary || {};
            html += `
        <div class="stats-grid">
          <div class="stat-card stat-critical"><div class="stat-value">${s.critical || 0}</div><div class="stat-label">Critical</div></div>
          <div class="stat-card stat-warning"><div class="stat-value">${s.warning || 0}</div><div class="stat-label">Warning</div></div>
          <div class="stat-card stat-info"><div class="stat-value">${s.info || 0}</div><div class="stat-label">Info</div></div>
          <div class="stat-card stat-pass"><div class="stat-value">${s.pass || 0}</div><div class="stat-label">Passed</div></div>
        </div>
      `;

            // Issues
            if (data.checks) {
                data.checks.forEach(check => {
                    html += `<div class="section-label">${check.check}</div>`;
                    (check.items || []).forEach(item => {
                        const icon = { critical: '🔴', warning: '🟡', info: '🔵', pass: '🟢' }[item.severity] || '⚪';
                        html += `
              <div class="issue-item issue-${item.severity}">
                <span class="issue-icon">${icon}</span>
                <div>
                  <div class="issue-message">${item.message}</div>
                  <div class="issue-detail">${item.detail || ''}</div>
                </div>
              </div>
            `;
                    });
                });
            }

            container.innerHTML = html;
        }
    } catch (err) {
        container.innerHTML = renderError(err.message);
    }

    hideLoading();
}

// ============================
// Structure Analysis
// ============================

async function runStructureAnalysis() {
    const url = getUrl();
    if (!url || url === 'Loading...') return;

    showLoading('Analyzing Structure...');
    const container = document.getElementById('structureResults');

    try {
        const res = await fetch(`${API_URL}/analyze/structure`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ url }),
        });
        const data = await res.json();

        if (data.error) {
            container.innerHTML = renderError(data.error);
        } else {
            let html = '';

            // Score
            const gradeClass = `grade-${(data.grade || 'f').toLowerCase()}`;
            html += `
        <div class="score-card">
          <div class="grade-badge ${gradeClass}">${data.grade || '?'}</div>
          <div class="score-info">
            <h3>Score: ${data.score}/100</h3>
            <p>Analyzed in ${data.duration_seconds}s</p>
          </div>
        </div>
      `;

            // DOM Stats
            const dom = data.dom || {};
            const links = data.links || {};
            html += `
        <div class="stats-grid">
          <div class="stat-card stat-info"><div class="stat-value">${dom.total_elements || 0}</div><div class="stat-label">Elements</div></div>
          <div class="stat-card stat-warning"><div class="stat-value">${dom.max_depth || 0}</div><div class="stat-label">Depth</div></div>
          <div class="stat-card stat-pass"><div class="stat-value">${links.internal_links || 0}</div><div class="stat-label">Internal</div></div>
          <div class="stat-card stat-critical"><div class="stat-value">${links.external_links || 0}</div><div class="stat-label">External</div></div>
        </div>
      `;

            // Headings
            const headings = data.headings?.headings || [];
            if (headings.length > 0) {
                html += `<div class="section-label">Heading Hierarchy</div>`;
                headings.slice(0, 15).forEach(h => {
                    const indent = (h.level - 1) * 12;
                    html += `
            <div class="data-field" style="padding-left: ${indent}px; margin-bottom: 2px;">
              <span class="data-key" style="min-width: 30px; font-family: monospace; font-size: 10px;">${h.tag}</span>
              <span class="data-value">${h.text}</span>
            </div>
          `;
                });
            }

            // SEO Issues
            const seoIssues = data.metadata?.seo_issues || [];
            if (seoIssues.length > 0) {
                html += `<div class="section-label" style="margin-top: 10px;">SEO Issues</div>`;
                seoIssues.forEach(issue => {
                    const icon = { critical: '🔴', warning: '🟡', info: '🔵', pass: '🟢' }[issue.severity] || '⚪';
                    html += `
            <div class="issue-item issue-${issue.severity}">
              <span class="issue-icon">${icon}</span>
              <div><div class="issue-message">${issue.message}</div></div>
            </div>
          `;
                });
            }

            // Resources
            const res2 = data.resources || {};
            html += `
        <div class="section-label" style="margin-top: 10px;">Resources</div>
        <div class="stats-grid">
          <div class="stat-card stat-info"><div class="stat-value">${res2.scripts?.total || 0}</div><div class="stat-label">Scripts</div></div>
          <div class="stat-card stat-pass"><div class="stat-value">${res2.stylesheets?.total || 0}</div><div class="stat-label">Styles</div></div>
          <div class="stat-card stat-warning"><div class="stat-value">${res2.images?.total || 0}</div><div class="stat-label">Images</div></div>
          <div class="stat-card stat-critical"><div class="stat-value">${res2.images?.accessibility_score || '?'}</div><div class="stat-label">Alt %</div></div>
        </div>
      `;

            container.innerHTML = html;
        }
    } catch (err) {
        container.innerHTML = renderError(err.message);
    }

    hideLoading();
}

// ============================
// Helpers
// ============================

function renderError(message) {
    return `<div class="error-card"><strong>Error</strong>${message}</div>`;
}
