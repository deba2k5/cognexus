/**
 * AWE Chrome Extension - Background Service Worker
 * ==================================================
 * Manages API communication and extension lifecycle.
 */

const API_URL = 'http://localhost:8000';

// On install, check API status
chrome.runtime.onInstalled.addListener(() => {
    console.log('AWE Extension installed');
    checkHealth();
});

async function checkHealth() {
    try {
        const res = await fetch(`${API_URL}/health`);
        const data = await res.json();
        console.log('AWE API Status:', data.status);
        await chrome.storage.local.set({ apiStatus: 'online', apiData: data });
    } catch {
        console.log('AWE API is offline');
        await chrome.storage.local.set({ apiStatus: 'offline' });
    }
}

// Periodic health check every 30 seconds
setInterval(checkHealth, 30000);

// Handle messages from popup
chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
    if (message.type === 'API_REQUEST') {
        handleApiRequest(message)
            .then(sendResponse)
            .catch(err => sendResponse({ error: err.message }));
        return true; // Async
    }

    if (message.type === 'CHECK_HEALTH') {
        checkHealth().then(() => sendResponse({ ok: true }));
        return true;
    }
});

async function handleApiRequest({ endpoint, method = 'POST', body }) {
    const response = await fetch(`${API_URL}${endpoint}`, {
        method,
        headers: { 'Content-Type': 'application/json' },
        body: body ? JSON.stringify(body) : undefined,
    });

    if (!response.ok) {
        throw new Error(`API error: ${response.statusText}`);
    }

    return response.json();
}
