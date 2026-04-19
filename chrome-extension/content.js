/**
 * AWE Chrome Extension - Content Script
 * =======================================
 * Runs on web pages and provides page data to the extension popup.
 */

// Listen for messages from popup
chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
    if (message.type === 'GET_PAGE_DATA') {
        const pageData = {
            url: window.location.href,
            title: document.title,
            html: document.documentElement.outerHTML,
            text: document.body?.innerText?.substring(0, 50000) || '',
            meta: getMetaData(),
            links: getPageLinks(),
            headings: getHeadings(),
        };
        sendResponse(pageData);
    }
    return true; // Async response
});

function getMetaData() {
    const meta = {};
    document.querySelectorAll('meta').forEach(el => {
        const name = el.getAttribute('name') || el.getAttribute('property');
        const content = el.getAttribute('content');
        if (name && content) {
            meta[name] = content;
        }
    });
    return meta;
}

function getPageLinks() {
    const links = [];
    document.querySelectorAll('a[href]').forEach(a => {
        links.push({
            url: a.href,
            text: a.innerText?.trim()?.substring(0, 80) || '',
        });
    });
    return links.slice(0, 100);
}

function getHeadings() {
    const headings = [];
    document.querySelectorAll('h1, h2, h3, h4, h5, h6').forEach(h => {
        headings.push({
            level: parseInt(h.tagName[1]),
            text: h.innerText?.trim()?.substring(0, 100) || '',
        });
    });
    return headings;
}
