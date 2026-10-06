"""Visible mouse cursor injected into recorded pages (local and MCP browser engines)."""

CURSOR_OVERLAY_SCRIPT = r"""
(() => {
  if (window.__thesisCursorOverlayInitialized) return;
  window.__thesisCursorOverlayInitialized = true;

  const init = () => {
    if (window.__thesisCursorOverlay) return;
    const cursor = document.createElement('div');
    cursor.id = '__thesis_cursor_overlay';
    Object.assign(cursor.style, {
      position: 'fixed',
      width: '16px',
      height: '16px',
      border: '2px solid #ff2d55',
      borderRadius: '50%',
      background: 'rgba(255,45,85,0.22)',
      boxShadow: '0 0 10px rgba(255,45,85,0.75)',
      pointerEvents: 'none',
      zIndex: '2147483647',
      transform: 'translate(-50%, -50%)',
      transition: 'top 70ms linear, left 70ms linear'
    });
    document.body.appendChild(cursor);

    window.addEventListener('mousemove', (event) => {
      cursor.style.left = `${event.clientX}px`;
      cursor.style.top = `${event.clientY}px`;
    }, true);

    window.__thesisCursorOverlay = cursor;
  };

  if (document.readyState === 'loading') {
    window.addEventListener('DOMContentLoaded', init, { once: true });
  } else {
    init();
  }
})();
"""
