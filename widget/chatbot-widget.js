/**
 * Deligo Chatbot Widget
 *
 * A self-contained, vanilla-JS floating chat widget. Inject this script
 * into your Next.js app via a single <script> tag (e.g. in pages/_app.tsx,
 * app/layout.tsx, or a custom _document.tsx) — no other frontend code
 * changes are required.
 *
 * Example (Next.js App Router, app/layout.tsx):
 *
 *   <Script
 *     src="http://localhost:8005/widget/chatbot-widget.js"
 *     strategy="lazyOnload"
 *     data-api-base="http://localhost:8005"
 *   />
 *
 * Or plain HTML:
 *   <script
 *     src="http://localhost:8005/widget/chatbot-widget.js"
 *     data-api-base="http://localhost:8005"
 *     defer
 *   ></script>
 *
 * Optional: to pass the logged-in user's ID for order lookups, set
 * `window.DELIGO_CHAT_USER_ID = "<userId>"` before this script loads,
 * or update it later — the widget reads it on each message send.
 */

(function () {
  "use strict";

  // ---------------------------------------------------------------------
  // Configuration
  // ---------------------------------------------------------------------
  var currentScript = document.currentScript;
  var API_BASE =
    (currentScript && currentScript.getAttribute("data-api-base")) ||
    "http://localhost:8005";

  var SESSION_STORAGE_KEY = "deligo_chat_session_id";

  function getSessionId() {
    try {
      var existing = sessionStorage.getItem(SESSION_STORAGE_KEY);
      if (existing) return existing;
      var fresh =
        (window.crypto && crypto.randomUUID && crypto.randomUUID()) ||
        "sess_" + Math.random().toString(36).slice(2) + Date.now();
      sessionStorage.setItem(SESSION_STORAGE_KEY, fresh);
      return fresh;
    } catch (e) {
      return "sess_" + Date.now();
    }
  }

  function getUserId() {
    return window.DELIGO_CHAT_USER_ID || null;
  }

  // ---------------------------------------------------------------------
  // Styles
  // ---------------------------------------------------------------------
  var STYLE_ID = "deligo-chat-widget-styles";
  if (!document.getElementById(STYLE_ID)) {
    var style = document.createElement("style");
    style.id = STYLE_ID;
    style.textContent = [
      "#deligo-chat-launcher {",
      "  position: fixed; bottom: 20px; right: 20px; width: 56px; height: 56px;",
      "  border-radius: 50%; background: #1a73e8; color: #fff; border: none;",
      "  cursor: pointer; box-shadow: 0 4px 12px rgba(0,0,0,0.2); z-index: 999999;",
      "  display: flex; align-items: center; justify-content: center; font-size: 24px;",
      "  transition: transform 0.2s ease;",
      "}",
      "#deligo-chat-launcher:hover { transform: scale(1.05); }",
      "#deligo-chat-window {",
      "  position: fixed; bottom: 90px; right: 20px; width: 360px; max-width: 92vw;",
      "  height: 520px; max-height: 75vh; background: #fff; border-radius: 12px;",
      "  box-shadow: 0 8px 30px rgba(0,0,0,0.25); display: none; flex-direction: column;",
      "  overflow: hidden; z-index: 999999; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;",
      "}",
      "#deligo-chat-window.open { display: flex; }",
      "#deligo-chat-header {",
      "  background: #1a73e8; color: #fff; padding: 14px 16px; font-weight: 600;",
      "  display: flex; justify-content: space-between; align-items: center; font-size: 15px;",
      "}",
      "#deligo-chat-close { cursor: pointer; background: none; border: none; color: #fff; font-size: 20px; line-height: 1; }",
      "#deligo-chat-messages {",
      "  flex: 1; overflow-y: auto; padding: 12px; background: #f7f8fa;",
      "  display: flex; flex-direction: column; gap: 8px;",
      "}",
      ".deligo-msg { max-width: 80%; padding: 8px 12px; border-radius: 12px; font-size: 13.5px; line-height: 1.4; white-space: pre-wrap; word-wrap: break-word; }",
      ".deligo-msg-user { align-self: flex-end; background: #1a73e8; color: #fff; border-bottom-right-radius: 2px; }",
      ".deligo-msg-bot { align-self: flex-start; background: #fff; color: #222; border: 1px solid #e2e5ea; border-bottom-left-radius: 2px; }",
      ".deligo-msg-typing { align-self: flex-start; background: #fff; color: #999; border: 1px solid #e2e5ea; font-style: italic; padding: 8px 12px; border-radius: 12px; font-size: 13px; }",
      "#deligo-chat-input-row {",
      "  display: flex; gap: 8px; padding: 10px; border-top: 1px solid #e2e5ea; background: #fff;",
      "}",
      "#deligo-chat-input {",
      "  flex: 1; border: 1px solid #d6d9dd; border-radius: 20px; padding: 8px 14px;",
      "  font-size: 13.5px; outline: none;",
      "}",
      "#deligo-chat-input:focus { border-color: #1a73e8; }",
      "#deligo-chat-send {",
      "  background: #1a73e8; color: #fff; border: none; border-radius: 50%; width: 36px; height: 36px;",
      "  cursor: pointer; font-size: 16px; display: flex; align-items: center; justify-content: center; flex-shrink: 0;",
      "}",
      "#deligo-chat-send:disabled { opacity: 0.5; cursor: default; }",
      "#deligo-chat-footer-note { font-size: 10px; color: #aaa; text-align: center; padding: 4px 0 8px; }",
    ].join("\n");
    document.head.appendChild(style);
  }

  // ---------------------------------------------------------------------
  // DOM Construction
  // ---------------------------------------------------------------------
  function buildWidget() {
    var launcher = document.createElement("button");
    launcher.id = "deligo-chat-launcher";
    launcher.setAttribute("aria-label", "Open chat assistant");
    launcher.innerHTML = "💬";

    var win = document.createElement("div");
    win.id = "deligo-chat-window";
    win.innerHTML = [
      '<div id="deligo-chat-header">',
      "  <span>Deligo Assistant</span>",
      '  <button id="deligo-chat-close" aria-label="Close chat">×</button>',
      "</div>",
      '<div id="deligo-chat-messages"></div>',
      '<div id="deligo-chat-input-row">',
      '  <input id="deligo-chat-input" type="text" placeholder="Ask about products, orders, or policies..." autocomplete="off" />',
      '  <button id="deligo-chat-send" aria-label="Send message">➤</button>',
      "</div>",
      '<div id="deligo-chat-footer-note">AI assistant - responses may not always be accurate</div>',
    ].join("\n");

    document.body.appendChild(launcher);
    document.body.appendChild(win);

    return { launcher: launcher, win: win };
  }

  // ---------------------------------------------------------------------
  // Chat Logic
  // ---------------------------------------------------------------------
  function init() {
    var els = buildWidget();
    var messagesEl = els.win.querySelector("#deligo-chat-messages");
    var inputEl = els.win.querySelector("#deligo-chat-input");
    var sendBtn = els.win.querySelector("#deligo-chat-send");
    var closeBtn = els.win.querySelector("#deligo-chat-close");

    var sessionId = getSessionId();
    var greeted = false;

    function addMessage(text, sender) {
      var msg = document.createElement("div");
      msg.className = "deligo-msg " + (sender === "user" ? "deligo-msg-user" : "deligo-msg-bot");
      msg.textContent = text;
      messagesEl.appendChild(msg);
      messagesEl.scrollTop = messagesEl.scrollHeight;
      return msg;
    }

    function showTyping() {
      var typing = document.createElement("div");
      typing.className = "deligo-msg-typing";
      typing.id = "deligo-chat-typing";
      typing.textContent = "Typing...";
      messagesEl.appendChild(typing);
      messagesEl.scrollTop = messagesEl.scrollHeight;
    }

    function hideTyping() {
      var typing = document.getElementById("deligo-chat-typing");
      if (typing) typing.remove();
    }

    function toggleWindow() {
      var isOpen = els.win.classList.contains("open");
      if (isOpen) {
        els.win.classList.remove("open");
      } else {
        els.win.classList.add("open");
        if (!greeted) {
          greeted = true;
          addMessage(
            "Hi! I'm the Deligo assistant. I can help with product info, order tracking, and store policies. How can I help you today?",
            "bot"
          );
        }
        inputEl.focus();
      }
    }

    async function sendMessage() {
      var text = inputEl.value.trim();
      if (!text) return;

      addMessage(text, "user");
      inputEl.value = "";
      sendBtn.disabled = true;
      showTyping();

      try {
        var response = await fetch(API_BASE + "/chat", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            session_id: sessionId,
            message: text,
            user_id: getUserId(),
          }),
        });

        var data = await response.json();
        hideTyping();

        if (data && data.success) {
          addMessage(data.response, "bot");
        } else {
          addMessage(
            (data && data.response) ||
              "Sorry, something went wrong. Please try again.",
            "bot"
          );
        }
      } catch (err) {
        hideTyping();
        addMessage(
          "Sorry, I couldn't reach the assistant. Please check your connection and try again.",
          "bot"
        );
      } finally {
        sendBtn.disabled = false;
        inputEl.focus();
      }
    }

    els.launcher.addEventListener("click", toggleWindow);
    closeBtn.addEventListener("click", toggleWindow);
    sendBtn.addEventListener("click", sendMessage);
    inputEl.addEventListener("keydown", function (e) {
      if (e.key === "Enter") sendMessage();
    });
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
