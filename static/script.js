/* script.js — Celestia Astrology Chatbot */

// ── State ───────────────────────────────────────────────────────────────────
const state = {
  zodiac:  null,  // detected sign name, e.g. "Scorpio"
  symbol:  "",    // emoji symbol
  history: [],    // [{role: "user"|"bot", content: "..."}]
};

// ── DOM refs ────────────────────────────────────────────────────────────────
const screens = {
  onboarding: document.getElementById("onboarding"),
  reveal:     document.getElementById("reveal"),
  chat:       document.getElementById("chat-screen"),
};

const dobInput        = document.getElementById("dob-input");
const detectBtn       = document.getElementById("detect-btn");
const detectError     = document.getElementById("detect-error");
const btnText         = detectBtn.querySelector(".btn-text");
const btnLoader       = detectBtn.querySelector(".btn-loader");

const revealSymbol    = document.getElementById("reveal-symbol");
const revealSign      = document.getElementById("reveal-sign");
const revealElement   = document.getElementById("reveal-element");
const revealTagline   = document.getElementById("reveal-tagline");
const confidenceFill  = document.getElementById("confidence-fill");
const confidencePct   = document.getElementById("confidence-pct");
const startChatBtn    = document.getElementById("start-chat-btn");

const headerSymbol    = document.getElementById("header-symbol");
const headerSign      = document.getElementById("header-sign");
const messagesEl      = document.getElementById("messages");
const userInput       = document.getElementById("user-input");
const sendBtn         = document.getElementById("send-btn");
const resetBtn        = document.getElementById("reset-btn");

// Horoscope tab
const getHoroscopeBtn  = document.getElementById("get-horoscope-btn");
const horoscopeResult  = document.getElementById("horoscope-result");
const horoscopeDate    = document.getElementById("horoscope-date");

// Compatibility tab
const compatSign1El    = document.getElementById("compat-sign1");
const compatSign2El    = document.getElementById("compat-sign2");
const checkCompatBtn   = document.getElementById("check-compat-btn");
const compatResult     = document.getElementById("compat-result");

// Tab buttons
const tabBtns          = document.querySelectorAll(".tab-btn");
const tabPanels        = document.querySelectorAll(".tab-panel");

// ── Screen transitions ───────────────────────────────────────────────────────
function showScreen(name) {
  Object.entries(screens).forEach(([key, el]) => {
    el.classList.toggle("active", key === name);
  });
}

// ── Tab switching ────────────────────────────────────────────────────────────
tabBtns.forEach(btn => {
  btn.addEventListener("click", () => {
    const tab = btn.dataset.tab;

    tabBtns.forEach(b => b.classList.remove("active"));
    btn.classList.add("active");

    tabPanels.forEach(p => {
      const isTarget = p.id === `tab-${tab}`;
      p.classList.toggle("hidden", !isTarget);
      if (isTarget) p.classList.add("active");
      else p.classList.remove("active");
    });
  });
});

// ── Starfield canvas ─────────────────────────────────────────────────────────
(function initStars() {
  const canvas = document.getElementById("stars");
  const ctx    = canvas.getContext("2d");
  let stars    = [];

  function resize() {
    canvas.width  = window.innerWidth;
    canvas.height = window.innerHeight;
  }

  function makeStars(count) {
    stars = Array.from({ length: count }, () => ({
      x:     Math.random() * canvas.width,
      y:     Math.random() * canvas.height,
      r:     Math.random() * 1.2 + 0.2,
      alpha: Math.random(),
      speed: Math.random() * 0.004 + 0.001,
    }));
  }

  function draw() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    stars.forEach(s => {
      s.alpha += s.speed;
      if (s.alpha > 1 || s.alpha < 0) s.speed *= -1;
      ctx.beginPath();
      ctx.arc(s.x, s.y, s.r, 0, Math.PI * 2);
      ctx.fillStyle = `rgba(200, 185, 255, ${s.alpha * 0.7})`;
      ctx.fill();
    });
    requestAnimationFrame(draw);
  }

  window.addEventListener("resize", () => { resize(); makeStars(200); });
  resize();
  makeStars(200);
  draw();
})();

// ── Zodiac detection ─────────────────────────────────────────────────────────
detectBtn.addEventListener("click", async () => {
  const dob = dobInput.value.trim();
  if (!dob) { showError("Please enter your date of birth."); return; }

  setLoading(true);
  hideError();

  try {
    const res  = await fetch("/detect", {
      method:  "POST",
      headers: { "Content-Type": "application/json" },
      body:    JSON.stringify({ dob }),
    });
    const data = await res.json();

    if (data.error) { showError(data.error); setLoading(false); return; }

    // Save to state
    state.zodiac = data.sign;
    state.symbol = data.symbol || "✦";

    // Populate reveal screen
    revealSymbol.textContent  = data.symbol  || "✦";
    revealSign.textContent    = data.sign    || "Unknown";
    revealElement.textContent = `${data.element || ""} sign`;
    revealTagline.textContent = data.tagline || "";

    // Animate confidence bar
    const pct = data.confidence || 0;
    confidencePct.textContent = `${pct}%`;
    setTimeout(() => {
      confidenceFill.style.width = `${pct}%`;
    }, 100);

    setLoading(false);
    showScreen("reveal");

  } catch {
    showError("Something went wrong. Is the server running?");
    setLoading(false);
  }
});

dobInput.addEventListener("keydown", e => {
  if (e.key === "Enter") detectBtn.click();
});

// ── Reveal → Chat ────────────────────────────────────────────────────────────
startChatBtn.addEventListener("click", () => {
  // Update header
  headerSymbol.textContent = state.symbol;
  headerSign.textContent   = state.zodiac;

  // Update compatibility tab sign label
  compatSign1El.textContent = state.zodiac;

  // Set horoscope date
  const today = new Date().toLocaleDateString("en-US", {
    weekday: "long", year: "numeric", month: "long", day: "numeric"
  });
  horoscopeDate.textContent = today;

  // Clear messages and add opening message
  messagesEl.innerHTML = "";
  state.history = [];
  addMessage("bot", openingMessage(state.zodiac, state.symbol));

  // Reset feature panels
  horoscopeResult.classList.add("hidden");
  horoscopeResult.innerHTML = "";
  compatResult.classList.add("hidden");
  compatResult.innerHTML = "";

  showScreen("chat");
  userInput.focus();
});

function openingMessage(sign, symbol) {
  const greetings = {
    Aries:       `${symbol} The Ram stirs — bold energy fills this space. What's on your mind?`,
    Taurus:      `${symbol} Settle in, dear Taurus. The stars are patient, and so am I. What brings you here?`,
    Gemini:      `${symbol} Ah, a Gemini! Two minds, endless curiosity. What shall we explore today?`,
    Cancer:      `${symbol} The moon watches over you, gentle Cancer. You're safe to ask anything here.`,
    Leo:         `${symbol} A Leo graces me with their presence! The cosmos is ready to shine for you.`,
    Virgo:       `${symbol} Welcome, Virgo. Let's cut through the noise — what would you like to understand?`,
    Libra:       `${symbol} Balance and beauty, dear Libra. I'm here to help you find your cosmic harmony.`,
    Scorpio:     `${symbol} The depths call to you, Scorpio. What truths are you seeking in the dark?`,
    Sagittarius: `${symbol} The archer draws back the bow — where shall we aim today, Sagittarius?`,
    Capricorn:   `${symbol} A Capricorn arrives with purpose. The mountain is steep, but the stars guide your path.`,
    Aquarius:    `${symbol} The water-bearer arrives! Let's break every convention the stars have to offer.`,
    Pisces:      `${symbol} You drift in like a dream, dear Pisces. What visions shall we explore together?`,
  };
  return greetings[sign] || `${symbol} Welcome, ${sign}. The stars have been waiting for you.`;
}

// ── Sending messages ─────────────────────────────────────────────────────────
sendBtn.addEventListener("click", sendMessage);
userInput.addEventListener("keydown", e => {
  if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); sendMessage(); }
});

// Auto-resize textarea
userInput.addEventListener("input", () => {
  userInput.style.height = "auto";
  userInput.style.height = Math.min(userInput.scrollHeight, 120) + "px";
});

async function sendMessage() {
  const text = userInput.value.trim();
  if (!text || sendBtn.disabled) return;

  addMessage("user", text);
  state.history.push({ role: "user", content: text });

  userInput.value = "";
  userInput.style.height = "auto";
  sendBtn.disabled = true;

  const typingEl = addTyping();

  try {
    const res  = await fetch("/chat", {
      method:  "POST",
      headers: { "Content-Type": "application/json" },
      body:    JSON.stringify({
        message: text,
        zodiac:  state.zodiac,
        history: state.history.slice(-10),
      }),
    });
    const data = await res.json();

    typingEl.remove();

    const reply      = data.response || data.error || "The stars are silent…";
    const confidence = data.confidence;
    const isFallback = data.fallback;

    addMessage("bot", reply, confidence, isFallback);
    state.history.push({ role: "bot", content: reply });

  } catch {
    typingEl.remove();
    addMessage("bot", "The cosmic signal was lost. Please try again.");
  }

  sendBtn.disabled = false;
  userInput.focus();
}

// ── Message helpers ───────────────────────────────────────────────────────────
function addMessage(role, text, confidence, isFallback) {
  const wrapper = document.createElement("div");
  wrapper.className = `message ${role}`;

  const bubble = document.createElement("div");
  bubble.className = "bubble";
  bubble.textContent = text;
  wrapper.appendChild(bubble);

  // Show confidence tag under bot messages
  if (role === "bot" && confidence !== undefined) {
    const tag = document.createElement("div");
    tag.className = `msg-confidence${isFallback ? " fallback" : ""}`;
    tag.textContent = isFallback
      ? `⚠ Low confidence — try a more specific question`
      : `✦ ${confidence}% match confidence`;
    wrapper.appendChild(tag);
  }

  messagesEl.appendChild(wrapper);
  scrollToBottom();
  return wrapper;
}

function addTyping() {
  const el = document.createElement("div");
  el.className = "typing-dots";
  el.innerHTML = "<span></span><span></span><span></span>";
  messagesEl.appendChild(el);
  scrollToBottom();
  return el;
}

function scrollToBottom() {
  messagesEl.scrollTop = messagesEl.scrollHeight;
}

// ── Daily Horoscope ───────────────────────────────────────────────────────────
getHoroscopeBtn.addEventListener("click", async () => {
  const btnTxt = getHoroscopeBtn.querySelector(".btn-text");
  const btnLdr = getHoroscopeBtn.querySelector(".btn-loader");

  getHoroscopeBtn.disabled = true;
  btnTxt.classList.add("hidden");
  btnLdr.classList.remove("hidden");
  horoscopeResult.classList.add("hidden");

  try {
    const res  = await fetch("/horoscope", {
      method:  "POST",
      headers: { "Content-Type": "application/json" },
      body:    JSON.stringify({ zodiac: state.zodiac }),
    });
    const data = await res.json();

    if (data.error) {
      showFeatureResult(horoscopeResult, `Error: ${data.error}`, null, "Horoscope");
    } else {
      showFeatureResult(
        horoscopeResult,
        data.horoscope,
        data.confidence,
        `${data.zodiac} · ${data.date}`
      );
    }

  } catch {
    showFeatureResult(horoscopeResult, "Could not reach the cosmos. Please try again.", null, "Error");
  }

  getHoroscopeBtn.disabled = false;
  btnTxt.classList.remove("hidden");
  btnLdr.classList.add("hidden");
});

// ── Compatibility ─────────────────────────────────────────────────────────────
checkCompatBtn.addEventListener("click", async () => {
  const sign2 = compatSign2El.value;
  if (!sign2) { alert("Please choose a sign to compare with."); return; }

  const btnTxt = checkCompatBtn.querySelector(".btn-text");
  const btnLdr = checkCompatBtn.querySelector(".btn-loader");

  checkCompatBtn.disabled = true;
  btnTxt.classList.add("hidden");
  btnLdr.classList.remove("hidden");
  compatResult.classList.add("hidden");

  try {
    const res  = await fetch("/compatibility", {
      method:  "POST",
      headers: { "Content-Type": "application/json" },
      body:    JSON.stringify({ sign1: state.zodiac, sign2 }),
    });
    const data = await res.json();

    if (data.error) {
      showFeatureResult(compatResult, `Error: ${data.error}`, null, "Compatibility");
    } else {
      // Extract rating line if present
      const raw    = data.compatibility || "";
      const ratingMatch = raw.match(/Compatibility Rating:\s*(.+)/i);
      const rating = ratingMatch ? ratingMatch[1].trim() : null;
      const body   = raw.replace(/Compatibility Rating:.+/i, "").trim();

      showFeatureResult(
        compatResult,
        body,
        data.confidence,
        `${data.sign1} & ${data.sign2}`,
        rating
      );
    }

  } catch {
    showFeatureResult(compatResult, "Could not reach the cosmos. Please try again.", null, "Error");
  }

  checkCompatBtn.disabled = false;
  btnTxt.classList.remove("hidden");
  btnLdr.classList.add("hidden");
});

// ── Feature result renderer ────────────────────────────────────────────────────
function showFeatureResult(el, text, confidence, label, rating) {
  el.innerHTML = "";

  if (label) {
    const labelEl = document.createElement("span");
    labelEl.className = "result-label";
    labelEl.textContent = label;
    el.appendChild(labelEl);
  }

  const body = document.createElement("p");
  body.textContent = text;
  el.appendChild(body);

  if (confidence !== undefined && confidence !== null) {
    const conf = document.createElement("div");
    conf.className = "msg-confidence";
    conf.textContent = `✦ ${confidence}% match confidence`;
    conf.style.marginTop = "12px";
    el.appendChild(conf);
  }

  if (rating) {
    const ratingEl = document.createElement("div");
    ratingEl.className = "result-rating";
    ratingEl.textContent = `Compatibility Rating: ${rating}`;
    el.appendChild(ratingEl);
  }

  el.classList.remove("hidden");
}

// ── Reset ─────────────────────────────────────────────────────────────────────
resetBtn.addEventListener("click", async () => {
  // Clear server session
  try {
    await fetch("/clear", { method: "POST" });
  } catch { /* ignore */ }

  state.zodiac  = null;
  state.symbol  = "";
  state.history = [];
  dobInput.value = "";

  // Reset confidence bar
  confidenceFill.style.width = "0%";
  confidencePct.textContent  = "";

  showScreen("onboarding");
});

// ── UI helpers ────────────────────────────────────────────────────────────────
function setLoading(loading) {
  detectBtn.disabled = loading;
  btnText.classList.toggle("hidden", loading);
  btnLoader.classList.toggle("hidden", !loading);
}

function showError(msg) {
  detectError.textContent = msg;
  detectError.classList.remove("hidden");
}

function hideError() {
  detectError.classList.add("hidden");
}