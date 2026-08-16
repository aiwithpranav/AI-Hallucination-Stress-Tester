/**
 * AI Hallucination Stress Tester — Frontend Script
 *
 * Responsibilities:
 *  - Validate user input
 *  - Manage button/loading state
 *  - POST question to /api/analyze
 *  - Dynamically render all result sections from backend response
 *  - Handle all error cases gracefully
 *
 * IMPORTANT: All displayed data comes from the backend response.
 *            Nothing is hardcoded or invented here.
 */

'use strict';

/* ─────────────────────────────────────────────
   CONFIG
───────────────────────────────────────────── */

const CONFIG = {
  // Backend API URL — change if your backend runs on a different host/port
  API_URL: 'http://localhost:8000/api/analyze',
  MAX_CHARS: 2000,
  MIN_CHARS: 3,
};

/* ─────────────────────────────────────────────
   DOM REFERENCES
───────────────────────────────────────────── */

const els = {
  questionInput:       document.getElementById('question-input'),
  analyzeBtn:          document.getElementById('analyze-btn'),
  charCount:           document.getElementById('char-count'),
  formError:           document.getElementById('form-error'),
  resultsSection:      document.getElementById('results-section'),
  emptyState:          document.getElementById('empty-state'),
  globalError:         document.getElementById('global-error'),
  globalErrorTitle:    document.getElementById('global-error-title'),
  globalErrorDetail:   document.getElementById('global-error-detail'),
  errorCloseBtn:       document.getElementById('error-close-btn'),

  // Result fields
  resultDate:              document.getElementById('result-date'),
  resultStatusBadge:       document.getElementById('result-status-badge'),
  resultQuestion:          document.getElementById('result-question'),
  resultAnswer:            document.getElementById('result-answer'),
  resultRiskBadge:         document.getElementById('result-risk-badge'),
  resultRiskBar:           document.getElementById('result-risk-bar'),
  confidenceRingFill:      document.getElementById('confidence-ring-fill'),
  resultConfidence:        document.getElementById('result-confidence'),
  resultConfidenceLabel:   document.getElementById('result-confidence-label'),
  resultSummary:           document.getElementById('result-summary'),
  resultEvidenceStatus:    document.getElementById('result-evidence-status'),
  resultVerificationMethod:document.getElementById('result-verification-method'),
  resultEvidenceText:      document.getElementById('result-evidence-text'),
  resultVerificationOutcome:document.getElementById('result-verification-outcome'),
  resultSource:            document.getElementById('result-source'),
  resultSourceStatus:      document.getElementById('result-source-status'),
  resultReliabilityDots:   document.getElementById('result-reliability-dots'),
  resultReliabilityLabel:  document.getElementById('result-reliability-label'),
  resultRiskExplanation:   document.getElementById('result-risk-explanation'),
  resultRecommendation:    document.getElementById('result-recommendation'),

  clearBtn:                document.getElementById('clear-btn'),
  tryAgainBtn:             document.getElementById('try-again-btn'),
  seeWhyBtn:               document.getElementById('see-why-btn'),
  seeWhyContent:           document.getElementById('see-why-content'),
};

/* ─────────────────────────────────────────────
   SVG GRADIENT FOR CONFIDENCE RING
   (injected into the SVG once on load)
───────────────────────────────────────────── */

function injectRingGradient() {
  const svgEl = document.querySelector('.confidence-ring');
  if (!svgEl) return;
  const defs = document.createElementNS('http://www.w3.org/2000/svg', 'defs');
  defs.innerHTML = `
    <linearGradient id="ringGrad" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" stop-color="#a78bfa"/>
      <stop offset="100%" stop-color="#38bdf8"/>
    </linearGradient>`;
  svgEl.prepend(defs);
}

/* ─────────────────────────────────────────────
   CHARACTER COUNTER
───────────────────────────────────────────── */

function updateCharCount() {
  const len = els.questionInput.value.length;
  els.charCount.textContent = `${len} / ${CONFIG.MAX_CHARS}`;
  els.charCount.classList.toggle('warn', len > CONFIG.MAX_CHARS * 0.85);
  els.charCount.classList.toggle('danger', len >= CONFIG.MAX_CHARS);
}

/* ─────────────────────────────────────────────
   INPUT VALIDATION
───────────────────────────────────────────── */

function validateInput(question) {
  const trimmed = question.trim();
  if (!trimmed) return 'Please enter a question before analyzing.';
  if (trimmed.length < CONFIG.MIN_CHARS) return 'Please enter a more complete question (at least 3 characters).';
  if (trimmed.length > CONFIG.MAX_CHARS) return `Question is too long. Maximum ${CONFIG.MAX_CHARS} characters.`;
  return null;
}

function showFormError(msg) {
  els.formError.textContent = msg;
  els.formError.hidden = false;
}

function clearFormError() {
  els.formError.textContent = '';
  els.formError.hidden = true;
}

/* ─────────────────────────────────────────────
   BUTTON STATE
───────────────────────────────────────────── */

function setAnalyzing(isAnalyzing) {
  els.analyzeBtn.disabled = isAnalyzing;
  els.analyzeBtn.classList.toggle('loading', isAnalyzing);
  els.analyzeBtn.setAttribute('aria-busy', isAnalyzing ? 'true' : 'false');
}

/* ─────────────────────────────────────────────
   GLOBAL ERROR BANNER
───────────────────────────────────────────── */

function showGlobalError(title, detail) {
  els.globalErrorTitle.textContent = title;
  els.globalErrorDetail.textContent = detail;
  els.globalError.hidden = false;
  els.globalError.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

function hideGlobalError() {
  els.globalError.hidden = true;
}

/* ─────────────────────────────────────────────
   SKELETON LOADING
   Show shimmering placeholders while waiting
───────────────────────────────────────────── */

function applySkeletons() {
  const skeletonTargets = [
    els.resultDate, els.resultQuestion, els.resultAnswer,
    els.resultSummary, els.resultEvidenceText,
    els.resultRiskExplanation, els.resultRecommendation,
  ];
  skeletonTargets.forEach(el => el.classList.add('skeleton'));
}

function removeSkeletons() {
  document.querySelectorAll('.skeleton').forEach(el => el.classList.remove('skeleton'));
}

/* ─────────────────────────────────────────────
   STATUS CLASSIFICATION → CSS CLASS
───────────────────────────────────────────── */

function statusToCssClass(status) {
  const map = {
    'Verified':             'status-verified',
    'Partially Supported':  'status-partial',
    'Not Supported':        'status-not-supported',
    'Unable to Verify':     'status-unable',
  };
  return map[status] || '';
}

/* ─────────────────────────────────────────────
   RISK → CSS CLASS + BAR CLASS
───────────────────────────────────────────── */

function riskToCssClasses(risk) {
  const map = {
    'NO Risk':   { badge: 'risk-no',    bar: 'risk-bar-no' },
    'Very Low':  { badge: 'risk-vlow',  bar: 'risk-bar-vlow' },
    'Low':       { badge: 'risk-low',   bar: 'risk-bar-low' },
    'Medium':    { badge: 'risk-medium',bar: 'risk-bar-medium' },
    'High':      { badge: 'risk-high',  bar: 'risk-bar-high' },
  };
  return map[risk] || { badge: '', bar: '' };
}

/* ─────────────────────────────────────────────
   SOURCE STATUS → CSS CLASS
───────────────────────────────────────────── */

function sourceStatusToCssClass(status) {
  const map = {
    'Official':      'source-status-official',
    'Authoritative': 'source-status-authoritative',
  };
  return map[status] || '';
}

/* ─────────────────────────────────────────────
   RELIABILITY → DOTS
───────────────────────────────────────────── */

function setReliabilityDots(reliability) {
  const dots = els.resultReliabilityDots.querySelectorAll('.rdot');
  // Reset all dots
  dots.forEach(d => { d.className = 'rdot'; });

  let activeDots = 0;
  let dotClass = '';

  if (reliability === 'High')   { activeDots = 5; dotClass = 'active-high'; }
  else if (reliability === 'Medium') { activeDots = 3; dotClass = 'active-medium'; }
  else if (reliability === 'Low')    { activeDots = 1; dotClass = 'active-low'; }

  for (let i = 0; i < activeDots; i++) {
    if (dots[i]) dots[i].classList.add(dotClass);
  }
}

/* ─────────────────────────────────────────────
   CONFIDENCE RING ANIMATION
───────────────────────────────────────────── */

function setConfidenceRing(confidenceValue) {
  // confidenceValue: 0.0 – 1.0
  const circumference = 2 * Math.PI * 32; // r=32 → ~201
  const offset = circumference * (1 - confidenceValue);
  els.confidenceRingFill.setAttribute('stroke-dashoffset', offset.toFixed(1));
  els.confidenceRingFill.setAttribute('stroke-dasharray', `${circumference} ${circumference}`);
}

/* ─────────────────────────────────────────────
   FORMAT DATE
───────────────────────────────────────────── */

function formatDate(dateStr) {
  if (!dateStr) return '—';
  try {
    const d = new Date(dateStr);
    return d.toLocaleString(undefined, {
      year: 'numeric', month: 'short', day: 'numeric',
      hour: '2-digit', minute: '2-digit',
    });
  } catch {
    return dateStr;
  }
}

/* ─────────────────────────────────────────────
   SAFE TEXT — prevent XSS
───────────────────────────────────────────── */

function safeText(val) {
  return (val !== undefined && val !== null && val !== '') ? String(val) : '—';
}

/* ─────────────────────────────────────────────
   RENDER RESULTS
   All data comes from the backend JSON response.
   Nothing is invented here.
───────────────────────────────────────────── */

function renderResults(data) {
  // ── Date ──
  els.resultDate.textContent = formatDate(data.date);

  // ── Status Badge ──
  els.resultStatusBadge.textContent = safeText(data.status);
  els.resultStatusBadge.className = 'status-badge';
  const statusClass = statusToCssClass(data.status);
  if (statusClass) els.resultStatusBadge.classList.add(statusClass);

  // ── Question + Answer ──
  els.resultQuestion.textContent = safeText(data.question);
  els.resultAnswer.textContent = safeText(data.answer);

  // ── Hallucination Risk ──
  const riskClasses = riskToCssClasses(data.hallucinationRisk);
  els.resultRiskBadge.textContent = safeText(data.hallucinationRisk);
  els.resultRiskBadge.className = 'risk-level-badge';
  if (riskClasses.badge) els.resultRiskBadge.classList.add(riskClasses.badge);

  // Risk bar
  els.resultRiskBar.className = 'risk-bar';
  if (riskClasses.bar) {
    // Small delay so transition animates
    requestAnimationFrame(() => {
      requestAnimationFrame(() => els.resultRiskBar.classList.add(riskClasses.bar));
    });
  }

  // ── Confidence Ring ──
  const confVal = typeof data.confidence === 'number' ? data.confidence : 0;
  const confPct = Math.round(confVal * 100);
  setConfidenceRing(confVal);
  els.resultConfidence.textContent = `${confPct}%`;

  let confLabel = 'Low confidence';
  if (confPct >= 85) confLabel = 'Very high confidence';
  else if (confPct >= 70) confLabel = 'High confidence';
  else if (confPct >= 50) confLabel = 'Moderate confidence';
  else if (confPct >= 30) confLabel = 'Low confidence';
  else confLabel = 'Very low confidence';
  els.resultConfidenceLabel.textContent = confLabel;

  // ── Analysis Summary ──
  els.resultSummary.textContent = safeText(data.summary);

  // ── Evidence ──
  els.resultEvidenceStatus.textContent = safeText(data.evidenceStatus);
  els.resultVerificationMethod.textContent = safeText(data.verificationMethod);
  els.resultEvidenceText.textContent = safeText(data.evidenceText);
  els.resultVerificationOutcome.textContent = safeText(data.verificationOutcome);

  // ── Source ──
  const source = safeText(data.source);
  if (source.startsWith('http')) {
    const link = document.createElement('a');
    link.href = source;
    link.target = '_blank';
    link.rel = 'noopener noreferrer';
    link.textContent = source;
    els.resultSource.innerHTML = '';
    els.resultSource.appendChild(link);
  } else {
    els.resultSource.textContent = source;
  }

  // Source status badge
  els.resultSourceStatus.textContent = safeText(data.sourceStatus);
  els.resultSourceStatus.className = 'source-status-badge';
  const srcClass = sourceStatusToCssClass(data.sourceStatus);
  if (srcClass) els.resultSourceStatus.classList.add(srcClass);

  // ── Source Reliability ──
  const rel = safeText(data.sourceReliability);
  setReliabilityDots(rel === '—' ? 'Low' : rel);
  els.resultReliabilityLabel.textContent = rel;

  // ── Risk Explanation + Recommendation ──
  els.resultRiskExplanation.textContent = safeText(data.riskExplanation);
  els.resultRecommendation.textContent = safeText(data.recommendation);

  // Show "See Why" button, hide content by default
  if (els.seeWhyBtn) {
    els.seeWhyBtn.hidden = false;
    els.seeWhyBtn.setAttribute('aria-expanded', 'false');
    els.seeWhyContent.hidden = true;
  }

  removeSkeletons();
}

/* ─────────────────────────────────────────────
   SHOW / HIDE SECTIONS
───────────────────────────────────────────── */

function showResults() {
  els.emptyState.hidden = true;
  els.resultsSection.hidden = false;
}

function showEmpty() {
  els.emptyState.hidden = false;
  els.resultsSection.hidden = true;
  
  if (els.seeWhyBtn) {
    els.seeWhyBtn.hidden = true;
    els.seeWhyBtn.setAttribute('aria-expanded', 'false');
    els.seeWhyContent.hidden = true;
  }
}

function handleClear() {
  els.questionInput.value = '';
  updateCharCount();
  clearFormError();
  hideGlobalError();
  showEmpty();
  els.questionInput.focus();
}

function handleTryAgain() {
  // If already analyzing, do nothing (prevent duplicate requests)
  if (els.analyzeBtn.disabled) return;

  // Keep the current question intact and re-run the analysis
  clearFormError();
  hideGlobalError();

  // If there's no question, prompt the user instead of clearing
  const question = els.questionInput.value;
  const validationError = validateInput(question);
  if (validationError) {
    showFormError(validationError);
    els.questionInput.focus();
    return;
  }

  // Scroll to top so user sees loading state, then run again
  window.scrollTo({ top: 0, behavior: 'smooth' });
  handleAnalyze();
}

function toggleSeeWhy() {
  const isExpanded = els.seeWhyBtn.getAttribute('aria-expanded') === 'true';
  const newExpanded = !isExpanded;
  els.seeWhyBtn.setAttribute('aria-expanded', newExpanded);
  els.seeWhyContent.hidden = !newExpanded;
}

/* ─────────────────────────────────────────────
   MAIN ANALYZE HANDLER
───────────────────────────────────────────── */

async function handleAnalyze() {
  const question = els.questionInput.value;

  // Clear previous errors
  clearFormError();
  hideGlobalError();

  // Validate
  const validationError = validateInput(question);
  if (validationError) {
    showFormError(validationError);
    els.questionInput.focus();
    return;
  }

  // Set loading state
  setAnalyzing(true);
  showResults();
  applySkeletons();

  // Reset dynamic fields
  els.resultStatusBadge.textContent = '—';
  els.resultStatusBadge.className = 'status-badge';
  els.resultRiskBadge.textContent = '—';
  els.resultRiskBar.className = 'risk-bar';
  els.resultConfidence.textContent = '—';
  els.resultConfidenceLabel.textContent = 'Analyzing…';
  setConfidenceRing(0);

  // Scroll results into view
  setTimeout(() => {
    els.resultsSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
  }, 100);

  try {
    const response = await fetch(CONFIG.API_URL, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Accept': 'application/json',
      },
      body: JSON.stringify({ question: question.trim() }),
    });

    if (!response.ok) {
      let errDetail = `Server returned ${response.status}`;
      try {
        const errBody = await response.json();
        errDetail = errBody.detail || errBody.error || errDetail;
      } catch { /* ignore JSON parse error on error body */ }

      throw new ApiError(`Analysis failed (HTTP ${response.status})`, errDetail);
    }

    let data;
    try {
      data = await response.json();
    } catch {
      throw new ApiError('Invalid response', 'The server returned a response that could not be parsed. Please try again.');
    }

    // Validate expected fields exist
    if (!data || typeof data !== 'object') {
      throw new ApiError('Unexpected response format', 'The server response was not in the expected format.');
    }

    renderResults(data);

  } catch (err) {
    removeSkeletons();
    showEmpty();

    if (err instanceof ApiError) {
      showGlobalError(err.title, err.detail);
    } else if (err.name === 'TypeError' && err.message.includes('fetch')) {
      showGlobalError(
        'Cannot reach the backend',
        'The backend server is not reachable. Make sure the FastAPI server is running at ' +
        CONFIG.API_URL.replace('/api/analyze', '') +
        ' and try again.'
      );
    } else {
      showGlobalError('Unexpected error', err.message || 'An unknown error occurred. Please try again.');
    }
  } finally {
    setAnalyzing(false);
  }
}

/* ─────────────────────────────────────────────
   CUSTOM ERROR CLASS
───────────────────────────────────────────── */

class ApiError extends Error {
  constructor(title, detail) {
    super(title);
    this.title = title;
    this.detail = detail;
    this.name = 'ApiError';
  }
}

/* ─────────────────────────────────────────────
   EVENT LISTENERS
───────────────────────────────────────────── */

// Char count update
els.questionInput.addEventListener('input', updateCharCount);

// Submit on button click
els.analyzeBtn.addEventListener('click', handleAnalyze);

// Submit on Enter
els.questionInput.addEventListener('keydown', (e) => {
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault();
    if (!els.analyzeBtn.disabled) {
      handleAnalyze();
    }
  }
});

// Close error banner
els.errorCloseBtn.addEventListener('click', hideGlobalError);

// Clear and Try Again
if (els.clearBtn) els.clearBtn.addEventListener('click', handleClear);
if (els.tryAgainBtn) els.tryAgainBtn.addEventListener('click', handleTryAgain);

// See Why
if (els.seeWhyBtn) els.seeWhyBtn.addEventListener('click', toggleSeeWhy);

// Cursor-reactive ambient background
// Uses a CSS radial gradient that smoothly follows the cursor,
// giving a soft disturbance on the background surface without
// literally moving the orbs or creating a cursor trail.
(function initCursorGlow() {
  // Respect reduced motion preference
  if (window.matchMedia('(prefers-reduced-motion: reduce)').matches) return;

  // Inject the glow overlay element
  const glow = document.createElement('div');
  glow.className = 'cursor-glow';
  document.body.prepend(glow);

  let targetX = 50, targetY = 50;   // % of viewport
  let currentX = 50, currentY = 50; // smoothed
  let rafId = null;
  let glowActive = false;
  let leaveTimer = null;

  const LERP = 0.07; // smoothing factor — lower = lazier trail

  function lerp(a, b, t) { return a + (b - a) * t; }

  function animate() {
    currentX = lerp(currentX, targetX, LERP);
    currentY = lerp(currentY, targetY, LERP);
    document.documentElement.style.setProperty('--mouse-x', `${currentX.toFixed(2)}%`);
    document.documentElement.style.setProperty('--mouse-y', `${currentY.toFixed(2)}%`);
    rafId = requestAnimationFrame(animate);
  }

  document.addEventListener('mousemove', (e) => {
    targetX = (e.clientX / window.innerWidth) * 100;
    targetY = (e.clientY / window.innerHeight) * 100;

    if (!glowActive) {
      glowActive = true;
      glow.classList.add('active');
      if (!rafId) rafId = requestAnimationFrame(animate);
    }

    // Reset leave timer
    clearTimeout(leaveTimer);
    leaveTimer = setTimeout(() => {
      glow.classList.remove('active');
      glowActive = false;
      if (rafId) { cancelAnimationFrame(rafId); rafId = null; }
    }, 2000);
  });

  // Touch devices — no persistent glow
  document.addEventListener('mouseleave', () => {
    clearTimeout(leaveTimer);
    glow.classList.remove('active');
    glowActive = false;
    if (rafId) { cancelAnimationFrame(rafId); rafId = null; }
  });
}());

/* ─────────────────────────────────────────────
   INIT
───────────────────────────────────────────── */

function init() {
  injectRingGradient();
  updateCharCount();
  showEmpty();
  els.questionInput.focus();
}

init();
