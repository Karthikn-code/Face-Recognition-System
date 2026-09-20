/**
 * FaceID Web Application - Client Controller
 */

// Application State
const state = {
  activeTab: 'tab-identify',
  threshold: 0.34,
  currentImage: null, // HTMLImageElement
  lastIdentifyResponse: null,
  enrolledIdentities: [],
  samples: []
};

// DOM Element References
const elements = {
  navItems: document.querySelectorAll('.nav-item'),
  tabPanels: document.querySelectorAll('.tab-panel'),
  mainHeading: document.getElementById('main-heading'),
  pageSubheading: document.getElementById('page-subheading'),
  
  // Status & Counts
  systemBackend: document.getElementById('system-backend'),
  enrolledBadge: document.getElementById('enrolled-badge'),
  topThreshDisplay: document.getElementById('top-thresh-display'),
  
  // Theme Switcher
  btnThemeToggle: document.getElementById('btn-theme-toggle'),
  themeDropdownMenu: document.getElementById('theme-menu-dropdown'),
  activeThemeName: document.getElementById('active-theme-name'),
  themeMenuItems: document.querySelectorAll('.theme-menu-item'),
  
  // Identifier Elements
  dropzone: document.getElementById('image-dropzone'),
  canvasWrapper: document.getElementById('canvas-wrapper'),
  faceCanvas: document.getElementById('face-canvas'),
  fileInput: document.getElementById('file-input'),
  btnBrowseFile: document.getElementById('btn-browse-file'),
  btnClearCanvas: document.getElementById('btn-clear-canvas'),
  loadingOverlay: document.getElementById('loading-overlay'),
  thresholdSlider: document.getElementById('threshold-slider'),
  sliderThresholdVal: document.getElementById('slider-threshold-val'),
  samplesContainer: document.getElementById('samples-container'),
  
  // Results Elements
  detectedFacesPill: document.getElementById('detected-faces-pill'),
  emptyResultsState: document.getElementById('empty-results-state'),
  facesResultsList: document.getElementById('faces-results-list'),
  
  // Gallery Elements
  galleryCardsGrid: document.getElementById('gallery-cards-grid'),
  gallerySearch: document.getElementById('gallery-search'),
  btnGalleryEnroll: document.getElementById('btn-gallery-enroll'),
  btnOpenEnrollModal: document.getElementById('btn-open-enroll-modal'),
  
  // Modal Elements
  enrollModal: document.getElementById('enroll-modal'),
  btnCloseEnroll: document.getElementById('btn-close-enroll'),
  btnCancelEnroll: document.getElementById('btn-cancel-enroll'),
  btnSubmitEnroll: document.getElementById('btn-submit-enroll'),
  enrollNameInput: document.getElementById('enroll-name-input'),
  modalEnrollDropzone: document.getElementById('modal-enroll-dropzone'),
  modalFileInput: document.getElementById('modal-file-input'),
  enrollPreviewGrid: document.getElementById('enroll-preview-grid'),
  
  // Failures Elements
  failuresGrid: document.getElementById('failures-grid'),
  
  // Toast
  toastContainer: document.getElementById('toast-container')
};

// Headings by Tab
const tabHeadings = {
  'tab-identify': {
    title: 'Live Face Identification',
    sub: 'Detect facial landmarks, extract ArcFace embeddings, and match against gallery with unknown rejection.'
  },
  'tab-gallery': {
    title: 'Enrolled Identity Gallery',
    sub: 'Manage stored facial identities, reference templates, and persistent identity records.'
  },
  'tab-benchmark': {
    title: 'System Evaluation & Benchmark',
    sub: 'Empirical zero-leakage evaluation metrics, threshold trade-off sweeps, and publication plots.'
  },
  'tab-failures': {
    title: 'Boundary & Failure Inspector',
    sub: 'Analyze empirical boundary stress cases: strict threshold false rejects and lax false accepts.'
  }
};

let modalEnrollImages = [];

/* ==========================================================================
   Initialization & Navigation
   ========================================================================== */

document.addEventListener('DOMContentLoaded', async () => {
  initThemeSwitcher();
  initNavigation();
  initDragAndDrop();
  initThresholdSlider();
  initEnrollmentModal();
  
  // Load initial server data
  await loadSystemStatus();
  await loadSamples();
  await loadIdentities();
  await loadMetricsAndFailures();
});

const themeLabels = {
  'obsidian': 'Obsidian',
  'emerald': 'Emerald',
  'amethyst': 'Amethyst',
  'cyber': 'Cyber',
  'light': 'Light'
};

function initThemeSwitcher() {
  const savedTheme = localStorage.getItem('faceid-theme') || 'obsidian';
  setTheme(savedTheme);

  if (elements.btnThemeToggle) {
    elements.btnThemeToggle.addEventListener('click', (e) => {
      e.stopPropagation();
      elements.themeDropdownMenu.classList.toggle('hidden');
    });
  }

  document.addEventListener('click', () => {
    if (elements.themeDropdownMenu && !elements.themeDropdownMenu.classList.contains('hidden')) {
      elements.themeDropdownMenu.classList.add('hidden');
    }
  });

  elements.themeMenuItems.forEach(item => {
    item.addEventListener('click', (e) => {
      e.stopPropagation();
      const theme = item.getAttribute('data-theme');
      setTheme(theme);
      elements.themeDropdownMenu.classList.add('hidden');
      showToast(`Color theme switched to ${themeLabels[theme] || theme}!`, 'info');
    });
  });
}

function setTheme(theme) {
  document.body.setAttribute('data-theme', theme);
  localStorage.setItem('faceid-theme', theme);

  if (elements.activeThemeName) {
    elements.activeThemeName.textContent = themeLabels[theme] || 'Obsidian';
  }

  elements.themeMenuItems.forEach(item => {
    const isMatch = item.getAttribute('data-theme') === theme;
    item.classList.toggle('active', isMatch);
  });
}

function initNavigation() {
  elements.navItems.forEach(btn => {
    btn.addEventListener('click', () => {
      const targetTab = btn.getAttribute('data-tab');
      switchTab(targetTab);
    });
  });
}

function switchTab(tabId) {
  state.activeTab = tabId;
  
  elements.navItems.forEach(b => {
    b.classList.toggle('active', b.getAttribute('data-tab') === tabId);
  });
  
  elements.tabPanels.forEach(panel => {
    panel.classList.toggle('active', panel.id === tabId);
  });
  
  if (tabHeadings[tabId]) {
    elements.mainHeading.textContent = tabHeadings[tabId].title;
    elements.pageSubheading.textContent = tabHeadings[tabId].sub;
  }
  
  if (tabId === 'tab-gallery') {
    loadIdentities();
  }
}

function showToast(message, type = 'info') {
  const toast = document.createElement('div');
  toast.className = `toast ${type}`;
  const iconClass = type === 'success' ? 'ph-check-circle' : (type === 'error' ? 'ph-warning-octagon' : 'ph-info');
  toast.innerHTML = `<i class="ph-fill ${iconClass}"></i><span>${message}</span>`;
  elements.toastContainer.appendChild(toast);
  setTimeout(() => {
    toast.style.opacity = '0';
    setTimeout(() => toast.remove(), 300);
  }, 3500);
}

/* ==========================================================================
   API Calls & Status
   ========================================================================== */

async function loadSystemStatus() {
  try {
    const res = await fetch('/api/status');
    const data = await res.json();
    if (data.success) {
      elements.systemBackend.textContent = data.backend;
      elements.enrolledBadge.textContent = data.enrolled_count;
      state.threshold = data.threshold;
      elements.thresholdSlider.value = data.threshold;
      elements.sliderThresholdVal.textContent = data.threshold.toFixed(2);
      elements.topThreshDisplay.textContent = data.threshold.toFixed(2);
    }
  } catch (err) {
    console.error('Failed to load status:', err);
  }
}

async function loadSamples() {
  try {
    const res = await fetch('/api/samples');
    const data = await res.json();
    if (data.success && data.samples.length > 0) {
      state.samples = data.samples;
      renderSamples(data.samples);
    } else {
      elements.samplesContainer.innerHTML = '<span class="text-muted">No sample dataset images found.</span>';
    }
  } catch (err) {
    elements.samplesContainer.innerHTML = '<span class="text-muted">Failed to load samples.</span>';
  }
}

function renderSamples(samples) {
  elements.samplesContainer.innerHTML = '';
  samples.forEach(sample => {
    const chip = document.createElement('button');
    const isKnown = sample.category.includes('Known');
    chip.className = `sample-chip ${isKnown ? 'known' : 'unknown'}`;
    const cleanName = sample.person.replace(/_/g, ' ');
    chip.innerHTML = `<strong>${cleanName}</strong> <span style="opacity: 0.7">(${isKnown ? 'Known' : 'Unknown'})</span>`;
    chip.addEventListener('click', () => {
      identifyImageFromPath(sample.path, cleanName);
    });
    elements.samplesContainer.appendChild(chip);
  });
}

async function loadIdentities() {
  try {
    const res = await fetch('/api/identities');
    const data = await res.json();
    if (data.success) {
      state.enrolledIdentities = data.identities;
      elements.enrolledBadge.textContent = data.identities.length;
      renderGallery(data.identities);
    }
  } catch (err) {
    console.error('Failed to load identities:', err);
  }
}

function renderGallery(identities) {
  const searchTerm = (elements.gallerySearch.value || '').toLowerCase();
  elements.galleryCardsGrid.innerHTML = '';
  
  const filtered = identities.filter(id => id.name.toLowerCase().includes(searchTerm));
  
  if (filtered.length === 0) {
    elements.galleryCardsGrid.innerHTML = `
      <div class="empty-state" style="grid-column: 1/-1;">
        <i class="ph ph-user-minus empty-icon"></i>
        <h3>No Identities Found</h3>
        <p>No persons enrolled in the database matching your query.</p>
      </div>
    `;
    return;
  }
  
  filtered.forEach(id => {
    const card = document.createElement('div');
    card.className = 'identity-card';
    const initials = id.name.split('_').map(w => w[0]).join('').substring(0, 2).toUpperCase();
    const cleanName = id.name.replace(/_/g, ' ');
    
    card.innerHTML = `
      <div class="id-header">
        <div class="id-avatar">${initials}</div>
        <div>
          <div class="id-name">${cleanName}</div>
          <div class="id-meta">${id.num_images} enrolled photo(s)</div>
        </div>
      </div>
      <div class="id-actions">
        <button class="btn-danger-sm" data-name="${id.name}">
          <i class="ph ph-trash"></i> Delete
        </button>
      </div>
    `;
    
    card.querySelector('.btn-danger-sm').addEventListener('click', async (e) => {
      const name = e.currentTarget.getAttribute('data-name');
      if (confirm(`Are you sure you want to remove '${cleanName}' from the database?`)) {
        await removeIdentity(name);
      }
    });
    
    elements.galleryCardsGrid.appendChild(card);
  });
}

elements.gallerySearch.addEventListener('input', () => {
  renderGallery(state.enrolledIdentities);
});

async function removeIdentity(name) {
  try {
    const res = await fetch('/api/remove', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ name })
    });
    const data = await res.json();
    if (data.success) {
      showToast(`Removed '${name}' from database.`, 'success');
      await loadIdentities();
      await loadSystemStatus();
    } else {
      showToast(data.error || 'Failed to remove identity.', 'error');
    }
  } catch (err) {
    showToast('Network error removing identity.', 'error');
  }
}

async function loadMetricsAndFailures() {
  try {
    const res = await fetch('/api/metrics');
    const data = await res.json();
    if (data.success && data.failure_images) {
      renderFailures(data.failure_images);
    }
  } catch (err) {
    console.error('Failed to load metrics/failures:', err);
  }
}

function renderFailures(images) {
  elements.failuresGrid.innerHTML = '';
  if (images.length === 0) {
    elements.failuresGrid.innerHTML = '<p class="text-muted">No boundary failure artifacts recorded.</p>';
    return;
  }
  
  images.forEach(item => {
    const card = document.createElement('div');
    card.className = 'failure-card';
    
    let tagClass = 'tag-frr';
    let tagText = 'False Reject';
    let desc = 'Intra-class lighting/pose variation dropped similarity below strict threshold.';
    
    if (item.filename.includes('false_accept')) {
      tagClass = 'tag-far';
      tagText = 'False Accept (Lax τ)';
      desc = 'Unknown intruder matched nearest template when threshold was lowered.';
    } else if (item.filename.includes('no_face')) {
      tagClass = 'tag-noface';
      tagText = 'No Face Detected';
      desc = 'Extreme low light/motion blur prevented detector landmark extraction.';
    }
    
    const cleanTitle = item.filename.replace('.jpg', '').replace(/_/g, ' ');
    
    card.innerHTML = `
      <img src="${item.url}" alt="${cleanTitle}" loading="lazy">
      <div class="failure-card-body">
        <span class="failure-tag ${tagClass}">${tagText}</span>
        <div class="failure-title">${cleanTitle}</div>
        <p class="failure-desc">${desc}</p>
      </div>
    `;
    elements.failuresGrid.appendChild(card);
  });
}

/* ==========================================================================
   Image Input, Drag-and-Drop & Identification
   ========================================================================== */

function initDragAndDrop() {
  const dropzone = elements.dropzone;
  
  ['dragenter', 'dragover'].forEach(eventName => {
    dropzone.addEventListener(eventName, (e) => {
      e.preventDefault();
      dropzone.classList.add('drag-over');
    });
  });
  
  ['dragleave', 'drop'].forEach(eventName => {
    dropzone.addEventListener(eventName, (e) => {
      e.preventDefault();
      dropzone.classList.remove('drag-over');
    });
  });
  
  dropzone.addEventListener('drop', (e) => {
    const files = e.dataTransfer.files;
    if (files.length > 0 && files[0].type.startsWith('image/')) {
      processSelectedFile(files[0]);
    }
  });
  
  elements.btnBrowseFile.addEventListener('click', () => elements.fileInput.click());
  elements.fileInput.addEventListener('change', () => {
    if (elements.fileInput.files.length > 0) {
      processSelectedFile(elements.fileInput.files[0]);
    }
  });
  
  elements.btnClearCanvas.addEventListener('click', resetIdentifier);
}

function processSelectedFile(file) {
  const reader = new FileReader();
  reader.onload = (e) => {
    const imgDataUrl = e.target.result;
    const img = new Image();
    img.onload = () => {
      state.currentImage = img;
      identifyImageBase64(imgDataUrl);
    };
    img.src = imgDataUrl;
  };
  reader.readAsDataURL(file);
}

async function identifyImageFromPath(imagePath, sampleName) {
  showLoading(`Analyzing '${sampleName}' with ArcFace...`);
  
  const cleanPath = imagePath.replace(/\\/g, '/');
  // Also load image onto canvas for visual display
  const img = new Image();
  img.onload = async () => {
    state.currentImage = img;
    try {
      const res = await fetch('/api/identify', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          image_path: cleanPath,
          threshold: state.threshold
        })
      });
      const data = await res.json();
      hideLoading();
      
      if (data.success) {
        state.lastIdentifyResponse = data;
        displayResults(data);
      } else {
        showToast(data.error || 'Identification failed.', 'error');
      }
    } catch (err) {
      hideLoading();
      showToast('Network error during identification.', 'error');
    }
  };
  img.onerror = () => {
    hideLoading();
    showToast(`Could not load sample image for '${sampleName}'.`, 'error');
  };
  img.src = cleanPath.startsWith('/') ? cleanPath : '/' + cleanPath;
}

async function identifyImageBase64(base64Data) {
  showLoading('Detecting faces & extracting embeddings...');
  try {
    const res = await fetch('/api/identify', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        image: base64Data,
        threshold: state.threshold
      })
    });
    const data = await res.json();
    hideLoading();
    
    if (data.success) {
      state.lastIdentifyResponse = data;
      displayResults(data);
    } else {
      showToast(data.error || 'Identification failed.', 'error');
    }
  } catch (err) {
    hideLoading();
    showToast('Network error during identification.', 'error');
  }
}

function showLoading(msg = 'Processing...') {
  elements.loadingOverlay.classList.remove('hidden');
  document.getElementById('loading-text').textContent = msg;
}

function hideLoading() {
  elements.loadingOverlay.classList.add('hidden');
}

function resetIdentifier() {
  state.currentImage = null;
  state.lastIdentifyResponse = null;
  elements.dropzone.classList.remove('hidden');
  elements.canvasWrapper.classList.add('hidden');
  elements.emptyResultsState.classList.remove('hidden');
  elements.facesResultsList.classList.add('hidden');
  elements.detectedFacesPill.textContent = '0 Faces';
  const ctx = elements.faceCanvas.getContext('2d');
  ctx.clearRect(0, 0, elements.faceCanvas.width, elements.faceCanvas.height);
}

/* ==========================================================================
   Canvas Rendering & Live Dynamic Thresholding
   ========================================================================== */

function initThresholdSlider() {
  elements.thresholdSlider.addEventListener('input', (e) => {
    const val = parseFloat(e.target.value);
    state.threshold = val;
    elements.sliderThresholdVal.textContent = val.toFixed(2);
    elements.topThreshDisplay.textContent = val.toFixed(2);
    
    // Real-time client-side re-evaluation without network call!
    if (state.lastIdentifyResponse && state.currentImage) {
      reEvaluateThreshold(val);
    }
  });
}

function reEvaluateThreshold(newThresh) {
  if (!state.lastIdentifyResponse || !state.lastIdentifyResponse.faces) return;
  
  // Update face decision states
  state.lastIdentifyResponse.faces.forEach(face => {
    if (face.score >= newThresh) {
      face.name = face.best_match_person;
      face.is_known = true;
    } else {
      face.name = 'unknown';
      face.is_known = false;
    }
  });
  
  renderCanvas(state.lastIdentifyResponse.faces);
  renderResultsCards(state.lastIdentifyResponse.faces);
}

function displayResults(data) {
  elements.dropzone.classList.add('hidden');
  elements.canvasWrapper.classList.remove('hidden');
  elements.emptyResultsState.classList.add('hidden');
  elements.facesResultsList.classList.remove('hidden');
  
  const count = data.faces ? data.faces.length : 0;
  elements.detectedFacesPill.textContent = `${count} Face${count === 1 ? '' : 's'}`;
  
  renderCanvas(data.faces || []);
  renderResultsCards(data.faces || []);
}

function renderCanvas(faces) {
  if (!state.currentImage) return;
  
  const canvas = elements.faceCanvas;
  const ctx = canvas.getContext('2d');
  
  canvas.width = state.currentImage.naturalWidth || state.currentImage.width;
  canvas.height = state.currentImage.naturalHeight || state.currentImage.height;
  
  // Draw base image
  ctx.drawImage(state.currentImage, 0, 0, canvas.width, canvas.height);
  
  if (!faces || faces.length === 0) return;
  
  const fontScale = Math.max(12, Math.round(canvas.width / 35));
  ctx.font = `bold ${fontScale}px Inter, sans-serif`;
  
  faces.forEach((face, idx) => {
    const [x1, y1, x2, y2] = face.bbox;
    const w = x2 - x1;
    const h = y2 - y1;
    const isKnown = face.is_known;
    const color = isKnown ? '#10B981' : '#EF4444';
    
    // Draw Bounding Box
    ctx.lineWidth = Math.max(2, Math.round(canvas.width / 200));
    ctx.strokeStyle = color;
    ctx.strokeRect(x1, y1, w, h);
    
    // Prepare Label
    const cleanName = face.name === 'unknown' ? 'Unknown' : face.name.replace(/_/g, ' ');
    const label = `${cleanName} (${face.score.toFixed(2)})`;
    const textWidth = ctx.measureText(label).width;
    const padding = 6;
    
    // Label Background Pill
    const pillHeight = fontScale + padding * 2;
    const pillY = Math.max(0, y1 - pillHeight - 4);
    
    ctx.fillStyle = color;
    ctx.beginPath();
    ctx.roundRect(x1, pillY, textWidth + padding * 2, pillHeight, 4);
    ctx.fill();
    
    // Label Text
    ctx.fillStyle = '#FFFFFF';
    ctx.fillText(label, x1 + padding, pillY + fontScale);
  });
}

function renderResultsCards(faces) {
  elements.facesResultsList.innerHTML = '';
  
  if (faces.length === 0) {
    elements.facesResultsList.innerHTML = `
      <div class="empty-state">
        <i class="ph ph-warning-octagon empty-icon" style="color: var(--status-warning);"></i>
        <h3>No Face Detected</h3>
        <p>No valid face bounding box was found meeting the minimum confidence threshold.</p>
      </div>
    `;
    return;
  }
  
  faces.forEach((face, idx) => {
    const card = document.createElement('div');
    const isKnown = face.is_known;
    card.className = `face-match-card ${isKnown ? 'is-known' : 'is-unknown'}`;
    
    const cleanName = face.name === 'unknown' ? 'UNKNOWN (Rejected)' : face.name.replace(/_/g, ' ');
    const bestMatchClean = face.best_match_person ? face.best_match_person.replace(/_/g, ' ') : 'None';
    
    let candidatesHtml = '';
    if (face.top_candidates && face.top_candidates.length > 0) {
      candidatesHtml = `
        <div class="candidates-list">
          <span style="font-size: 11px; font-weight: 600; color: var(--text-muted); text-transform: uppercase;">Top Gallery Candidates:</span>
          ${face.top_candidates.map(c => `
            <div class="candidate-row">
              <div class="cand-name-score">
                <span>${c.name.replace(/_/g, ' ')}</span>
                <strong>${c.score.toFixed(3)}</strong>
              </div>
              <div class="cand-bar-bg">
                <div class="cand-bar-fill" style="width: ${Math.min(100, Math.max(0, c.score * 100))}%;"></div>
              </div>
            </div>
          `).join('')}
        </div>
      `;
    }
    
    card.innerHTML = `
      <div class="face-card-header">
        <div class="face-title">
          <i class="ph-fill ${isKnown ? 'ph-check-circle' : 'ph-x-circle'}" style="color: ${isKnown ? 'var(--status-success)' : 'var(--status-danger)'};"></i>
          <span>Face #${idx + 1}</span>
        </div>
        <span class="decision-badge ${isKnown ? 'verified' : 'rejected'}">
          ${isKnown ? 'VERIFIED MATCH' : 'UNKNOWN REJECTED'}
        </span>
      </div>
      
      <div class="similarity-metric-row">
        <div>
          <div style="font-size: 11px; color: var(--text-muted);">Assigned Identity</div>
          <strong style="font-size: 15px; color: var(--text-primary);">${cleanName}</strong>
        </div>
        <div style="text-align: right;">
          <div style="font-size: 11px; color: var(--text-muted);">Cosine Similarity</div>
          <strong style="font-size: 15px; color: ${isKnown ? 'var(--status-success)' : 'var(--status-danger)'};">${face.score.toFixed(3)}</strong>
        </div>
      </div>
      
      ${!isKnown ? `<div style="font-size: 12px; color: var(--text-muted); margin-bottom: 12px; padding: 6px 10px; background: rgba(239, 68, 68, 0.1); border-radius: 6px;">
        Intruder score (${face.score.toFixed(3)}) is below operating threshold (${state.threshold.toFixed(2)}). Nearest template: <strong>${bestMatchClean}</strong>.
      </div>` : ''}
      
      ${candidatesHtml}
    `;
    
    elements.facesResultsList.appendChild(card);
  });
}

/* ==========================================================================
   Enrollment Modal Logic
   ========================================================================== */

function initEnrollmentModal() {
  const openModal = () => {
    modalEnrollImages = [];
    elements.enrollPreviewGrid.innerHTML = '';
    elements.enrollNameInput.value = '';
    elements.enrollModal.classList.remove('hidden');
  };
  
  const closeModal = () => {
    elements.enrollModal.classList.add('hidden');
  };
  
  elements.btnOpenEnrollModal.addEventListener('click', openModal);
  elements.btnGalleryEnroll.addEventListener('click', openModal);
  elements.btnCloseEnroll.addEventListener('click', closeModal);
  elements.btnCancelEnroll.addEventListener('click', closeModal);
  
  elements.modalEnrollDropzone.addEventListener('click', () => elements.modalFileInput.click());
  
  elements.modalFileInput.addEventListener('change', () => {
    const files = Array.from(elements.modalFileInput.files);
    files.slice(0, 5).forEach(file => {
      const reader = new FileReader();
      reader.onload = (e) => {
        modalEnrollImages.push(e.target.result);
        const thumb = document.createElement('img');
        thumb.src = e.target.result;
        thumb.className = 'preview-thumb';
        elements.enrollPreviewGrid.appendChild(thumb);
      };
      reader.readAsDataURL(file);
    });
  });
  
  elements.btnSubmitEnroll.addEventListener('click', async () => {
    const name = elements.enrollNameInput.value.trim();
    if (!name) {
      showToast('Please enter an identity name.', 'error');
      return;
    }
    if (modalEnrollImages.length === 0) {
      showToast('Please select at least 1 photo.', 'error');
      return;
    }
    
    showToast('Extracting ArcFace embeddings for enrollment...', 'info');
    closeModal();
    
    try {
      const res = await fetch('/api/enroll', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          name: name,
          images: modalEnrollImages
        })
      });
      const data = await res.json();
      if (data.success) {
        showToast(`Successfully enrolled '${data.name}' with ${data.added_faces} faces!`, 'success');
        await loadIdentities();
        await loadSystemStatus();
      } else {
        showToast(data.error || 'Enrollment failed.', 'error');
      }
    } catch (err) {
      showToast('Network error during enrollment.', 'error');
    }
  });
}
