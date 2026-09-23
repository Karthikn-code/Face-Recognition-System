/**
 * FaceID Web Application - Client Controller
 */

// Application State
const state = {
  activeTab: 'tab-welcome',
  threshold: 0.60,
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
  
  // Welcome Portal Elements
  cardNewEnroll: document.getElementById('card-new-enroll'),
  cardAlreadyEnrolled: document.getElementById('card-already-enrolled'),
  btnCardEnroll: document.getElementById('btn-card-enroll'),
  btnCardVerify: document.getElementById('btn-card-verify'),
  portalEnrolledCount: document.getElementById('portal-enrolled-count'),
  portalTemplatesCount: document.getElementById('portal-templates-count'),
  linkOpenGallery: document.getElementById('link-open-gallery'),
  linkOpenBenchmark: document.getElementById('link-open-benchmark'),
  brandLogo: document.querySelector('.brand'),

  // Full-Page Dedicated Enrollment Elements
  btnBackToWelcome: document.getElementById('btn-back-to-welcome'),
  fullEnrollNameInput: document.getElementById('full-enroll-name-input'),
  btnFullModeCamera: document.getElementById('btn-full-mode-camera'),
  btnFullModeUpload: document.getElementById('btn-full-mode-upload'),
  fullEnrollCameraSection: document.getElementById('full-enroll-camera-section'),
  fullEnrollUploadSection: document.getElementById('full-enroll-upload-section'),
  fullEnrollVideoFeed: document.getElementById('full-enroll-video-feed'),
  fullEnrollSnapCanvas: document.getElementById('full-enroll-snap-canvas'),
  btnFullSnapPhoto: document.getElementById('btn-full-snap-photo'),
  fullSnapCount: document.getElementById('full-snap-count'),
  fullPreviewCount: document.getElementById('full-preview-count'),
  fullEnrollDropzone: document.getElementById('full-enroll-dropzone'),
  fullModalFileInput: document.getElementById('full-modal-file-input'),
  fullEnrollPreviewGrid: document.getElementById('full-enroll-preview-grid'),
  btnFullCancelEnroll: document.getElementById('btn-full-cancel-enroll'),
  btnFullSubmitEnroll: document.getElementById('btn-full-submit-enroll'),

  // Attendance Alert Banner Elements
  attendanceAlertBanner: document.getElementById('attendance-alert-banner'),
  attendanceAlertIcon: document.getElementById('attendance-alert-icon'),
  attendanceAlertTitle: document.getElementById('attendance-alert-title'),
  attendanceAlertDetails: document.getElementById('attendance-alert-details'),
  attendanceAlertTime: document.getElementById('attendance-alert-time'),

  // Identifier Elements
  dropzone: document.getElementById('image-dropzone'),
  canvasWrapper: document.getElementById('canvas-wrapper'),
  faceCanvas: document.getElementById('face-canvas'),
  fileInput: document.getElementById('file-input'),
  btnBrowseFile: document.getElementById('btn-browse-file'),
  btnOpenQueryCamera: document.getElementById('btn-open-query-camera'),
  webcamQueryWrapper: document.getElementById('webcam-query-wrapper'),
  webcamQueryVideo: document.getElementById('webcam-query-video'),
  btnSnapIdentify: document.getElementById('btn-snap-identify'),
  btnCloseQueryCamera: document.getElementById('btn-close-query-camera'),
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
  
  // Camera Enrollment Elements
  btnModeUpload: document.getElementById('btn-mode-upload'),
  btnModeCamera: document.getElementById('btn-mode-camera'),
  enrollUploadSection: document.getElementById('enroll-upload-section'),
  enrollCameraSection: document.getElementById('enroll-camera-section'),
  enrollVideoFeed: document.getElementById('enroll-video-feed'),
  enrollSnapCanvas: document.getElementById('enroll-snap-canvas'),
  btnSnapPhoto: document.getElementById('btn-snap-photo'),
  snapCount: document.getElementById('snap-count'),
  
  // Failures Elements
  failuresGrid: document.getElementById('failures-grid'),
  
  // Toast
  toastContainer: document.getElementById('toast-container')
};

// Headings by Tab
const tabHeadings = {
  'tab-welcome': {
    title: 'Welcome to FaceID.ai',
    sub: 'Enterprise Biometric Face Recognition, Verification & Attendance System powered by ArcFace.'
  },
  'tab-enroll': {
    title: 'New Identity Enrollment',
    sub: 'Register a new identity profile with up to 5 photos via live webcam or file upload.'
  },
  'tab-identify': {
    title: 'Live Face Identification & Attendance',
    sub: 'Detect facial landmarks, extract ArcFace embeddings, verify identity, and record attendance.'
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
let queryCameraStream = null;

/* ==========================================================================
   Initialization & Navigation
   ========================================================================== */

document.addEventListener('DOMContentLoaded', async () => {
  initThemeSwitcher();
  initNavigation();
  initWelcomePortal();
  initFullEnrollment();
  initDragAndDrop();
  initThresholdSlider();
  initEnrollmentModal();

  // Set initial tab state to welcome screen
  switchTab('tab-welcome');
  
  // Load initial server data
  await loadSystemStatus();
  await loadSamples();
  await loadIdentities();
  await loadMetricsAndFailures();
});

const themeLabels = {
  'dark': 'Dark',
  'obsidian': 'Dark',
  'light': 'Light'
};

function initThemeSwitcher() {
  let savedTheme = localStorage.getItem('faceid-theme') || 'dark';
  // Normalize legacy themes to dark/light
  if (savedTheme !== 'light') {
    savedTheme = 'dark';
  }
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

  // Re-query theme menu items in case of DOM updates
  const items = document.querySelectorAll('.theme-menu-item');
  items.forEach(item => {
    item.addEventListener('click', (e) => {
      e.stopPropagation();
      const theme = item.getAttribute('data-theme') || 'dark';
      setTheme(theme);
      if (elements.themeDropdownMenu) {
        elements.themeDropdownMenu.classList.add('hidden');
      }
      showToast(`Switched to ${theme === 'light' ? 'Light' : 'Dark'} Mode!`, 'info');
    });
  });
}

function setTheme(theme) {
  const activeTheme = theme === 'light' ? 'light' : 'dark';
  document.body.setAttribute('data-theme', activeTheme);
  localStorage.setItem('faceid-theme', activeTheme);

  if (elements.activeThemeName) {
    elements.activeThemeName.textContent = activeTheme === 'light' ? 'Light' : 'Dark';
  }

  const themeIcon = document.getElementById('theme-icon');
  if (themeIcon) {
    themeIcon.className = activeTheme === 'light' ? 'ph ph-sun' : 'ph ph-moon';
    themeIcon.style.color = activeTheme === 'light' ? '#F59E0B' : '#818CF8';
  }

  const items = document.querySelectorAll('.theme-menu-item');
  items.forEach(item => {
    const itemTheme = item.getAttribute('data-theme');
    const isMatch = itemTheme === activeTheme;
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

  // Deactivate query camera if switching away from live identifier tab
  if (tabId !== 'tab-identify') {
    closeQueryCamera();
  }

  // Control full-page enrollment camera stream
  if (tabId !== 'tab-enroll') {
    stopFullCameraStream();
  } else {
    onOpenFullEnroll();
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
      if (elements.portalEnrolledCount) elements.portalEnrolledCount.textContent = data.enrolled_count;
      if (elements.portalTemplatesCount) elements.portalTemplatesCount.textContent = data.total_templates;
      const startingThresh = 0.60;
      state.threshold = startingThresh;
      elements.thresholdSlider.value = startingThresh;
      elements.sliderThresholdVal.textContent = startingThresh.toFixed(2);
      elements.topThreshDisplay.textContent = startingThresh.toFixed(2);
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
  
  const mobileCameraInput = document.getElementById('mobile-camera-input');
  if (mobileCameraInput) {
    mobileCameraInput.addEventListener('change', () => {
      if (mobileCameraInput.files.length > 0) {
        processSelectedFile(mobileCameraInput.files[0]);
      }
    });
  }

  elements.btnClearCanvas.addEventListener('click', resetIdentifier);

  // Live Query Webcam Listeners
  if (elements.btnOpenQueryCamera) {
    elements.btnOpenQueryCamera.addEventListener('click', openQueryCamera);
  }
  if (elements.btnCloseQueryCamera) {
    elements.btnCloseQueryCamera.addEventListener('click', closeQueryCamera);
  }
  if (elements.btnSnapIdentify) {
    elements.btnSnapIdentify.addEventListener('click', snapAndIdentifyQuery);
  }
}

/* ==========================================================================
   Query Webcam Capture Logic
   ========================================================================== */

async function openQueryCamera() {
  const mobileCameraInput = document.getElementById('mobile-camera-input');

  // Mobile browsers strictly block WebRTC getUserMedia on non-HTTPS IP origins.
  // Fall back smoothly to native phone camera capture (front selfie)
  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    if (mobileCameraInput) {
      mobileCameraInput.click();
      return;
    }
  }

  try {
    if (queryCameraStream) closeQueryCamera();
    queryCameraStream = await navigator.mediaDevices.getUserMedia({
      video: { width: { ideal: 1280 }, height: { ideal: 720 }, facingMode: 'user' }
    });
    if (elements.webcamQueryVideo) {
      elements.webcamQueryVideo.srcObject = queryCameraStream;
    }
    elements.dropzone.classList.add('hidden');
    elements.webcamQueryWrapper.classList.remove('hidden');
  } catch (err) {
    console.warn('WebRTC camera unavailable, opening native mobile camera intent:', err);
    if (mobileCameraInput) {
      mobileCameraInput.click();
    } else {
      showToast('Camera access denied or unavailable: ' + (err.message || err.name), 'error');
    }
  }
}

function closeQueryCamera() {
  if (queryCameraStream) {
    queryCameraStream.getTracks().forEach(track => track.stop());
    queryCameraStream = null;
  }
  if (elements.webcamQueryVideo) {
    elements.webcamQueryVideo.srcObject = null;
  }
  if (elements.webcamQueryWrapper) {
    elements.webcamQueryWrapper.classList.add('hidden');
  }
  // If no analyzed image is displayed, restore dropzone
  if (!state.currentImage && elements.dropzone) {
    elements.dropzone.classList.remove('hidden');
  }
}

function snapAndIdentifyQuery() {
  const video = elements.webcamQueryVideo;
  if (!video || !video.videoWidth) {
    showToast('Waiting for live camera feed...', 'error');
    return;
  }

  // Create offscreen canvas to capture current video frame
  const snapCanvas = document.createElement('canvas');
  snapCanvas.width = video.videoWidth;
  snapCanvas.height = video.videoHeight;
  const ctx = snapCanvas.getContext('2d');

  // Mirror horizontally so the snapped frame aligns with mirrored user preview
  ctx.translate(snapCanvas.width, 0);
  ctx.scale(-1, 1);
  ctx.drawImage(video, 0, 0, snapCanvas.width, snapCanvas.height);
  ctx.setTransform(1, 0, 0, 1, 0, 0);

  const dataUrl = snapCanvas.toDataURL('image/jpeg', 0.95);

  // Stop camera stream immediately
  closeQueryCamera();

  // Load into currentImage and trigger identification
  const img = new Image();
  img.onload = () => {
    state.currentImage = img;
    identifyImageBase64(dataUrl);
  };
  img.src = dataUrl;
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
  closeQueryCamera();
  state.currentImage = null;
  state.lastIdentifyResponse = null;
  if (elements.attendanceAlertBanner) {
    elements.attendanceAlertBanner.classList.add('hidden');
  }
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
  
  // Update attendance banner dynamically with threshold change
  if (elements.attendanceAlertBanner) {
    const knownFace = state.lastIdentifyResponse.faces.find(f => f.is_known);
    const timeStr = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });
    if (knownFace) {
      elements.attendanceAlertBanner.className = 'attendance-alert-banner success';
      elements.attendanceAlertIcon.innerHTML = '<i class="ph-fill ph-check-circle"></i>';
      elements.attendanceAlertTitle.textContent = `Attendance Verified: ${knownFace.name.replace(/_/g, ' ')}`;
      elements.attendanceAlertDetails.textContent = `Matched biometric template with ${(knownFace.score * 100).toFixed(1)}% cosine similarity (τ: ${newThresh.toFixed(2)}).`;
      elements.attendanceAlertTime.textContent = timeStr;
    } else {
      elements.attendanceAlertBanner.className = 'attendance-alert-banner warning';
      elements.attendanceAlertIcon.innerHTML = '<i class="ph-fill ph-warning-octagon"></i>';
      elements.attendanceAlertTitle.textContent = 'Access Denied / Unrecognized Person';
      elements.attendanceAlertDetails.textContent = `Candidate score (${(state.lastIdentifyResponse.faces[0].score * 100).toFixed(1)}%) is below operational threshold (τ: ${newThresh.toFixed(2)}).`;
      elements.attendanceAlertTime.textContent = timeStr;
    }
  }

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
  
  // Attendance Verification Banner Update
  if (elements.attendanceAlertBanner) {
    if (data.faces && data.faces.length > 0) {
      const knownFace = data.faces.find(f => f.is_known);
      elements.attendanceAlertBanner.classList.remove('hidden');
      const timeStr = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });

      if (knownFace) {
        elements.attendanceAlertBanner.className = 'attendance-alert-banner success';
        elements.attendanceAlertIcon.innerHTML = '<i class="ph-fill ph-check-circle"></i>';
        elements.attendanceAlertTitle.textContent = `Attendance Verified: ${knownFace.name.replace(/_/g, ' ')}`;
        elements.attendanceAlertDetails.textContent = `Matched biometric template with ${(knownFace.score * 100).toFixed(1)}% cosine similarity (τ: ${state.threshold.toFixed(2)}).`;
        elements.attendanceAlertTime.textContent = timeStr;
      } else {
        elements.attendanceAlertBanner.className = 'attendance-alert-banner warning';
        elements.attendanceAlertIcon.innerHTML = '<i class="ph-fill ph-warning-octagon"></i>';
        elements.attendanceAlertTitle.textContent = 'Access Denied / Unrecognized Person';
        elements.attendanceAlertDetails.textContent = `Candidate score (${(data.faces[0].score * 100).toFixed(1)}%) is below operational threshold (τ: ${state.threshold.toFixed(2)}).`;
        elements.attendanceAlertTime.textContent = timeStr;
      }
    } else {
      elements.attendanceAlertBanner.classList.add('hidden');
    }
  }

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

let cameraStream = null;

function initEnrollmentModal() {
  const openModal = () => {
    modalEnrollImages = [];
    renderPreviewThumbnails();
    elements.enrollNameInput.value = '';
    setEnrollMode('upload');
    elements.enrollModal.classList.remove('hidden');
  };
  
  const closeModal = () => {
    stopCameraStream();
    modalEnrollImages = [];
    renderPreviewThumbnails();
    elements.enrollModal.classList.add('hidden');
  };

  function setEnrollMode(mode) {
    if (mode === 'camera') {
      elements.btnModeCamera.classList.add('active');
      elements.btnModeUpload.classList.remove('active');
      elements.enrollUploadSection.classList.add('hidden');
      elements.enrollCameraSection.classList.remove('hidden');
      startCameraStream();
    } else {
      elements.btnModeUpload.classList.add('active');
      elements.btnModeCamera.classList.remove('active');
      elements.enrollCameraSection.classList.add('hidden');
      elements.enrollUploadSection.classList.remove('hidden');
      stopCameraStream();
    }
  }

  async function startCameraStream() {
    try {
      if (cameraStream) stopCameraStream();
      cameraStream = await navigator.mediaDevices.getUserMedia({
        video: { width: { ideal: 640 }, height: { ideal: 480 }, facingMode: 'user' }
      });
      if (elements.enrollVideoFeed) {
        elements.enrollVideoFeed.srcObject = cameraStream;
      }
    } catch (err) {
      console.error('Camera stream error:', err);
      showToast('Camera access denied or unavailable: ' + (err.message || err.name), 'error');
      setEnrollMode('upload');
    }
  }

  function stopCameraStream() {
    if (cameraStream) {
      cameraStream.getTracks().forEach(track => track.stop());
      cameraStream = null;
    }
    if (elements.enrollVideoFeed) {
      elements.enrollVideoFeed.srcObject = null;
    }
  }

  function renderPreviewThumbnails() {
    elements.enrollPreviewGrid.innerHTML = '';
    if (elements.snapCount) {
      elements.snapCount.textContent = modalEnrollImages.length;
    }
    modalEnrollImages.forEach((imgData, idx) => {
      const wrapper = document.createElement('div');
      wrapper.className = 'preview-thumb-wrapper';
      
      const thumb = document.createElement('img');
      thumb.src = imgData;
      thumb.className = 'preview-thumb';
      thumb.title = `Photo #${idx + 1}`;
      
      const removeBtn = document.createElement('button');
      removeBtn.className = 'btn-remove-thumb';
      removeBtn.innerHTML = '×';
      removeBtn.title = 'Remove photo';
      removeBtn.type = 'button';
      removeBtn.addEventListener('click', (e) => {
        e.stopPropagation();
        modalEnrollImages.splice(idx, 1);
        renderPreviewThumbnails();
      });
      
      wrapper.appendChild(thumb);
      wrapper.appendChild(removeBtn);
      elements.enrollPreviewGrid.appendChild(wrapper);
    });
  }
  
  elements.btnOpenEnrollModal.addEventListener('click', openModal);
  elements.btnGalleryEnroll.addEventListener('click', openModal);
  elements.btnCloseEnroll.addEventListener('click', closeModal);
  elements.btnCancelEnroll.addEventListener('click', closeModal);

  // Mode switcher listeners
  if (elements.btnModeUpload) {
    elements.btnModeUpload.addEventListener('click', () => setEnrollMode('upload'));
  }
  if (elements.btnModeCamera) {
    elements.btnModeCamera.addEventListener('click', () => setEnrollMode('camera'));
  }

  // Camera snap photo listener
  if (elements.btnSnapPhoto) {
    elements.btnSnapPhoto.addEventListener('click', () => {
      if (modalEnrollImages.length >= 5) {
        showToast('Maximum 5 enrollment photos allowed.', 'error');
        return;
      }
      const video = elements.enrollVideoFeed;
      if (!video || !video.videoWidth) {
        showToast('Waiting for live camera stream...', 'error');
        return;
      }
      const canvas = elements.enrollSnapCanvas;
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
      const ctx = canvas.getContext('2d');
      // Mirror image for natural appearance
      ctx.translate(canvas.width, 0);
      ctx.scale(-1, 1);
      ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
      ctx.setTransform(1, 0, 0, 1, 0, 0);
      
      const dataUrl = canvas.toDataURL('image/jpeg', 0.95);
      modalEnrollImages.push(dataUrl);
      renderPreviewThumbnails();
      showToast(`Captured enrollment photo (${modalEnrollImages.length}/5)!`, 'success');
    });
  }
  
  elements.modalEnrollDropzone.addEventListener('click', () => elements.modalFileInput.click());
  
  elements.modalFileInput.addEventListener('change', () => {
    const files = Array.from(elements.modalFileInput.files);
    const availableSlots = 5 - modalEnrollImages.length;
    if (availableSlots <= 0) {
      showToast('Maximum 5 enrollment photos allowed.', 'error');
      return;
    }
    files.slice(0, availableSlots).forEach(file => {
      const reader = new FileReader();
      reader.onload = (e) => {
        modalEnrollImages.push(e.target.result);
        renderPreviewThumbnails();
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
      showToast('Please upload or snap at least 1 photo.', 'error');
      return;
    }
    
    // Copy images so they are not wiped if modal resets
    const imagesToEnroll = [...modalEnrollImages];
    const originalBtnHtml = elements.btnSubmitEnroll.innerHTML;
    elements.btnSubmitEnroll.disabled = true;
    elements.btnSubmitEnroll.innerHTML = '<i class="ph ph-spinner ph-spin"></i> Enrolling...';

    showToast('Extracting ArcFace embeddings for enrollment...', 'info');
    
    try {
      const res = await fetch('/api/enroll', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          name: name,
          images: imagesToEnroll
        })
      });
      const data = await res.json();
      if (data.success) {
        showToast(`Successfully enrolled '${data.name}' with ${data.added_faces} face templates!`, 'success');
        closeModal();
        await loadIdentities();
        await loadSystemStatus();
      } else {
        showToast(data.error || 'Enrollment failed.', 'error');
      }
    } catch (err) {
      showToast('Network error during enrollment: ' + (err.message || err), 'error');
    } finally {
      elements.btnSubmitEnroll.disabled = false;
      elements.btnSubmitEnroll.innerHTML = originalBtnHtml;
    }
  });
}

/* ==========================================================================
   Welcome Portal Logic
   ========================================================================== */

function initWelcomePortal() {
  if (elements.cardNewEnroll) {
    elements.cardNewEnroll.addEventListener('click', () => switchTab('tab-enroll'));
  }
  if (elements.btnCardEnroll) {
    elements.btnCardEnroll.addEventListener('click', (e) => {
      e.stopPropagation();
      switchTab('tab-enroll');
    });
  }
  if (elements.cardAlreadyEnrolled) {
    elements.cardAlreadyEnrolled.addEventListener('click', () => switchTab('tab-identify'));
  }
  if (elements.btnCardVerify) {
    elements.btnCardVerify.addEventListener('click', (e) => {
      e.stopPropagation();
      switchTab('tab-identify');
    });
  }
  if (elements.linkOpenGallery) {
    elements.linkOpenGallery.addEventListener('click', () => switchTab('tab-gallery'));
  }
  if (elements.linkOpenBenchmark) {
    elements.linkOpenBenchmark.addEventListener('click', () => switchTab('tab-benchmark'));
  }
  if (elements.brandLogo) {
    elements.brandLogo.style.cursor = 'pointer';
    elements.brandLogo.addEventListener('click', () => switchTab('tab-welcome'));
  }
}

/* ==========================================================================
   Dedicated Full-Page Enrollment Controller
   ========================================================================== */

let fullEnrollImages = [];
let fullCameraStream = null;
let fullEnrollMode = 'camera';

function onOpenFullEnroll() {
  setFullEnrollMode('camera');
}

function initFullEnrollment() {
  if (elements.btnBackToWelcome) {
    elements.btnBackToWelcome.addEventListener('click', () => switchTab('tab-welcome'));
  }
  if (elements.btnFullCancelEnroll) {
    elements.btnFullCancelEnroll.addEventListener('click', () => switchTab('tab-welcome'));
  }

  // Full-page mode switcher
  if (elements.btnFullModeCamera) {
    elements.btnFullModeCamera.addEventListener('click', () => setFullEnrollMode('camera'));
  }
  if (elements.btnFullModeUpload) {
    elements.btnFullModeUpload.addEventListener('click', () => setFullEnrollMode('upload'));
  }

  // Camera Snap button
  if (elements.btnFullSnapPhoto) {
    elements.btnFullSnapPhoto.addEventListener('click', () => {
      if (fullEnrollImages.length >= 5) {
        showToast('Maximum 5 enrollment photos allowed.', 'error');
        return;
      }
      const video = elements.fullEnrollVideoFeed;
      const mobileEnrollInput = document.getElementById('mobile-enroll-camera-input');

      // If WebRTC live video is unavailable (e.g. mobile browser over HTTP), open phone camera
      if (!video || !video.videoWidth || !fullCameraStream) {
        if (mobileEnrollInput) {
          mobileEnrollInput.click();
          return;
        }
        showToast('Waiting for live camera stream...', 'error');
        return;
      }

      const canvas = elements.fullEnrollSnapCanvas;
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
      const ctx = canvas.getContext('2d');
      // Mirror image for natural user experience
      ctx.translate(canvas.width, 0);
      ctx.scale(-1, 1);
      ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
      ctx.setTransform(1, 0, 0, 1, 0, 0);

      const dataUrl = canvas.toDataURL('image/jpeg', 0.95);
      fullEnrollImages.push(dataUrl);
      renderFullPreviewThumbnails();
      showToast(`Captured enrollment photo (${fullEnrollImages.length}/5)!`, 'success');
    });
  }

  // Native phone camera capture handler for mobile enrollment
  const mobileEnrollInput = document.getElementById('mobile-enroll-camera-input');
  if (mobileEnrollInput) {
    mobileEnrollInput.addEventListener('change', () => {
      if (mobileEnrollInput.files && mobileEnrollInput.files[0]) {
        if (fullEnrollImages.length >= 5) {
          showToast('Maximum 5 enrollment photos allowed.', 'error');
          return;
        }
        const reader = new FileReader();
        reader.onload = (e) => {
          fullEnrollImages.push(e.target.result);
          renderFullPreviewThumbnails();
          showToast(`Captured enrollment photo (${fullEnrollImages.length}/5)!`, 'success');
        };
        reader.readAsDataURL(mobileEnrollInput.files[0]);
      }
    });
  }

  // Upload Dropzone
  if (elements.fullEnrollDropzone) {
    elements.fullEnrollDropzone.addEventListener('click', () => elements.fullModalFileInput.click());
  }

  if (elements.fullModalFileInput) {
    elements.fullModalFileInput.addEventListener('change', () => {
      const files = Array.from(elements.fullModalFileInput.files);
      const availableSlots = 5 - fullEnrollImages.length;
      if (availableSlots <= 0) {
        showToast('Maximum 5 enrollment photos allowed.', 'error');
        return;
      }
      files.slice(0, availableSlots).forEach(file => {
        const reader = new FileReader();
        reader.onload = (e) => {
          fullEnrollImages.push(e.target.result);
          renderFullPreviewThumbnails();
        };
        reader.readAsDataURL(file);
      });
    });
  }

  // Submit Enrollment
  if (elements.btnFullSubmitEnroll) {
    elements.btnFullSubmitEnroll.addEventListener('click', async () => {
      const name = (elements.fullEnrollNameInput.value || '').trim();
      if (!name) {
        showToast('Please enter an identity name.', 'error');
        elements.fullEnrollNameInput.focus();
        return;
      }
      if (fullEnrollImages.length === 0) {
        showToast('Please capture or upload at least 1 photo.', 'error');
        return;
      }

      const imagesToEnroll = [...fullEnrollImages];
      const originalBtnHtml = elements.btnFullSubmitEnroll.innerHTML;
      elements.btnFullSubmitEnroll.disabled = true;
      elements.btnFullSubmitEnroll.innerHTML = '<i class="ph ph-spinner ph-spin"></i> Enrolling...';

      showToast('Extracting ArcFace embeddings for enrollment...', 'info');

      try {
        const res = await fetch('/api/enroll', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            name: name,
            images: imagesToEnroll
          })
        });
        const data = await res.json();
        if (data.success) {
          showToast(`Successfully enrolled '${data.name}' with ${data.added_faces} face templates!`, 'success');
          // Reset form
          stopFullCameraStream();
          fullEnrollImages = [];
          elements.fullEnrollNameInput.value = '';
          renderFullPreviewThumbnails();
          await loadIdentities();
          await loadSystemStatus();
          // Navigate to gallery to view new record
          switchTab('tab-gallery');
        } else {
          showToast(data.error || 'Enrollment failed.', 'error');
        }
      } catch (err) {
        showToast('Network error during enrollment: ' + (err.message || err), 'error');
      } finally {
        elements.btnFullSubmitEnroll.disabled = false;
        elements.btnFullSubmitEnroll.innerHTML = originalBtnHtml;
      }
    });
  }
}

function setFullEnrollMode(mode) {
  fullEnrollMode = mode;
  if (mode === 'camera') {
    if (elements.btnFullModeCamera) elements.btnFullModeCamera.classList.add('active');
    if (elements.btnFullModeUpload) elements.btnFullModeUpload.classList.remove('active');
    if (elements.fullEnrollUploadSection) elements.fullEnrollUploadSection.classList.add('hidden');
    if (elements.fullEnrollCameraSection) elements.fullEnrollCameraSection.classList.remove('hidden');
    startFullCameraStream();
  } else {
    if (elements.btnFullModeUpload) elements.btnFullModeUpload.classList.add('active');
    if (elements.btnFullModeCamera) elements.btnFullModeCamera.classList.remove('active');
    if (elements.fullEnrollCameraSection) elements.fullEnrollCameraSection.classList.add('hidden');
    if (elements.fullEnrollUploadSection) elements.fullEnrollUploadSection.classList.remove('hidden');
    stopFullCameraStream();
  }
}

async function startFullCameraStream() {
  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    if (elements.btnFullSnapPhoto) {
      elements.btnFullSnapPhoto.innerHTML = `<i class="ph ph-camera"></i> Snap with Phone Camera (<span id="full-snap-count">${fullEnrollImages.length}</span>/5)`;
    }
    return;
  }
  try {
    if (fullCameraStream) stopFullCameraStream();
    fullCameraStream = await navigator.mediaDevices.getUserMedia({
      video: { width: { ideal: 640 }, height: { ideal: 480 }, facingMode: 'user' }
    });
    if (elements.fullEnrollVideoFeed) {
      elements.fullEnrollVideoFeed.srcObject = fullCameraStream;
    }
  } catch (err) {
    console.warn('Full enrollment WebRTC stream unavailable, enabling direct phone snap:', err);
    if (elements.btnFullSnapPhoto) {
      elements.btnFullSnapPhoto.innerHTML = `<i class="ph ph-camera"></i> Snap with Phone Camera (<span id="full-snap-count">${fullEnrollImages.length}</span>/5)`;
    }
  }
}

function stopFullCameraStream() {
  if (fullCameraStream) {
    fullCameraStream.getTracks().forEach(track => track.stop());
    fullCameraStream = null;
  }
  if (elements.fullEnrollVideoFeed) {
    elements.fullEnrollVideoFeed.srcObject = null;
  }
}

function renderFullPreviewThumbnails() {
  if (!elements.fullEnrollPreviewGrid) return;
  elements.fullEnrollPreviewGrid.innerHTML = '';
  if (elements.fullSnapCount) elements.fullSnapCount.textContent = fullEnrollImages.length;
  if (elements.fullPreviewCount) elements.fullPreviewCount.textContent = fullEnrollImages.length;

  fullEnrollImages.forEach((imgData, idx) => {
    const wrapper = document.createElement('div');
    wrapper.className = 'preview-thumb-wrapper';

    const thumb = document.createElement('img');
    thumb.src = imgData;
    thumb.className = 'preview-thumb';
    thumb.title = `Photo #${idx + 1}`;

    const removeBtn = document.createElement('button');
    removeBtn.className = 'btn-remove-thumb';
    removeBtn.innerHTML = '×';
    removeBtn.title = 'Remove photo';
    removeBtn.type = 'button';
    removeBtn.addEventListener('click', (e) => {
      e.stopPropagation();
      fullEnrollImages.splice(idx, 1);
      renderFullPreviewThumbnails();
    });

    wrapper.appendChild(thumb);
    wrapper.appendChild(removeBtn);
    elements.fullEnrollPreviewGrid.appendChild(wrapper);
  });
}

