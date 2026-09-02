/**
 * SatQuery.ai — Interactive Frontend, Multi-Route SPA Router, & Three.js Celestial Background
 */

// ============================================================================
// 1. Three.js Celestial Background Animation
// ============================================================================
(function initCelestialBackground() {
  const canvas = document.getElementById('sky');
  if (!canvas || typeof THREE === 'undefined') {
    console.warn('Three.js or #sky canvas not found.');
    return;
  }

  const renderer = new THREE.WebGLRenderer({ canvas: canvas, alpha: true, antialias: true });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  renderer.setSize(window.innerWidth, window.innerHeight);

  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(45, window.innerWidth / window.innerHeight, 0.1, 200);
  camera.position.set(0, 14, 34);
  camera.lookAt(0, 0, 0);

  // Starfield
  const starGeo = new THREE.BufferGeometry();
  const starCount = 900;
  const positions = new Float32Array(starCount * 3);
  for (let i = 0; i < starCount; i++) {
    positions[i * 3] = (Math.random() - 0.5) * 160;
    positions[i * 3 + 1] = (Math.random() - 0.5) * 160;
    positions[i * 3 + 2] = (Math.random() - 0.5) * 160;
  }
  starGeo.setAttribute('position', new THREE.BufferAttribute(positions, 3));
  const starMat = new THREE.PointsMaterial({ color: 0xffffff, size: 0.18, transparent: true, opacity: 0.7 });
  scene.add(new THREE.Points(starGeo, starMat));

  const solarSystem = new THREE.Group();
  scene.add(solarSystem);

  // Glow Sprite behind Sun
  function makeGlowTexture() {
    const c = document.createElement('canvas');
    c.width = c.height = 256;
    const ctx = c.getContext('2d');
    const g = ctx.createRadialGradient(128, 128, 0, 128, 128, 128);
    g.addColorStop(0, 'rgba(255,180,90,0.9)');
    g.addColorStop(0.4, 'rgba(255,120,40,0.35)');
    g.addColorStop(1, 'rgba(255,120,40,0)');
    ctx.fillStyle = g;
    ctx.fillRect(0, 0, 256, 256);
    return new THREE.CanvasTexture(c);
  }

  const glow = new THREE.Sprite(new THREE.SpriteMaterial({ map: makeGlowTexture(), transparent: true, depthWrite: false }));
  glow.scale.set(18, 18, 1);
  solarSystem.add(glow);

  // Sun
  const sun = new THREE.Mesh(
    new THREE.SphereGeometry(2.1, 32, 32),
    new THREE.MeshBasicMaterial({ color: 0xffb35c })
  );
  solarSystem.add(sun);
  const sunLight = new THREE.PointLight(0xffcf9e, 2.2, 100);
  solarSystem.add(sunLight);
  scene.add(new THREE.AmbientLight(0x404050, 0.6));

  const planetDefs = [
    { r: 0.35, dist: 4.2,  speed: 1.4,  color: 0x9aa4b8 },
    { r: 0.55, dist: 6.4,  speed: 1.0,  color: 0x5eead4 },
    { r: 0.5,  dist: 8.8,  speed: 0.72, color: 0x4f8cff },
    { r: 0.75, dist: 11.6, speed: 0.5,  color: 0xe8622c },
    { r: 0.9,  dist: 14.8, speed: 0.34, color: 0xd8c39a }
  ];

  const planets = planetDefs.map(def => {
    const orbitPts = [];
    const seg = 128;
    for (let i = 0; i <= seg; i++) {
      const a = (i / seg) * Math.PI * 2;
      orbitPts.push(new THREE.Vector3(Math.cos(a) * def.dist, 0, Math.sin(a) * def.dist));
    }
    const orbitGeo = new THREE.BufferGeometry().setFromPoints(orbitPts);
    const orbitMat = new THREE.LineBasicMaterial({ color: 0x3a4157, transparent: true, opacity: 0.5 });
    solarSystem.add(new THREE.LineLoop(orbitGeo, orbitMat));

    const mesh = new THREE.Mesh(
      new THREE.SphereGeometry(def.r, 24, 24),
      new THREE.MeshStandardMaterial({ color: def.color, roughness: 0.6, metalness: 0.1 })
    );
    solarSystem.add(mesh);
    return { mesh, dist: def.dist, speed: def.speed, angle: Math.random() * Math.PI * 2 };
  });

  solarSystem.rotation.x = 0.35;
  const clock = new THREE.Clock();

  function render() {
    const dt = clock.getDelta();
    planets.forEach(p => {
      p.angle += dt * p.speed * 0.35;
      p.mesh.position.set(Math.cos(p.angle) * p.dist, 0, Math.sin(p.angle) * p.dist);
      p.mesh.rotation.y += dt * 0.6;
    });
    sun.rotation.y += dt * 0.15;

    renderer.render(scene, camera);
    requestAnimationFrame(render);
  }
  render();

  function onScroll() {
    const max = document.body.scrollHeight - window.innerHeight;
    const t = max > 0 ? window.scrollY / max : 0;
    solarSystem.rotation.y = t * Math.PI * 1.4;
    solarSystem.rotation.z = t * 0.25;
    camera.position.y = 14 - t * 10;
    camera.position.z = 34 - t * 14;
    camera.lookAt(0, 0, 0);
  }
  window.addEventListener('scroll', onScroll, { passive: true });
  onScroll();

  window.addEventListener('resize', () => {
    camera.aspect = window.innerWidth / window.innerHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(window.innerWidth, window.innerHeight);
  });
})();


// ============================================================================
// 2. SatQuery AI Multi-Route SPA & Agentic Logic
// ============================================================================
document.addEventListener('DOMContentLoaded', () => {
  let currentMode = 'vqa'; // 'vqa', 'grounding', 'bitemporal_change', 'optical_sar_fusion'
  let cachedBoxes = [];
  let executionHistory = [];

  // DOM Form & Input Elements
  const vqaForm = document.getElementById('vqaForm');
  const imageInput = document.getElementById('imageInput');
  const secondaryImageInput = document.getElementById('secondaryImageInput');
  const dropZone = document.getElementById('dropZone');
  const dropZone2 = document.getElementById('dropZone2');
  const secondaryGroup = document.getElementById('secondaryGroup');
  const questionInput = document.getElementById('questionInput');
  const btnAnalyse = document.getElementById('btnAnalyse');
  const spinner = document.getElementById('spinner');
  const alertBox = document.getElementById('alertBox');
  
  const fileInfo = document.getElementById('fileInfo');
  const fileName = document.getElementById('fileName');
  const btnClearFile = document.getElementById('btnClearFile');

  const fileInfo2 = document.getElementById('fileInfo2');
  const fileName2 = document.getElementById('fileName2');
  const btnClearFile2 = document.getElementById('btnClearFile2');

  const imagePreview = document.getElementById('imagePreview');
  const changeOverlay = document.getElementById('changeOverlay');
  const groundingCanvas = document.getElementById('groundingCanvas');
  const previewPlaceholder = document.getElementById('previewPlaceholder');
  const modelBadge = document.getElementById('modelBadge');
  const quickQuestions = document.getElementById('quickQuestions');

  // Studio Header Elements
  const workspaceBadge = document.getElementById('workspaceBadge');
  const workspaceTitle = document.getElementById('workspaceTitle');
  const workspaceSubtitle = document.getElementById('workspaceSubtitle');
  const panelTitle = document.getElementById('panelTitle');
  const panelSubtitle = document.getElementById('panelSubtitle');

  // Layer Controls
  const layerControls = document.getElementById('layerControls');
  const toggleGrounding = document.getElementById('toggleGrounding');
  const toggleChangeLayer = document.getElementById('toggleChangeLayer');
  const toggleChange = document.getElementById('toggleChange');

  // Metadata Telemetry Elements
  const metaCrs = document.getElementById('metaCrs');
  const metaShape = document.getElementById('metaShape');
  const metaBands = document.getElementById('metaBands');
  const metaDriver = document.getElementById('metaDriver');

  // Results & Provenance Elements
  const resTask = document.getElementById('resTask');
  const resTool = document.getElementById('resTool');
  const resModel = document.getElementById('resModel');
  const resLatency = document.getElementById('resLatency');
  const resConfidence = document.getElementById('resConfidence');
  const resAnswer = document.getElementById('resAnswer');
  const agentTraceContainer = document.getElementById('agentTraceContainer');
  const traceSteps = document.getElementById('traceSteps');
  const provenanceFeed = document.getElementById('provenanceFeed');
  const provenanceFeedFull = document.getElementById('provenanceFeedFull');
  const btnRefreshProv = document.getElementById('btnRefreshProv');
  const btnExportJson = document.getElementById('btnExportJson');

  // Studio Profiles per Route
  const studioProfiles = {
    vqa: {
      badge: 'Ask Questions (VQA)',
      title: 'Ask Anything About a Satellite Image',
      subtitle: 'Upload any satellite image or GeoTIFF and ask questions in plain English.',
      panelTitle: 'Choose Your Image',
      panelSubtitle: 'Upload a satellite picture or select one of the sample images.',
      presets: [
        { label: 'What is here?', q: 'What land types and objects are visible in this image?' },
        { label: 'Any Water?', q: 'Are there any water bodies, lakes, or rivers here?' },
        { label: 'Buildings & Cities', q: 'Are there buildings, roads, or cities present?' },
        { label: 'Farms & Crops', q: 'Describe the farms, crops, and greenery in this scene.' }
      ]
    },
    grounding: {
      badge: 'Find Objects',
      title: 'Locate & Highlight Specific Areas',
      subtitle: 'Find specific features on the ground with glowing highlight boxes.',
      panelTitle: 'Choose Your Image',
      panelSubtitle: 'Upload an image where you want to find and mark specific things.',
      presets: [
        { label: 'Find Water', q: 'Find and highlight the water body in this image.' },
        { label: 'Find Buildings', q: 'Draw boxes around the buildings and urban areas.' },
        { label: 'Find Farms', q: 'Highlight the farm fields and crops.' },
        { label: 'Find Roads', q: 'Locate the roads and transportation lines.' }
      ]
    },
    bitemporal_change: {
      badge: 'Before & After Change',
      title: 'Spot Changes Over Time',
      subtitle: 'Compare two satellite images of the same area taken on different dates.',
      panelTitle: 'Upload Both Images',
      panelSubtitle: 'Upload the "before" image on the left and the "after" image on the right.',
      presets: [
        { label: 'Flood Damage', q: 'What flood or water changes happened between these two dates?' },
        { label: 'New Construction', q: 'Did the built-up city area grow or change between these dates?' },
        { label: 'Forest & Trees', q: 'What changes happened to the trees, plants, or greenery?' },
        { label: 'All Changes', q: 'What changed between these two pictures and where did it happen?' }
      ]
    },
    optical_sar_fusion: {
      badge: 'See Through Clouds',
      title: 'Combine Normal Photos with Radar',
      subtitle: 'Use radar to see ground details clearly even when thick clouds block normal cameras.',
      panelTitle: 'Upload Camera & Radar Images',
      panelSubtitle: 'Upload the cloudy optical picture on the left and the radar image on the right.',
      presets: [
        { label: 'Find Buildings in Clouds', q: 'Use both images together to find buildings through the clouds.' },
        { label: 'Map Water', q: 'Combine color and radar data to accurately map water bodies.' },
        { label: 'Check Farm Soil', q: 'Use optical color and radar roughness to inspect farm fields.' },
        { label: 'Full Summary', q: 'Give a complete summary using both the optical photo and radar data.' }
      ]
    }
  };

  // ==========================================================================
  // Router Implementation
  // ==========================================================================
  function navigateToRoute(route, pushState = true) {
    let cleanRoute = (route || '').replace(/^\//, '').trim().toLowerCase();
    if (!cleanRoute || cleanRoute === 'index.html') cleanRoute = 'overview';

    const viewOverview = document.getElementById('viewOverview');
    const viewWorkspace = document.getElementById('viewWorkspace');
    const viewProvenance = document.getElementById('viewProvenance');

    // Hide all views
    viewOverview.classList.add('hidden');
    viewWorkspace.classList.add('hidden');
    viewProvenance.classList.add('hidden');

    // Update Nav Link Active States
    document.querySelectorAll('.nav-link').forEach(link => {
      const linkRoute = (link.dataset.route || '').toLowerCase();
      if (linkRoute === cleanRoute || (cleanRoute === 'overview' && linkRoute === '')) {
        link.classList.add('active');
      } else {
        link.classList.remove('active');
      }
    });

    if (cleanRoute === 'overview') {
      viewOverview.classList.remove('hidden');
      window.scrollTo({ top: 0, behavior: 'smooth' });
    } else if (cleanRoute === 'provenance') {
      viewProvenance.classList.remove('hidden');
      loadProvenanceFull();
      window.scrollTo({ top: 0, behavior: 'smooth' });
    } else {
      // Workspace Studios: vqa, grounding, change, fusion
      viewWorkspace.classList.remove('hidden');

      if (cleanRoute === 'vqa') currentMode = 'vqa';
      else if (cleanRoute === 'grounding') currentMode = 'grounding';
      else if (cleanRoute === 'change' || cleanRoute === 'bitemporal_change') currentMode = 'bitemporal_change';
      else if (cleanRoute === 'fusion' || cleanRoute === 'optical_sar_fusion') currentMode = 'optical_sar_fusion';
      else currentMode = 'vqa';

      updateModeUI();
      window.scrollTo({ top: 0, behavior: 'smooth' });
    }

    if (pushState) {
      const targetUrl = cleanRoute === 'overview' ? '/' : `/${cleanRoute}`;
      history.pushState({ route: cleanRoute }, '', targetUrl);
    }
  }

  // Intercept Nav Links
  document.querySelectorAll('.nav-link').forEach(link => {
    link.addEventListener('click', (e) => {
      e.preventDefault();
      const route = link.dataset.route || link.getAttribute('href');
      navigateToRoute(route, true);
    });
  });

  window.addEventListener('popstate', () => {
    const currentPath = window.location.pathname.replace(/^\//, '') || 'overview';
    navigateToRoute(currentPath, false);
  });

  function updateModeUI() {
    clearAlert();
    clearGroundingCanvas();
    changeOverlay.classList.add('hidden');
    toggleChangeLayer.classList.add('hidden');

    const profile = studioProfiles[currentMode] || studioProfiles.vqa;

    workspaceBadge.textContent = profile.badge;
    workspaceTitle.textContent = profile.title;
    workspaceSubtitle.textContent = profile.subtitle;
    panelTitle.textContent = profile.panelTitle;
    panelSubtitle.textContent = profile.panelSubtitle;

    if (currentMode === 'bitemporal_change') {
      secondaryGroup.classList.remove('hidden');
      labelPrimary.textContent = 'First Image (Before Date)';
      dropTextPrimary.textContent = 'Click or drop the Before image here';
      document.getElementById('labelSecondaryImage').textContent = 'Second Image (After Date)';
      document.getElementById('dropTextSecondary').textContent = 'Click or drop the After image here';
    } else if (currentMode === 'optical_sar_fusion') {
      secondaryGroup.classList.remove('hidden');
      labelPrimary.textContent = 'Cloudy Camera Image (Optical)';
      dropTextPrimary.textContent = 'Click or drop the Optical image here';
      document.getElementById('labelSecondaryImage').textContent = 'Radar Image (SAR)';
      document.getElementById('dropTextSecondary').textContent = 'Click or drop the Radar image here';
    } else {
      secondaryGroup.classList.add('hidden');
      labelPrimary.textContent = 'Satellite Image (GeoTIFF, PNG, JPEG)';
      dropTextPrimary.textContent = 'Click or drop your satellite image here';
    }

    // Populate Presets
    const presets = profile.presets || [];
    quickQuestions.innerHTML = '<span class="quick-title">Quick Examples:</span>' + presets.map(p => `
      <button type="button" class="chip" data-q="${escapeHtml(p.q)}">${escapeHtml(p.label)}</button>
    `).join('');

    // Attach chip listeners
    quickQuestions.querySelectorAll('.chip').forEach(chip => {
      chip.addEventListener('click', () => {
        questionInput.value = chip.dataset.q;
        questionInput.focus();
      });
    });

    if (presets.length > 0) {
      questionInput.placeholder = presets[0].q;
    }
  }

  // Fetch Health & Active Model
  async function checkHealth() {
    try {
      const res = await fetch('/health');
      if (res.ok) {
        const data = await res.json();
        modelBadge.textContent = data.active_model || 'Ready';
      } else {
        modelBadge.textContent = 'Backend Offline';
      }
    } catch (e) {
      modelBadge.textContent = 'Offline';
    }
  }

  // Fetch Provenance Feed
  async function loadProvenance() {
    try {
      const res = await fetch('/executions?limit=5');
      if (res.ok) {
        const data = await res.json();
        executionHistory = data;
      }
    } catch (e) {
      console.error('Failed to load provenance feed', e);
    }
  }

  // Fullscreen Provenance Feed
  async function loadProvenanceFull() {
    try {
      const res = await fetch('/executions?limit=50');
      if (res.ok) {
        const data = await res.json();
        executionHistory = data;
        if (provenanceFeedFull) {
          if (data.length === 0) {
            provenanceFeedFull.innerHTML = '<div class="prov-empty">No executions recorded yet. Launch a studio above to run queries!</div>';
            return;
          }
          provenanceFeedFull.innerHTML = data.map((item, idx) => `
            <div class="prov-item">
              <div>
                <div><strong>#${data.length - idx} [${escapeHtml((item.task || 'vqa').toUpperCase())}]</strong> — <em>${escapeHtml(item.model)}</em></div>
                <div style="color: #c5cedd; margin-top: 4px;"><strong>Query:</strong> "${escapeHtml(item.question)}"</div>
                <div style="color: #9aa4b8; font-size: 12px; margin-top: 2px;"><strong>Files:</strong> ${escapeHtml(item.input)}</div>
                <div style="color: #5eead4; margin-top: 4px;"><strong>Result:</strong> ${escapeHtml(item.output)}</div>
              </div>
              <div style="text-align: right; flex-shrink: 0;">
                <div style="color: #ff9a56; font-weight: 600;">${item.execution_time_sec}s</div>
                <div style="font-size: 11px; color: #6f7a8c; margin-top: 4px;">${item.timestamp ? item.timestamp.replace('T', ' ').split('.')[0] : ''}</div>
              </div>
            </div>
          `).join('');
        }
      }
    } catch (e) {
      console.error('Failed to load full provenance', e);
    }
  }

  // Export JSON functionality
  if (btnExportJson) {
    btnExportJson.addEventListener('click', () => {
      const dataStr = "data:text/json;charset=utf-8," + encodeURIComponent(JSON.stringify(executionHistory, null, 2));
      const downloadAnchor = document.createElement('a');
      downloadAnchor.setAttribute("href", dataStr);
      downloadAnchor.setAttribute("download", "satquery_executions_ledger.json");
      document.body.appendChild(downloadAnchor);
      downloadAnchor.click();
      downloadAnchor.remove();
    });
  }

  function escapeHtml(str) {
    if (!str) return '';
    return str.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');
  }

  function showAlert(msg, isError = true) {
    alertBox.textContent = msg;
    alertBox.className = `alert-box ${isError ? 'error' : ''}`;
    alertBox.classList.remove('hidden');
  }

  function clearAlert() {
    alertBox.classList.add('hidden');
    alertBox.textContent = '';
  }

  // Handle File Upload & Server-side Preview
  async function handleFile(file, isSecondary = false) {
    clearAlert();
    if (!file) return;

    if (!isSecondary) {
      fileName.textContent = `${file.name} (${(file.size / 1024).toFixed(1)} KB)`;
      fileInfo.classList.remove('hidden');
    } else {
      fileName2.textContent = `${file.name} (${(file.size / 1024).toFixed(1)} KB)`;
      fileInfo2.classList.remove('hidden');
    }

    if (!isSecondary) {
      const isTiff = file.name.toLowerCase().endsWith('.tif') || file.name.toLowerCase().endsWith('.tiff');
      if (!isTiff && file.type.startsWith('image/')) {
        const reader = new FileReader();
        reader.onload = (e) => {
          imagePreview.src = e.target.result;
          imagePreview.classList.remove('hidden');
          previewPlaceholder.classList.add('hidden');
        };
        reader.readAsDataURL(file);
      } else {
        imagePreview.classList.add('hidden');
        previewPlaceholder.classList.remove('hidden');
        const spanText = previewPlaceholder.querySelector('span');
        if (spanText) spanText.textContent = `Normalizing GeoTIFF bands for ${file.name}...`;
      }
    }

    const formData = new FormData();
    formData.append('image', file);

    try {
      const res = await fetch('/preview', {
        method: 'POST',
        body: formData
      });

      if (res.ok) {
        const data = await res.json();
        if (!isSecondary && data.preview_url) {
          imagePreview.src = data.preview_url;
          imagePreview.classList.remove('hidden');
          previewPlaceholder.classList.add('hidden');
        }
        if (!isSecondary && data.metadata) {
          metaCrs.textContent = data.metadata.crs || 'Local / None';
          metaShape.textContent = data.metadata.shape ? `[${data.metadata.shape.join(', ')}]` : '—';
          metaBands.textContent = data.metadata.bands || data.metadata.count || '—';
          metaDriver.textContent = data.metadata.driver || 'Raster';
        }
      }
    } catch (err) {
      console.error('Failed to generate GeoTIFF preview', err);
    }
  }

  function resetFileSelection(isSecondary = false) {
    if (!isSecondary) {
      imageInput.value = '';
      fileInfo.classList.add('hidden');
      imagePreview.classList.add('hidden');
      clearGroundingCanvas();
      changeOverlay.classList.add('hidden');
      previewPlaceholder.classList.remove('hidden');
      const spanText = previewPlaceholder.querySelector('span');
      if (spanText) spanText.textContent = 'Upload or select a scene to preview raster and extract coordinates';
      metaCrs.textContent = '—';
      metaShape.textContent = '—';
      metaBands.textContent = '—';
      metaDriver.textContent = '—';
    } else {
      secondaryImageInput.value = '';
      fileInfo2.classList.add('hidden');
    }
  }

  // Draw Visual Grounding Bounding Boxes
  function drawGroundingBoxes(boxes) {
    if (!boxes || boxes.length === 0) {
      clearGroundingCanvas();
      layerControls.classList.add('hidden');
      return;
    }

    cachedBoxes = boxes;
    layerControls.classList.remove('hidden');
    groundingCanvas.classList.remove('hidden');

    const w = imagePreview.clientWidth || 300;
    const h = imagePreview.clientHeight || 240;

    groundingCanvas.width = w;
    groundingCanvas.height = h;
    groundingCanvas.style.width = `${w}px`;
    groundingCanvas.style.height = `${h}px`;

    const ctx = groundingCanvas.getContext('2d');
    ctx.clearRect(0, 0, w, h);

    if (!toggleGrounding.checked) return;

    boxes.forEach((box) => {
      const bx = box.xmin * w;
      const by = box.ymin * h;
      const bw = (box.xmax - box.xmin) * w;
      const bh = (box.ymax - box.ymin) * h;

      // Glow outline
      ctx.shadowColor = '#5eead4';
      ctx.shadowBlur = 10;
      ctx.strokeStyle = '#5eead4';
      ctx.lineWidth = 2.5;
      ctx.strokeRect(bx, by, bw, bh);

      // Translucent fill
      ctx.fillStyle = 'rgba(94, 234, 212, 0.12)';
      ctx.fillRect(bx, by, bw, bh);

      // Label badge
      ctx.shadowBlur = 0;
      const label = `${box.label || 'Target'} ${box.confidence ? `(${Math.round(box.confidence * 100)}%)` : ''}`;
      ctx.font = '11px JetBrains Mono, monospace';
      const textWidth = ctx.measureText(label).width;

      ctx.fillStyle = 'rgba(15, 23, 42, 0.85)';
      ctx.fillRect(bx, Math.max(0, by - 20), textWidth + 12, 18);
      ctx.strokeStyle = '#5eead4';
      ctx.lineWidth = 1;
      ctx.strokeRect(bx, Math.max(0, by - 20), textWidth + 12, 18);

      ctx.fillStyle = '#5eead4';
      ctx.fillText(label, bx + 6, Math.max(13, by - 6));
    });
  }

  function clearGroundingCanvas() {
    cachedBoxes = [];
    const ctx = groundingCanvas.getContext('2d');
    ctx.clearRect(0, 0, groundingCanvas.width, groundingCanvas.height);
    groundingCanvas.classList.add('hidden');
  }

  toggleGrounding.addEventListener('change', () => {
    drawGroundingBoxes(cachedBoxes);
  });

  toggleChange.addEventListener('change', () => {
    if (toggleChange.checked) {
      changeOverlay.classList.remove('hidden');
    } else {
      changeOverlay.classList.add('hidden');
    }
  });

  // Render Observable Execution Trace
  function renderAgentTrace(trace) {
    if (!trace || !trace.steps) {
      agentTraceContainer.classList.add('hidden');
      return;
    }

    agentTraceContainer.classList.remove('hidden');
    traceSteps.innerHTML = trace.steps.map(s => `
      <div class="trace-step-item">
        <span class="trace-step-num">${s.step}</span>
        <div class="trace-step-content">
          <strong>${escapeHtml(s.action)}:</strong> ${escapeHtml(s.details || '')}
        </div>
      </div>
    `).join('');
  }

  // Event Listeners for File Selection
  imageInput.addEventListener('change', (e) => {
    if (e.target.files && e.target.files.length > 0) {
      handleFile(e.target.files[0], false);
    }
  });

  secondaryImageInput.addEventListener('change', (e) => {
    if (e.target.files && e.target.files.length > 0) {
      handleFile(e.target.files[0], true);
    }
  });

  btnClearFile.addEventListener('click', () => resetFileSelection(false));
  btnClearFile2.addEventListener('click', () => resetFileSelection(true));

  // Drag & Drop Handling
  ['dragenter', 'dragover'].forEach(name => {
    dropZone.addEventListener(name, (e) => {
      e.preventDefault();
      dropZone.classList.add('drag-over');
    });
    if (dropZone2) {
      dropZone2.addEventListener(name, (e) => {
        e.preventDefault();
        dropZone2.classList.add('drag-over');
      });
    }
  });

  ['dragleave', 'drop'].forEach(name => {
    dropZone.addEventListener(name, (e) => {
      e.preventDefault();
      dropZone.classList.remove('drag-over');
    });
    if (dropZone2) {
      dropZone2.addEventListener(name, (e) => {
        e.preventDefault();
        dropZone2.classList.remove('drag-over');
      });
    }
  });

  dropZone.addEventListener('drop', (e) => {
    if (e.dataTransfer && e.dataTransfer.files.length > 0) {
      imageInput.files = e.dataTransfer.files;
      handleFile(e.dataTransfer.files[0], false);
    }
  });

  if (dropZone2) {
    dropZone2.addEventListener('drop', (e) => {
      if (e.dataTransfer && e.dataTransfer.files.length > 0) {
        secondaryImageInput.files = e.dataTransfer.files;
        handleFile(e.dataTransfer.files[0], true);
      }
    });
  }

  // Form Submission
  vqaForm.addEventListener('submit', async (e) => {
    e.preventDefault();
    clearAlert();
    clearGroundingCanvas();
    changeOverlay.classList.add('hidden');
    toggleChangeLayer.classList.add('hidden');

    if (!imageInput.files || imageInput.files.length === 0) {
      showAlert('Please select or upload a primary satellite image.');
      return;
    }

    if ((currentMode === 'bitemporal_change' || currentMode === 'optical_sar_fusion') && (!secondaryImageInput.files || secondaryImageInput.files.length === 0)) {
      showAlert(`Please upload the secondary scene required for ${currentMode.replace('_', ' ').toUpperCase()} analysis.`);
      return;
    }

    const question = questionInput.value.trim();
    if (!question) {
      showAlert('Please enter a natural language question.');
      return;
    }

    const formData = new FormData();
    formData.append('image', imageInput.files[0]);
    formData.append('question', question);
    formData.append('task_mode', currentMode);

    if (secondaryImageInput.files && secondaryImageInput.files.length > 0) {
      formData.append('secondary_image', secondaryImageInput.files[0]);
    }

    // UI Loading state
    btnAnalyse.disabled = true;
    spinner.classList.remove('hidden');
    resAnswer.textContent = 'Agentic orchestrator is classifying intent and executing specialist remote-sensing tools...';
    agentTraceContainer.classList.add('hidden');

    try {
      const response = await fetch('/agent/analyze', {
        method: 'POST',
        body: formData,
      });

      const data = await response.json();

      if (!response.ok) {
        const errorMsg = data.detail || 'An error occurred during agentic processing.';
        throw new Error(errorMsg);
      }

      // Display Results
      resAnswer.textContent = data.answer;
      resTask.textContent = (data.task || currentMode).toUpperCase();
      resTool.textContent = data.tool_used || 'Specialist Tool';
      resModel.textContent = data.model || '—';
      resLatency.textContent = data.execution_time_sec !== undefined ? `${data.execution_time_sec}s` : '—';
      resConfidence.textContent = data.confidence !== null && data.confidence !== undefined ? `${Math.round(data.confidence * 100)}%` : 'null';

      // Visual Grounding Bounding Boxes
      if (data.boxes && data.boxes.length > 0) {
        setTimeout(() => drawGroundingBoxes(data.boxes), 100);
      }

      // Change Heatmap Overlay
      if (data.change_map_url) {
        changeOverlay.src = data.change_map_url;
        changeOverlay.classList.remove('hidden');
        toggleChangeLayer.classList.remove('hidden');
        layerControls.classList.remove('hidden');
      }

      // Observable Agent Execution Trace
      if (data.execution_trace) {
        renderAgentTrace(data.execution_trace);
      }

      // Display Metadata
      if (data.metadata) {
        metaCrs.textContent = data.metadata.crs || 'Non-georeferenced';
        metaShape.textContent = data.metadata.shape ? `[${data.metadata.shape.join(', ')}]` : '—';
        metaBands.textContent = data.metadata.count || data.metadata.bands || '—';
        metaDriver.textContent = data.metadata.driver || '—';
      }

      loadProvenance();

    } catch (err) {
      showAlert(err.message, true);
      resAnswer.textContent = 'Inference failed.';
    } finally {
      btnAnalyse.disabled = false;
      spinner.classList.add('hidden');
    }
  });

  if (btnRefreshProv) {
    btnRefreshProv.addEventListener('click', () => {
      loadProvenance();
      loadProvenanceFull();
    });
  }

  // Initial Route Hydration from URL
  const initialPath = window.location.pathname.replace(/^\//, '') || 'overview';
  navigateToRoute(initialPath, false);

  checkHealth();
  loadProvenance();
});
