/**
 * SatQuery.ai — Interactive Frontend & Three.js Celestial Background
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
// 2. SatQuery AI Application Logic
// ============================================================================
document.addEventListener('DOMContentLoaded', () => {
  // DOM Form & Input Elements
  const vqaForm = document.getElementById('vqaForm');
  const imageInput = document.getElementById('imageInput');
  const dropZone = document.getElementById('dropZone');
  const questionInput = document.getElementById('questionInput');
  const btnAnalyse = document.getElementById('btnAnalyse');
  const spinner = document.getElementById('spinner');
  const alertBox = document.getElementById('alertBox');
  const fileInfo = document.getElementById('fileInfo');
  const fileName = document.getElementById('fileName');
  const btnClearFile = document.getElementById('btnClearFile');
  const imagePreview = document.getElementById('imagePreview');
  const previewPlaceholder = document.getElementById('previewPlaceholder');
  const modelBadge = document.getElementById('modelBadge');

  // Metadata Telemetry Elements
  const metaCrs = document.getElementById('metaCrs');
  const metaShape = document.getElementById('metaShape');
  const metaBands = document.getElementById('metaBands');
  const metaDriver = document.getElementById('metaDriver');

  // Results & Provenance Elements
  const resModel = document.getElementById('resModel');
  const resLatency = document.getElementById('resLatency');
  const resConfidence = document.getElementById('resConfidence');
  const resAnswer = document.getElementById('resAnswer');
  const provenanceFeed = document.getElementById('provenanceFeed');
  const btnRefreshProv = document.getElementById('btnRefreshProv');



  // Fetch Health & Active Model on load
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

  // Fetch Provenance Audit Feed
  async function loadProvenance() {
    try {
      const res = await fetch('/executions?limit=5');
      if (res.ok) {
        const data = await res.json();
        if (data.length === 0) {
          provenanceFeed.innerHTML = '<div class="prov-empty">No executions recorded yet. Run your first query above!</div>';
          return;
        }
        provenanceFeed.innerHTML = data.map(item => `
          <div class="prov-item">
            <span><strong>${escapeHtml(item.input)}</strong>: "${escapeHtml(item.question)}"</span>
            <span>${item.execution_time_sec}s • ${item.timestamp ? item.timestamp.split('T')[1].split('.')[0] : ''}</span>
          </div>
        `).join('');
      }
    } catch (e) {
      console.error('Failed to load provenance feed', e);
    }
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

  // Handle File Selection & Server-Side GeoTIFF Normalization
  async function handleFile(file) {
    clearAlert();
    if (!file) return;

    fileName.textContent = `${file.name} (${(file.size / 1024).toFixed(1)} KB)`;
    fileInfo.classList.remove('hidden');

    const isTiff = file.name.toLowerCase().endsWith('.tif') || file.name.toLowerCase().endsWith('.tiff');

    // For standard images, show immediate client-side preview
    if (!isTiff && file.type.startsWith('image/')) {
      const reader = new FileReader();
      reader.onload = (e) => {
        imagePreview.src = e.target.result;
        imagePreview.classList.remove('hidden');
        previewPlaceholder.classList.add('hidden');
      };
      reader.readAsDataURL(file);
    } else {
      // For GeoTIFFs, display loading state while server renders RGB preview
      imagePreview.classList.add('hidden');
      previewPlaceholder.classList.remove('hidden');
      const spanText = previewPlaceholder.querySelector('span');
      if (spanText) spanText.textContent = `Normalizing GeoTIFF bands for ${file.name}...`;
    }

    // Call backend /preview endpoint for server-side normalization and metadata extraction
    const formData = new FormData();
    formData.append('image', file);

    try {
      const res = await fetch('/preview', {
        method: 'POST',
        body: formData
      });

      if (res.ok) {
        const data = await res.json();
        if (data.preview_url) {
          imagePreview.src = data.preview_url;
          imagePreview.classList.remove('hidden');
          previewPlaceholder.classList.add('hidden');
        }
        if (data.metadata) {
          metaCrs.textContent = data.metadata.crs || 'Local / None';
          metaShape.textContent = data.metadata.shape ? `[${data.metadata.shape.join(', ')}]` : '—';
          metaBands.textContent = data.metadata.bands || data.metadata.count || '—';
          metaDriver.textContent = data.metadata.driver || 'Raster';
        }
      } else {
        console.warn('Preview generation returned non-OK status');
      }
    } catch (err) {
      console.error('Failed to generate GeoTIFF preview', err);
    }
  }

  // Clear File Handler
  function resetFileSelection() {
    imageInput.value = '';
    fileInfo.classList.add('hidden');
    imagePreview.classList.add('hidden');
    previewPlaceholder.classList.remove('hidden');
    const spanText = previewPlaceholder.querySelector('span');
    if (spanText) spanText.textContent = 'Upload or select a scene to preview raster and extract coordinates';
    metaCrs.textContent = '—';
    metaShape.textContent = '—';
    metaBands.textContent = '—';
    metaDriver.textContent = '—';
  }

  imageInput.addEventListener('change', (e) => {
    if (e.target.files && e.target.files.length > 0) {
      handleFile(e.target.files[0]);
    }
  });

  btnClearFile.addEventListener('click', resetFileSelection);

  // Drag & Drop Interactions
  ['dragenter', 'dragover'].forEach(name => {
    dropZone.addEventListener(name, (e) => {
      e.preventDefault();
      dropZone.classList.add('drag-over');
    });
  });

  ['dragleave', 'drop'].forEach(name => {
    dropZone.addEventListener(name, (e) => {
      e.preventDefault();
      dropZone.classList.remove('drag-over');
    });
  });

  dropZone.addEventListener('drop', (e) => {
    if (e.dataTransfer && e.dataTransfer.files.length > 0) {
      imageInput.files = e.dataTransfer.files;
      handleFile(e.dataTransfer.files[0]);
    }
  });

  // Preset Question Chips
  document.querySelectorAll('.chip').forEach(chip => {
    chip.addEventListener('click', () => {
      questionInput.value = chip.dataset.q;
      questionInput.focus();
    });
  });



  // Form Submission
  vqaForm.addEventListener('submit', async (e) => {
    e.preventDefault();
    clearAlert();

    if (!imageInput.files || imageInput.files.length === 0) {
      showAlert('Please select or upload a satellite image.');
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

    // UI Loading state
    btnAnalyse.disabled = true;
    spinner.classList.remove('hidden');
    resAnswer.textContent = 'Analyzing satellite image with Remote-Sensing VLM...';

    try {
      const response = await fetch('/vqa', {
        method: 'POST',
        body: formData,
      });

      const data = await response.json();

      if (!response.ok) {
        const errorMsg = data.detail || 'An error occurred during VQA processing.';
        throw new Error(errorMsg);
      }

      // Display Results
      resAnswer.textContent = data.answer;
      resModel.textContent = data.model || '—';
      resLatency.textContent = data.execution_time_sec !== undefined ? `${data.execution_time_sec}s` : '—';
      resConfidence.textContent = data.confidence !== null && data.confidence !== undefined ? data.confidence : 'null';

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
    btnRefreshProv.addEventListener('click', loadProvenance);
  }

  // Initial Data Load
  checkHealth();
  loadProvenance();
});
