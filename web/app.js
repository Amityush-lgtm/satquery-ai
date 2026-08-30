document.addEventListener('DOMContentLoaded', () => {
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

    // Metadata Elements
    const metaCrs = document.getElementById('metaCrs');
    const metaShape = document.getElementById('metaShape');
    const metaBands = document.getElementById('metaBands');
    const metaDriver = document.getElementById('metaDriver');

    // Results Elements
    const resModel = document.getElementById('resModel');
    const resLatency = document.getElementById('resLatency');
    const resConfidence = document.getElementById('resConfidence');
    const resAnswer = document.getElementById('resAnswer');
    const provenanceFeed = document.getElementById('provenanceFeed');
    const btnRefreshProv = document.getElementById('btnRefreshProv');

    // Fetch Health / Model on load
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
                if (data.length === 0) {
                    provenanceFeed.innerHTML = '<div class="prov-empty">No executions recorded yet.</div>';
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
            console.error('Failed to load provenance', e);
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

    // Handle File Selection & Preview
    function handleFile(file) {
        clearAlert();
        if (!file) return;

        fileName.textContent = `${file.name} (${(file.size / 1024).toFixed(1)} KB)`;
        fileInfo.classList.remove('hidden');

        // Preview standard images immediately
        if (file.type.startsWith('image/') && !file.name.toLowerCase().endsWith('.tif') && !file.name.toLowerCase().endsWith('.tiff')) {
            const reader = new FileReader();
            reader.onload = (e) => {
                imagePreview.src = e.target.result;
                imagePreview.classList.remove('hidden');
                previewPlaceholder.classList.add('hidden');
            };
            reader.readAsDataURL(file);
        } else {
            // For GeoTIFF, show placeholder indicating server-side normalization
            imagePreview.classList.add('hidden');
            previewPlaceholder.classList.remove('hidden');
            previewPlaceholder.querySelector('span').textContent = `GeoTIFF: ${file.name} (Ready for VLM)`;
        }
    }

    imageInput.addEventListener('change', (e) => {
        if (e.target.files.length > 0) {
            handleFile(e.target.files[0]);
        }
    });

    btnClearFile.addEventListener('click', () => {
        imageInput.value = '';
        fileInfo.classList.add('hidden');
        imagePreview.classList.add('hidden');
        previewPlaceholder.classList.remove('hidden');
        previewPlaceholder.querySelector('span').textContent = 'Image preview will appear here';
    });

    // Drag & Drop
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
        if (e.dataTransfer.files.length > 0) {
            imageInput.files = e.dataTransfer.files;
            handleFile(e.dataTransfer.files[0]);
        }
    });

    // Quick Question Preset Chips
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
            showAlert('Please enter a question.');
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
            resModel.textContent = data.model;
            resLatency.textContent = data.execution_time_sec ? `${data.execution_time_sec}s` : '—';
            resConfidence.textContent = data.confidence !== null && data.confidence !== undefined ? data.confidence : 'null';

            // Display Metadata
            if (data.metadata) {
                metaCrs.textContent = data.metadata.crs || 'Non-georeferenced';
                metaShape.textContent = data.metadata.shape ? `[${data.metadata.shape.join(', ')}]` : '—';
                metaBands.textContent = data.metadata.count || '—';
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

    btnRefreshProv.addEventListener('click', loadProvenance);

    // Initial load
    checkHealth();
    loadProvenance();
});
