document.addEventListener('DOMContentLoaded', () => {

    // --- Navigation Logic ---
    const navButtons = document.querySelectorAll('.nav-btn');
    const sections = document.querySelectorAll('.glass-card');
    const pageTitle = document.getElementById('page-title');

    const titles = {
        'dashboard': 'Dashboard Overview',
        'crop-section': 'Crop Match AI',
        'irrigation-section': 'Irrigation Optimization',
        'disease-section': 'Plant Disease Scan',
        'yield-section': 'Yield Predictor'
    };

    navButtons.forEach(btn => {
        btn.addEventListener('click', () => {
            // Remove active classes
            navButtons.forEach(b => b.classList.remove('active'));
            sections.forEach(s => s.classList.add('hidden'));

            // Add active class to clicked
            btn.classList.add('active');
            const targetId = btn.getAttribute('data-target');
            document.getElementById(targetId).classList.remove('hidden');
            pageTitle.innerText = titles[targetId];
        });
    });

    // --- Loader Logic ---
    const showLoader = () => document.getElementById('loader').style.display = 'flex';
    const hideLoader = () => document.getElementById('loader').style.display = 'none';

    // --- File Upload Logic ---
    const uploadArea = document.getElementById('upload-area');
    const fileInput = document.getElementById('file-input');
    const imagePreview = document.getElementById('image-preview');
    const previewImg = document.getElementById('preview-img');
    const clearImgBtn = document.getElementById('clear-img');

    uploadArea.addEventListener('click', () => fileInput.click());

    fileInput.addEventListener('change', function() {
        if(this.files && this.files[0]) {
            const reader = new FileReader();
            reader.onload = (e) => {
                previewImg.src = e.target.result;
                uploadArea.style.display = 'none';
                imagePreview.style.display = 'block';
            }
            reader.readAsDataURL(this.files[0]);
        }
    });

    clearImgBtn.addEventListener('click', () => {
        fileInput.value = '';
        previewImg.src = '';
        imagePreview.style.display = 'none';
        uploadArea.style.display = 'block';
        document.getElementById('disease-result').style.display = 'none';
    });


    // --- AI API Handlers ---
    const handleFormSubmit = async (formId, endpoint, resultId, isFileUpload = false) => {
        const form = document.getElementById(formId);
        form.addEventListener('submit', async (e) => {
            e.preventDefault();
            showLoader();
            const resultBox = document.getElementById(resultId);
            resultBox.style.display = 'none';
            resultBox.className = 'result-box dark-bg';

            const formData = new FormData(form);

            try {
                const response = await fetch(`/${endpoint}`, {
                    method: 'POST',
                    body: formData
                });
                const data = await response.json();

                if (data.success) {
                    if (data.confidence) {
                        resultBox.innerHTML = `Prediction: <span class="text-green">${data.prediction}</span><br>Confidence: ${data.confidence}`;
                    } else {
                        resultBox.innerHTML = `Recommended Output: <span class="text-green">${data.prediction}</span>`;
                    }
                } else {
                    resultBox.classList.add('error');
                    resultBox.innerHTML = `Error: ${data.error}`;
                }
            } catch (error) {
                resultBox.classList.add('error');
                resultBox.innerHTML = `Network Error: Could not connect to the AI model.`;
            }

            hideLoader();
            resultBox.style.display = 'block';
        });
    }

    // Attach Handlers
    handleFormSubmit('crop-form', 'predict_crop', 'crop-result');
    handleFormSubmit('irrigation-form', 'predict_irrigation', 'irrigation-result');
    handleFormSubmit('yield-form', 'predict_yield', 'yield-result');
    handleFormSubmit('disease-form', 'predict_disease', 'disease-result', true);

});
