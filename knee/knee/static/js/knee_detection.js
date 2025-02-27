let cropper;

function initializeCropper(uploadedImage) {
    if (cropper) cropper.destroy();
    cropper = new Cropper(uploadedImage, {
        aspectRatio: 1,
        viewMode: 1,
        autoCropArea: 0.8,
        responsive: true,
        crop: function () {
            const canvas = cropper.getCroppedCanvas();
            const dataUrl = canvas.toDataURL();
            const croppedPreview = document.getElementById("croppedPreview");
            croppedPreview.src = dataUrl;
            croppedPreview.hidden = false;
        }
    });
}

function handleImageUpload(event) {
    const file = event.target.files[0];
    if (!file) return;

    const uploadedImage = document.getElementById("uploadedImage");
    const diagnoseButton = document.getElementById("diagnoseButton");

    uploadedImage.src = URL.createObjectURL(file);
    uploadedImage.hidden = false;
    diagnoseButton.disabled = false;

    initializeCropper(uploadedImage);
}

function handleDiagnose() {
    if (!cropper) return;

    const croppedImage = cropper.getCroppedCanvas().toDataURL("image/jpeg");
    sendToBackend(croppedImage);

    // Cuộn xuống phần bottom-section sau khi nhấn nút "Chẩn đoán"
    const bottomSection = document.getElementById("bottomSection");
    bottomSection.style.display = "block"; // Hiển thị bottom-section

    // Cuộn xuống bottom-section
    bottomSection.scrollIntoView({ behavior: "smooth" });
}

function sendToBackend(croppedImage) {
    fetch("/diagnose_knee", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ cropped_image: croppedImage }),
    })
        .then(response => {
            if (!response.ok) throw new Error(`HTTP error! Status: ${response.status}`);
            return response.json();
        })
        .then(data => handleBackendResponse(data, croppedImage))
        .catch(error => console.error("Error:", error));
}

function handleBackendResponse(data, croppedImage) {
    const bottomSection = document.getElementById("bottomSection");
    bottomSection.style.display = "block";

    const originalCroppedImage = document.getElementById("originalCroppedImage");
    const diagnosedImage = document.getElementById("diagnosedImage");
    const diagnosisText = document.getElementById("diagnosisText");
    const diagnosisText1 = document.getElementById("diagnosisText1");
    const diagnosisText2 = document.getElementById("diagnosisText2");

    originalCroppedImage.src = croppedImage;
    originalCroppedImage.hidden = false;

    if (data.result_image) {
        diagnosedImage.src = data.result_image;
        diagnosedImage.hidden = false;
    }

    if (data.advice) {
        diagnosisText.textContent = data.advice;
    }
    if (data.advice1) {
        diagnosisText1.textContent = data.advice1;
    }
    if (data.advice2) {
        diagnosisText2.textContent = data.advice2;
    }
}

function showPopup(triggerElement) {
    const popupElement = document.getElementById('popup');
    const popupImageElement = document.getElementById('popupImage');

    if (popupElement && popupImageElement) {
        popupImageElement.src = triggerElement.src;
        popupElement.style.display = 'flex';
        document.body.classList.add('popup-active');
    } else {
        console.error('Popup or image element is missing.');
    }
}

function closePopup(event) {
    const popupElement = document.getElementById('popup');
    if (popupElement && (event.target === popupElement || event.target.classList.contains('popup-close'))) {
        popupElement.style.display = 'none';
        document.body.classList.remove('popup-active');
    }
}

function attachPopupHandlers(triggerElementId, popupElementId, imageElementId) {
    const triggerElement = document.getElementById(triggerElementId);
    const popupElement = document.getElementById(popupElementId);

    if (triggerElement && popupElement) {
        triggerElement.addEventListener('click', () => showPopup(triggerElementId, popupElementId, imageElementId));
        popupElement.addEventListener('click', (event) => closePopup(popupElementId, event));
    } else {
        console.error('Invalid element IDs provided.');
    }
}

function setupEventListeners() {
    document.getElementById("uploadImage").addEventListener("change", handleImageUpload);
    document.getElementById("diagnoseButton").addEventListener("click", handleDiagnose);
}

setupEventListeners();
