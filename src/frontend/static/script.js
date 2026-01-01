// Function to preview selected image before uploading
function previewImage(event) {
    const fileInput = event.target;
    const preview = document.getElementById('imagePreview');

    if (fileInput.files && fileInput.files[0]) {
        const reader = new FileReader();

        reader.onload = function (e) {
            preview.src = e.target.result;
            preview.style.display = 'block'; // Ensure the image is visible
        };

        reader.readAsDataURL(fileInput.files[0]);
    }
}

// Attach the event listener to the file input on the upload page
document.addEventListener('DOMContentLoaded', () => {
    const fileInput = document.querySelector('input[type="file"]');
    if (fileInput) {
        fileInput.addEventListener('change', previewImage);
    }
});
