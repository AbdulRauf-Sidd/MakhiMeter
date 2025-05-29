function uploadImage() {
    document.getElementById('imageUpload').click(); // Trigger file input
}

let originalImageData = null; // Variable to store the original image data

function handleImageUpload(event) {
    const file = event.target.files[0];
    const uploadedImage = document.getElementById('uploadedImage');
    const placeholderText = document.getElementById('placeholderText');
    const hiddenCanvas = document.createElement('canvas'); // Temporary canvas for processing

    if (file) {
        const reader = new FileReader();
        reader.onload = function (e) {
            uploadedImage.src = e.target.result; // Set uploaded image source
            uploadedImage.style.display = 'block'; // Show the image
            placeholderText.style.display = 'none'; // Hide placeholder text

            // Wait for the image to load before storing its data
            uploadedImage.onload = function () {
                // Store the original image data
                hiddenCanvas.width = uploadedImage.naturalWidth;
                hiddenCanvas.height = uploadedImage.naturalHeight;
                const ctx = hiddenCanvas.getContext('2d');
                ctx.drawImage(uploadedImage, 0, 0, hiddenCanvas.width, hiddenCanvas.height);
                originalImageData = ctx.getImageData(0, 0, hiddenCanvas.width, hiddenCanvas.height);
            };
        };
        reader.readAsDataURL(file);
    }
}

function updateBackend(value) {
    const uploadedImage = document.getElementById('uploadedImage');
    const hiddenCanvas = document.createElement('canvas'); // Temporary canvas for reprocessing
    const histogramCanvas = document.getElementById('histogram');

    if (originalImageData) {
        // Reprocess the original image data with the new intensity adjustment
        adjustImageIntensity(originalImageData, hiddenCanvas, histogramCanvas, value);
    } else {
        console.log("No image uploaded yet.");
    }
}

function adjustImageIntensity(originalData, hiddenCanvas, histogramCanvas, adjustmentValue) {
    // Create a copy of the original image data to work on
    const imageData = new ImageData(
        new Uint8ClampedArray(originalData.data), // Clone original pixel data
        originalData.width,
        originalData.height
    );

    const ctx = hiddenCanvas.getContext('2d');
    hiddenCanvas.width = originalData.width;
    hiddenCanvas.height = originalData.height;

    // Adjust pixel intensity based on the slider value
    const data = imageData.data;
    const histogram = new Array(256).fill(0); // Initialize histogram
    for (let i = 0; i < data.length; i += 4) {
        const grayscale = Math.round(0.299 * data[i] + 0.587 * data[i + 1] + 0.114 * data[i + 2]); // Grayscale value
        let adjustedGrayscale = Math.min(255, Math.max(0, grayscale + parseInt(adjustmentValue))); // Clamp to [0, 255]

        // Update pixel values
        data[i] = data[i + 1] = data[i + 2] = adjustedGrayscale; // Set R, G, B to adjusted grayscale

        // Update histogram
        histogram[adjustedGrayscale]++;
    }

    // Update the canvas with adjusted image
    ctx.putImageData(imageData, 0, 0);

    // Display the adjusted image back in the placeholder
    const updatedImage = hiddenCanvas.toDataURL();
    const uploadedImage = document.getElementById('uploadedImage');
    uploadedImage.src = updatedImage;

    // Render the updated histogram
    renderHistogram(histogram, histogramCanvas);
}

function renderHistogram(histogram, canvas) {
    const ctx = canvas.getContext('2d');
    const labels = Array.from({ length: 256 }, (_, i) => i); // Intensity values 0-255

    if (window.histogramChart) {
        window.histogramChart.destroy(); // Destroy previous chart if it exists
    }

    window.histogramChart = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: labels,
            datasets: [
                {
                    label: 'Number of Pixels',
                    data: histogram,
                    backgroundColor: 'rgba(75, 192, 192, 0.5)',
                    borderColor: 'rgba(75, 192, 192, 1)',
                    borderWidth: 1,
                },
            ],
        },
        options: {
            responsive: true,
            scales: {
                x: {
                    title: {
                        display: true,
                        text: 'Intensity (0-255)',
                    },
                },
                y: {
                    beginAtZero: true,
                    title: {
                        display: true,
                        text: 'Number of Pixels',
                    },
                },
            },
        },
    });
}


// async function postData() {
//     const fileInput = document.getElementById('imageUpload'); // Get the image input
//     const flyIdInput = document.getElementById('fly-id'); // Get the Fly ID input
//     const csrfToken = document.getElementById('csrfToken').value;
//     const flyId = flyIdInput.value; // Get the Fly ID value
//     const file = fileInput.files[0]; // Get the selected file

//     if (!file || !flyId) {
//         alert("Please upload an image and provide a Fly ID.");
//         return;
//     }

//     const formData = new FormData();
//     formData.append('image', file); // Append the image file
//     formData.append('fly_id', flyId); // Append the Fly ID

//     try {
//         const response = await fetch('/wing/input/', {
//             method: 'POST',
//             body: formData,
//             headers: {
//                 'X-CSRFToken': csrfToken, // Include CSRF token in the request headers
//             },
//         });

//         if (response.ok) {
//             const data = await response.json(); // Assuming backend sends JSON response
//             console.log('Response from backend:', data);
//             // Redirect to the next page or display a success message
//             // window.location.href = '/segment-results/';
//         } else {
//             console.error('Error posting data:', response.statusText);
//             alert('Failed to process the image. Please try again.');
//         }
//     } catch (error) {
//         console.error('Network error:', error);
//         alert('A network error occurred. Please try again.');
//     }
// }