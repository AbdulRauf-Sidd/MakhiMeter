function goBack() {
    // Redirect back to the previous page
    window.history.back();
}

function downloadResults() {
    // Trigger download for results (you can modify the URL as needed)
    const downloadUrl = "/download-results";
    window.location.href = downloadUrl;
}
