function goBack() {
    // Redirect back to the previous page
    window.history.back();
}

function downloadResults() {
    const imgSrc = document.getElementById('resultImage').src;
    const areaData = [];
    
    // Collect area data from the table
    const rows = document.querySelectorAll('.segment-table tbody tr');
    rows.forEach(row => {
        const cells = row.querySelectorAll('td');
        areaData.push({
            segment: cells[0].innerText,
            segment_name: cells[1].innerText,
            area: cells[2].innerText
        });
    });

    const payload = {
        image_url: imgSrc,
        area_data: areaData
    };

    fetch('/wing/download/', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
            'X-CSRFToken': getCSRFToken()
        },
        body: JSON.stringify(payload)
    })
    .then(response => {
        if (!response.ok) {
            throw new Error('Failed to download PDF');
        }
        return response.blob();
    })
    .then(blob => {
        const downloadUrl = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = downloadUrl;
        a.download = 'wing_results.pdf';
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
    })
    .catch(error => {
        alert('Error: ' + error.message);
    });
}