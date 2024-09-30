document.getElementById('submit-button').addEventListener('click', function() {
    const prompt = document.getElementById('prompt').value;
    if (!prompt) {
        alert('Please enter a question.');
        return;
    }

    fetch('/generate-response', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({ prompt: prompt }),
    })
    .then(response => response.json())
    .then(data => {
        if (data.error) {
            document.getElementById('response-container').innerHTML = `<p style="color: red;">Error: ${data.error}</p>`;
        } else {
            document.getElementById('response-container').innerHTML = `<p>${data.response}</p>`;
        }
    })
    .catch(error => {
        console.error('Error:', error);
        document.getElementById('response-container').innerHTML = `<p style="color: red;">An error occurred. Please try again later.</p>`;
    });
});
