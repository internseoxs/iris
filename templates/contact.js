document.getElementById('contactForm').addEventListener('submit', function(event) {
    event.preventDefault();

    const name = document.getElementById('name').value;
    const email = document.getElementById('email').value;
    const message = document.getElementById('message').value;

    // Here you can handle the form submission, e.g., sending data to a server
    alert(`Thank you for your message, ${name}!`);

    // Optionally, you could clear the form fields
    document.getElementById('contactForm').reset();
});
