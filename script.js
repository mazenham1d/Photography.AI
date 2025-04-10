// Get references to the HTML elements we need to interact with
const chatOutput = document.getElementById('chat-output');
const userInput = document.getElementById('user-input');
const sendButton = document.getElementById('send-button');
const loadingIndicator = document.getElementById('loading-indicator');

// --- IMPORTANT: Replace this with the actual URL of your backend API endpoint ---
const BACKEND_URL = 'http://127.0.0.1:5000/api/chat';


// Function to add a message to the chat output
function addMessage(sender, text) {
    const messageDiv = document.createElement('div'); // Create a new <div> element
    messageDiv.classList.add('message', sender === 'user' ? 'user-message' : 'ai-message'); // Add CSS classes

    const paragraph = document.createElement('p'); // Create a <p> element for the text
    paragraph.textContent = text; // Set the text content

    messageDiv.appendChild(paragraph); // Put the <p> inside the <div>
    chatOutput.appendChild(messageDiv); // Add the new message <div> to the chat output area

    // Scroll to the bottom of the chat output to see the latest message
    chatOutput.scrollTop = chatOutput.scrollHeight;
}

// Function to handle sending a message
async function handleSendMessage() {
    const userText = userInput.value.trim(); // Get text from input, remove whitespace

    if (userText === '') {
        return; // Do nothing if input is empty
    }

    // 1. Display the user's message immediately
    addMessage('user', userText);
    userInput.value = ''; // Clear the input field
    userInput.disabled = true; // Disable input while waiting for AI
    sendButton.disabled = true; // Disable button while waiting for AI
    loadingIndicator.style.display = 'block'; // Show the 'Thinking...' indicator

    // 2. Send the message to the backend and get the AI response
    try {
        // Use the 'fetch' API to send a POST request to your backend
        const response = await fetch(BACKEND_URL, {
            method: 'POST', // HTTP method
            headers: {
                'Content-Type': 'application/json', // Tell the backend we're sending JSON
            },
            // IMPORTANT: The structure of this body MUST match what your backend expects!
            body: JSON.stringify({ message: userText }), // Convert JS object to JSON string
        });

        // Check if the request was successful (status code 200-299)
        if (!response.ok) {
            // If not okay, throw an error to be caught by the catch block
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        // Parse the JSON response from the backend
        // IMPORTANT: The structure of this 'data' object depends on what your backend sends!
        // We assume it sends an object like: { "reply": "The AI's answer..." }
        const data = await response.json();

        // 3. Display the AI's response
        addMessage('ai', data.reply); // Adjust 'data.reply' if your backend uses a different key

    } catch (error) {
        // 4. Handle errors (network issue, backend error)
        console.error('Error fetching AI response:', error);
        addMessage('ai', "Sorry, I encountered an error. Please try again. (" + error.message + ")");
    } finally {
        // 5. Re-enable input and hide loading indicator regardless of success or error
        loadingIndicator.style.display = 'none'; // Hide 'Thinking...'
        userInput.disabled = false; // Re-enable input
        sendButton.disabled = false; // Re-enable button
        userInput.focus(); // Put the cursor back in the input field
    }
}

// --- Event Listeners ---

// Send message when the button is clicked
sendButton.addEventListener('click', handleSendMessage);

// Send message when the user presses 'Enter' in the input field
userInput.addEventListener('keypress', function(event) {
    // Check if the key pressed was 'Enter'
    if (event.key === 'Enter') {
        handleSendMessage(); // Call the same function as the button click
    }
});