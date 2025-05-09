// filename: script.js
document.addEventListener('DOMContentLoaded', () => {
    const chatMessages = document.getElementById('chat-messages');
    const userInput = document.getElementById('user-input');
    const sendButton = document.getElementById('send-button');

    // Basic check if elements exist
    if (!chatMessages || !userInput || !sendButton) {
        console.error("Error: One or more chat elements (chat-messages, user-input, send-button) not found in the HTML.");
        // Optionally disable input/button if elements are missing
        if(userInput) userInput.disabled = true;
        if(sendButton) sendButton.disabled = true;
        // Display an error message in the chat area if possible
        if(chatMessages) {
             const errorElement = document.createElement('div');
             errorElement.style.color = 'red';
             errorElement.textContent = "Chat interface failed to load correctly. Please check HTML element IDs.";
             chatMessages.appendChild(errorElement);
        }
        return; // Stop script execution if critical elements are missing
    }


    // Function to add a message to the chat display
    function addMessage(sender, text) {
        const messageElement = document.createElement('div');
        messageElement.classList.add('message', sender); // 'user' or 'assistant'

        const bubble = document.createElement('div');
        bubble.classList.add('bubble');
        // Basic text setting
        bubble.textContent = text;

        messageElement.appendChild(bubble);
        chatMessages.appendChild(messageElement);
        // Scroll to the bottom
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }

    // Function to handle sending messages
    async function sendMessage() {
        const message = userInput.value.trim();

        if (message === "") {
             return; // Don't send empty messages
        }

        // Display user message immediately
        addMessage('user', message);
        userInput.value = ''; // Clear input field

        // Disable input/button while waiting for response
        userInput.disabled = true;
        sendButton.disabled = true;

        try {
            // Send message to backend
            const response = await fetch('/query', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ query: message }),
            });

            if (!response.ok) {
                 // Try to get error message from backend response if possible
                 let errorMsg = `Error: ${response.status} ${response.statusText}`;
                 try {
                     const errorData = await response.json();
                     errorMsg = errorData.error || JSON.stringify(errorData);
                 } catch (e) {
                     // Ignore if error response isn't JSON
                 }
                throw new Error(errorMsg); // Throw detailed error if possible
            }

            const data = await response.json();

            // Display assistant response
            if (data && data.response) {
                addMessage('assistant', data.response);
            } else {
                 addMessage('assistant', "Sorry, I received an unexpected response.");
                 console.error("Unexpected response data structure:", data); // Log unexpected structure
            }

        } catch (error) {
            console.error('Error sending/receiving message:', error);
            // Display specific error in chat
            addMessage('assistant', `Sorry, an error occurred: ${error.message}`);
        } finally {
             // Re-enable input/button regardless of success or failure
             userInput.disabled = false;
             sendButton.disabled = false;
             userInput.focus(); // Put cursor back in input field
        }
    }

    // --- Event Listeners ---
    sendButton.addEventListener('click', sendMessage);
    userInput.addEventListener('keypress', (event) => {
        if (event.key === 'Enter') {
            event.preventDefault(); // Prevent default Enter behavior (like submitting a form)
            sendMessage();
        }
    });

    // Add an initial greeting
     addMessage('assistant', 'Hello! Ask me anything about Photography.');

}); // End of DOMContentLoaded