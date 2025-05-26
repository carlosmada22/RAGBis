document.addEventListener('DOMContentLoaded', function() {
    const chatForm = document.getElementById('chat-form');
    const userInput = document.getElementById('user-input');
    const chatMessages = document.getElementById('chat-messages');

    // Session management
    let sessionId = localStorage.getItem('chatSessionId') || null;

    // Function to add a message to the chat
    function addMessage(content, isUser = false) {
        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${isUser ? 'user' : 'assistant'}`;

        const messageContent = document.createElement('div');
        messageContent.className = 'message-content';

        const messageParagraph = document.createElement('p');
        messageParagraph.textContent = content;

        messageContent.appendChild(messageParagraph);
        messageDiv.appendChild(messageContent);
        chatMessages.appendChild(messageDiv);

        // Scroll to the bottom of the chat
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }

    // Function to add a loading indicator
    function addLoadingIndicator() {
        const loadingDiv = document.createElement('div');
        loadingDiv.className = 'message assistant';
        loadingDiv.id = 'loading-message';

        const loadingContent = document.createElement('div');
        loadingContent.className = 'loading';

        for (let i = 0; i < 3; i++) {
            const dot = document.createElement('span');
            loadingContent.appendChild(dot);
        }

        loadingDiv.appendChild(loadingContent);
        chatMessages.appendChild(loadingDiv);

        // Scroll to the bottom of the chat
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }

    // Function to remove the loading indicator
    function removeLoadingIndicator() {
        const loadingMessage = document.getElementById('loading-message');
        if (loadingMessage) {
            loadingMessage.remove();
        }
    }

    // Function to send a message to the server
    async function sendMessage(message) {
        try {
            // Add loading indicator
            addLoadingIndicator();

            // Prepare request body with session management
            const requestBody = {
                query: message,
                session_id: sessionId
            };

            // Send the message to the server
            const response = await fetch('/api/chat', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(requestBody)
            });

            // Remove loading indicator
            removeLoadingIndicator();

            // Parse the response
            const data = await response.json();

            if (data.success) {
                // Update session ID if provided
                if (data.session_id) {
                    sessionId = data.session_id;
                    localStorage.setItem('chatSessionId', sessionId);
                }

                // Add the assistant's response to the chat
                addMessage(data.answer);

                // Log metadata for debugging (optional)
                if (data.metadata) {
                    console.log('Chat metadata:', data.metadata);
                }
            } else {
                // Add an error message to the chat
                addMessage('Sorry, I encountered an error: ' + (data.error || 'Unknown error'));
            }
        } catch (error) {
            // Remove loading indicator
            removeLoadingIndicator();

            // Add an error message to the chat
            addMessage('Sorry, I encountered an error: ' + error.message);
        }
    }

    // Function to clear chat history
    function clearChatHistory() {
        if (sessionId && confirm('Are you sure you want to clear the chat history?')) {
            fetch(`/api/chat/clear/${sessionId}`, {
                method: 'POST'
            })
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    // Clear the chat messages display
                    chatMessages.innerHTML = `
                        <div class="message assistant">
                            <div class="message-content">
                                <p>Hello! I'm the openBIS Assistant. How can I help you today?</p>
                            </div>
                        </div>
                    `;
                    // Clear session from localStorage
                    localStorage.removeItem('chatSessionId');
                    sessionId = null;
                } else {
                    console.error('Failed to clear chat history:', data.error);
                }
            })
            .catch(error => {
                console.error('Error clearing chat history:', error);
            });
        }
    }

    // Handle form submission
    chatForm.addEventListener('submit', function(event) {
        event.preventDefault();

        const message = userInput.value.trim();
        if (message) {
            // Add the user's message to the chat
            addMessage(message, true);

            // Clear the input field
            userInput.value = '';

            // Send the message to the server
            sendMessage(message);
        }
    });

    // Handle clear chat button
    const clearChatBtn = document.getElementById('clear-chat-btn');
    if (clearChatBtn) {
        clearChatBtn.addEventListener('click', clearChatHistory);
    }

    // Focus the input field when the page loads
    userInput.focus();
});
