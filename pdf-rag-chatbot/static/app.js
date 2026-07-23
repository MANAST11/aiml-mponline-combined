document.addEventListener("DOMContentLoaded", () => {
    // DOM Elements
    const dropZone = document.getElementById("drop-zone");
    const fileInput = document.getElementById("file-input");
    const documentList = document.getElementById("document-list");
    const chatForm = document.getElementById("chat-form");
    const chatInput = document.getElementById("chat-input");
    const chatMessages = document.getElementById("chat-messages");
    const typingIndicator = document.getElementById("typing-indicator");
    const clearChatBtn = document.getElementById("clear-chat-btn");
    const clearDbBtn = document.getElementById("clear-db-btn");

    // Fetch and display active documents in the sidebar
    async function loadDocuments() {
        try {
            const response = await fetch("/api/documents");
            if (!response.ok) throw new Error("Failed to load documents");
            
            const data = await response.json();
            renderDocumentList(data.documents);
        } catch (error) {
            console.error("Error fetching documents:", error);
            showToast("Could not load documents from server.", "error");
        }
    }

    // Render document list
    function renderDocumentList(documents) {
        if (!documents || documents.length === 0) {
            documentList.innerHTML = `
                <div class="doc-list-empty">
                    <i class="fa-regular fa-file-pdf"></i>
                    <p>No documents uploaded yet.</p>
                </div>
            `;
            return;
        }

        documentList.innerHTML = documents.map(doc => `
            <div class="doc-item">
                <i class="fa-solid fa-file-pdf doc-item-icon"></i>
                <div class="doc-item-info">
                    <div class="doc-item-name" title="${doc.name}">${doc.name}</div>
                    <div class="doc-item-size">${doc.size_kb} KB</div>
                </div>
            </div>
        `).join("");
    }

    // Upload PDF file
    async function uploadFile(file) {
        if (!file) return;
        
        if (file.type !== "application/pdf" && !file.name.endsWith(".pdf")) {
            showToast("Only PDF files are supported.", "error");
            return;
        }

        const formData = new FormData();
        formData.append("file", file);

        showToast(`Uploading ${file.name}...`, "info");

        try {
            const response = await fetch("/api/upload", {
                method: "POST",
                body: formData
            });

            const data = await response.json();

            if (!response.ok) {
                throw new Error(data.detail || "Upload failed");
            }

            showToast(`Indexed: ${file.name}`, "success");
            loadDocuments();

            // Append a friendly message explaining that document is ready
            appendSystemMessage(`📚 <strong>System:</strong> Successfully ingested and indexed <em>${file.name}</em>. You can now ask questions about this document!`);
        } catch (error) {
            console.error("Upload error:", error);
            showToast(`Error: ${error.message}`, "error");
        }
    }

    // Handle Drag & Drop Events
    dropZone.addEventListener("click", () => fileInput.click());
    
    fileInput.addEventListener("change", (e) => {
        if (e.target.files.length > 0) {
            uploadFile(e.target.files[0]);
            fileInput.value = ""; // Reset file input
        }
    });

    ["dragenter", "dragover"].forEach(eventName => {
        dropZone.addEventListener(eventName, (e) => {
            e.preventDefault();
            dropZone.classList.add("drag-over");
        }, false);
    });

    ["dragleave", "drop"].forEach(eventName => {
        dropZone.addEventListener(eventName, (e) => {
            e.preventDefault();
            dropZone.classList.remove("drag-over");
        }, false);
    });

    dropZone.addEventListener("drop", (e) => {
        const dt = e.dataTransfer;
        const files = dt.files;
        if (files.length > 0) {
            uploadFile(files[0]);
        }
    });

    // Chat Functions
    function appendMessage(text, isUser = false, sources = []) {
        const messageDiv = document.createElement("div");
        messageDiv.className = `message ${isUser ? "user-message" : "assistant-message"}`;
        
        // Structure the message body
        let contentHtml = `
            <div class="message-content">
                <div class="message-text">${formatMarkdown(text)}</div>
        `;

        // If assistant has sources, append citation buttons
        if (!isUser && sources && sources.length > 0) {
            contentHtml += `
                <div class="sources-container">
                    <div class="sources-title">Sources Citations:</div>
                    <div class="sources-list">
            `;
            
            // Deduplicate sources to avoid repeating same page/file badges
            const seenSources = new Set();
            sources.forEach((src, idx) => {
                const key = `${src.file_name}-p${src.page}`;
                if (!seenSources.has(key)) {
                    seenSources.add(key);
                    contentHtml += `
                        <button class="source-badge" data-snippet="${encodeURIComponent(src.snippet)}" data-score="${src.score}" title="Click to view reference text snippet">
                            <i class="fa-solid fa-circle-info"></i>
                            <span>${src.file_name} (Page ${src.page})</span>
                        </button>
                    `;
                }
            });

            contentHtml += `
                    </div>
                </div>
            `;
        }

        contentHtml += `</div>`;
        messageDiv.innerHTML = contentHtml;
        chatMessages.appendChild(messageDiv);
        scrollToBottom();

        // Add event listeners to newly created source badges
        if (!isUser && sources.length > 0) {
            messageDiv.querySelectorAll(".source-badge").forEach(badge => {
                badge.addEventListener("click", () => {
                    const snippet = decodeURIComponent(badge.getAttribute("data-snippet"));
                    const score = badge.getAttribute("data-score");
                    alert(`Reference Snippet (Relevance Score: ${score}):\n\n"${snippet}"`);
                });
            });
        }
    }

    function appendSystemMessage(htmlContent) {
        const messageDiv = document.createElement("div");
        messageDiv.className = "message system-message";
        messageDiv.innerHTML = `
            <div class="message-content">
                <div class="message-text">${htmlContent}</div>
            </div>
        `;
        chatMessages.appendChild(messageDiv);
        scrollToBottom();
    }

    // A lightweight text formatter to replace basic markdown formatting
    function formatMarkdown(text) {
        if (!text) return "";
        let formatted = text
            .replace(/&/g, "&amp;")
            .replace(/</g, "&lt;")
            .replace(/>/g, "&gt;")
            // Bold
            .replace(/\*\*(.*?)\*\*/g, "<strong>$1</strong>")
            .replace(/\*(.*?)\*/g, "<em>$1</em>")
            // Code block
            .replace(/`([^`]+)`/g, "<code>$1</code>")
            // New lines
            .replace(/\n/g, "<br>");
        return formatted;
    }

    function scrollToBottom() {
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }

    // Submit user queries
    chatForm.addEventListener("submit", async (e) => {
        e.preventDefault();
        const queryText = chatInput.value.trim();
        if (!queryText) return;

        // Add user message to UI
        appendMessage(queryText, true);
        chatInput.value = "";

        // Show typing indicator
        typingIndicator.style.display = "flex";
        scrollToBottom();

        try {
            const response = await fetch("/api/query", {
                method: "POST",
                headers: {
                    "Content-Type": "application/json"
                },
                body: JSON.stringify({ message: queryText })
            });

            const data = await response.json();

            // Hide typing indicator
            typingIndicator.style.display = "none";

            if (!response.ok) {
                throw new Error(data.detail || "Query failed");
            }

            // Append assistant response
            appendMessage(data.response, false, data.sources);
        } catch (error) {
            typingIndicator.style.display = "none";
            console.error("Query error:", error);
            appendMessage(`⚠️ Error: ${error.message}. Please verify that your Gemini API key is configured correctly in the .env file.`, false);
        }
    });

    // Clear chat handler
    clearChatBtn.addEventListener("click", () => {
        if (confirm("Are you sure you want to clear the conversation history?")) {
            // Keep the system welcome message but remove user messages
            const welcomeCard = chatMessages.firstElementChild;
            chatMessages.innerHTML = "";
            if (welcomeCard) {
                chatMessages.appendChild(welcomeCard);
            }
            showToast("Chat history cleared.", "info");
        }
    });

    // Reset Database handler
    if (clearDbBtn) {
        clearDbBtn.addEventListener("click", async () => {
            if (confirm("Are you sure you want to delete all uploaded PDFs and reset the database? This cannot be undone.")) {
                showToast("Clearing database...", "info");
                try {
                    const response = await fetch("/api/clear", {
                        method: "POST"
                    });
                    const data = await response.json();
                    if (!response.ok) {
                        throw new Error(data.detail || "Reset failed");
                    }
                    
                    showToast("Database successfully reset!", "success");
                    loadDocuments();
                    
                    // Clear chat history as well since database context is gone
                    const welcomeCard = chatMessages.firstElementChild;
                    chatMessages.innerHTML = "";
                    if (welcomeCard) {
                        chatMessages.appendChild(welcomeCard);
                    }
                    appendSystemMessage("🗑️ <strong>System:</strong> All documents and indexes have been cleared. Ready for fresh uploads!");
                } catch (error) {
                    console.error("Reset error:", error);
                    showToast(`Error: ${error.message}`, "error");
                }
            }
        });
    }

    // Helper toast notification system
    function showToast(message, type = "info") {
        const toastContainer = document.getElementById("toast-container");
        const toast = document.createElement("div");
        toast.className = `toast ${type}`;
        
        let icon = "fa-circle-info";
        if (type === "success") icon = "fa-circle-check";
        if (type === "error") icon = "fa-triangle-exclamation";

        toast.innerHTML = `
            <i class="fa-solid ${icon}"></i>
            <span>${message}</span>
        `;
        
        toastContainer.appendChild(toast);

        // Auto remove toast after 4 seconds
        setTimeout(() => {
            toast.style.animation = "slideIn 0.3s reverse forwards";
            setTimeout(() => toast.remove(), 300);
        }, 4000);
    }

    // Startup Initialization
    loadDocuments();
});
