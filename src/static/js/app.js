/**
 * RAG Chat Application
 * 채팅 및 문서 관리 기능을 위한 클라이언트 로직
 */

/**
 * 채팅 앱 클래스
 */
class ChatApp {
    constructor() {
        this.chatMessages = document.getElementById('chatMessages');
        this.chatInput = document.getElementById('chatInput');
        this.sendButton = document.getElementById('sendButton');
        this.isLoading = false;

        this.init();
    }

    init() {
        // 이벤트 리스너 등록
        this.sendButton.addEventListener('click', () => this.sendMessage());
        this.chatInput.addEventListener('keydown', (e) => this.handleKeyDown(e));
        this.chatInput.addEventListener('input', () => this.autoResize());
    }

    handleKeyDown(e) {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            this.sendMessage();
        }
    }

    autoResize() {
        this.chatInput.style.height = 'auto';
        this.chatInput.style.height = Math.min(this.chatInput.scrollHeight, 200) + 'px';
    }

    async sendMessage() {
        const message = this.chatInput.value.trim();
        if (!message || this.isLoading) return;

        // 환영 메시지 제거
        const welcomeMessage = this.chatMessages.querySelector('.welcome-message');
        if (welcomeMessage) {
            welcomeMessage.remove();
        }

        // 사용자 메시지 표시
        this.addMessage('user', message);

        // 입력창 초기화
        this.chatInput.value = '';
        this.chatInput.style.height = 'auto';

        // 로딩 상태 시작
        this.setLoading(true);
        this.showTypingIndicator();

        try {
            const response = await fetch('/api/chat', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    question: message,
                    top_k: 5,
                }),
            });

            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            const data = await response.json();

            // 타이핑 인디케이터 제거
            this.hideTypingIndicator();

            // AI 응답 표시
            this.addMessage('assistant', data.answer, data.sources);

        } catch (error) {
            console.error('Error:', error);
            this.hideTypingIndicator();
            this.addMessage('assistant', '죄송합니다. 오류가 발생했습니다. 다시 시도해 주세요.');
        } finally {
            this.setLoading(false);
        }
    }

    addMessage(role, content, sources = []) {
        const messageEl = document.createElement('div');
        messageEl.className = `message ${role}`;

        const avatarIcon = role === 'user'
            ? '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2"></path><circle cx="12" cy="7" r="4"></circle></svg>'
            : '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"></circle><path d="M8 14s1.5 2 4 2 4-2 4-2"></path><line x1="9" y1="9" x2="9.01" y2="9"></line><line x1="15" y1="9" x2="15.01" y2="9"></line></svg>';

        const roleName = role === 'user' ? 'You' : 'RAG Chat';

        let html = `
            <div class="message-header">
                <div class="message-avatar">${avatarIcon}</div>
                <span class="message-role">${roleName}</span>
            </div>
            <div class="message-content">
                <div class="message-text">${this.escapeHtml(content)}</div>
            </div>
        `;

        // 소스가 있으면 추가
        if (sources && sources.length > 0) {
            html += this.createSourcesHtml(sources);
        }

        messageEl.innerHTML = html;
        this.chatMessages.appendChild(messageEl);

        // 소스 토글 이벤트 리스너 추가
        const toggleBtn = messageEl.querySelector('.sources-toggle');
        if (toggleBtn) {
            toggleBtn.addEventListener('click', () => this.toggleSources(toggleBtn));
        }

        // 스크롤
        this.scrollToBottom();
    }

    createSourcesHtml(sources) {
        const sourceCards = sources.map((source, index) => `
            <div class="source-card">
                <div class="source-header">
                    <span class="source-filename">${this.escapeHtml(source.filename)} (chunk ${source.chunk_index})</span>
                    <span class="source-score">${(source.relevance_score * 100).toFixed(1)}%</span>
                </div>
                <div class="source-preview">${this.escapeHtml(source.content_preview)}</div>
            </div>
        `).join('');

        return `
            <div class="sources-container">
                <button class="sources-toggle">
                    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                        <polyline points="6 9 12 15 18 9"></polyline>
                    </svg>
                    <span>참조 소스 ${sources.length}개</span>
                </button>
                <div class="sources-list">
                    ${sourceCards}
                </div>
            </div>
        `;
    }

    toggleSources(button) {
        button.classList.toggle('expanded');
        const sourcesList = button.nextElementSibling;
        sourcesList.classList.toggle('expanded');
    }

    showTypingIndicator() {
        const indicator = document.createElement('div');
        indicator.className = 'typing-indicator';
        indicator.id = 'typingIndicator';
        indicator.innerHTML = '<span></span><span></span><span></span>';
        this.chatMessages.appendChild(indicator);
        this.scrollToBottom();
    }

    hideTypingIndicator() {
        const indicator = document.getElementById('typingIndicator');
        if (indicator) {
            indicator.remove();
        }
    }

    setLoading(loading) {
        this.isLoading = loading;
        this.sendButton.disabled = loading;
        this.chatInput.disabled = loading;
    }

    scrollToBottom() {
        this.chatMessages.scrollTop = this.chatMessages.scrollHeight;
    }

    escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }
}

/**
 * 문서 관리 클래스
 */
class DocumentManager {
    constructor() {
        this.dropzone = document.getElementById('uploadDropzone');
        this.fileInput = document.getElementById('fileInput');
        this.uploadProgress = document.getElementById('uploadProgress');
        this.progressFill = document.getElementById('progressFill');
        this.progressText = document.getElementById('progressText');
        this.documentsTableBody = document.getElementById('documentsTableBody');
        this.emptyState = document.getElementById('emptyState');
        this.loadingState = document.getElementById('loadingState');
        this.refreshButton = document.getElementById('refreshButton');
        this.deleteModal = document.getElementById('deleteModal');
        this.deleteFilename = document.getElementById('deleteFilename');
        this.cancelDeleteBtn = document.getElementById('cancelDelete');
        this.confirmDeleteBtn = document.getElementById('confirmDelete');

        this.documentToDelete = null;

        this.init();
    }

    init() {
        // 드래그 앤 드롭 이벤트
        this.dropzone.addEventListener('click', () => this.fileInput.click());
        this.dropzone.addEventListener('dragover', (e) => this.handleDragOver(e));
        this.dropzone.addEventListener('dragleave', () => this.handleDragLeave());
        this.dropzone.addEventListener('drop', (e) => this.handleDrop(e));
        this.fileInput.addEventListener('change', (e) => this.handleFileSelect(e));

        // 새로고침 버튼
        this.refreshButton.addEventListener('click', () => this.loadDocuments());

        // 모달 이벤트
        this.cancelDeleteBtn.addEventListener('click', () => this.hideDeleteModal());
        this.confirmDeleteBtn.addEventListener('click', () => this.confirmDelete());
        this.deleteModal.querySelector('.modal-backdrop').addEventListener('click', () => this.hideDeleteModal());

        // 문서 목록 로드
        this.loadDocuments();
    }

    handleDragOver(e) {
        e.preventDefault();
        this.dropzone.classList.add('dragover');
    }

    handleDragLeave() {
        this.dropzone.classList.remove('dragover');
    }

    handleDrop(e) {
        e.preventDefault();
        this.dropzone.classList.remove('dragover');

        const files = e.dataTransfer.files;
        if (files.length > 0) {
            this.uploadFile(files[0]);
        }
    }

    handleFileSelect(e) {
        const files = e.target.files;
        if (files.length > 0) {
            this.uploadFile(files[0]);
        }
        // 파일 입력 초기화 (같은 파일 다시 선택 가능하도록)
        this.fileInput.value = '';
    }

    async uploadFile(file) {
        // 파일 형식 확인
        const validExtensions = ['txt', 'md', 'json'];
        const extension = file.name.split('.').pop().toLowerCase();
        if (!validExtensions.includes(extension)) {
            this.showError(`지원하지 않는 파일 형식입니다. 지원 형식: ${validExtensions.join(', ')}`);
            return;
        }

        // 업로드 진행 상태 표시
        this.uploadProgress.style.display = 'block';
        this.progressFill.style.width = '0%';
        this.progressText.textContent = '업로드 중...';

        const formData = new FormData();
        formData.append('file', file);

        try {
            // 진행 상태 시뮬레이션 (실제로는 XHR을 사용해야 정확한 진행률 표시 가능)
            this.progressFill.style.width = '30%';

            const response = await fetch('/api/documents', {
                method: 'POST',
                body: formData,
            });

            this.progressFill.style.width = '70%';

            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.detail?.message || '업로드 실패');
            }

            const data = await response.json();

            this.progressFill.style.width = '100%';
            this.progressText.textContent = `업로드 완료! ${data.chunk_count}개의 청크가 생성되었습니다.`;

            // 문서 목록 새로고침
            setTimeout(() => {
                this.uploadProgress.style.display = 'none';
                this.loadDocuments();
            }, 1500);

        } catch (error) {
            console.error('Upload error:', error);
            this.progressFill.style.width = '0%';
            this.progressText.textContent = `오류: ${error.message}`;

            setTimeout(() => {
                this.uploadProgress.style.display = 'none';
            }, 3000);
        }
    }

    async loadDocuments() {
        // 로딩 상태 표시
        this.documentsTableBody.innerHTML = '';
        this.emptyState.style.display = 'none';
        this.loadingState.style.display = 'flex';

        try {
            const response = await fetch('/api/documents');
            if (!response.ok) {
                throw new Error('Failed to load documents');
            }

            const data = await response.json();
            this.loadingState.style.display = 'none';

            if (data.documents.length === 0) {
                this.emptyState.style.display = 'flex';
                return;
            }

            this.renderDocuments(data.documents);

        } catch (error) {
            console.error('Load error:', error);
            this.loadingState.style.display = 'none';
            this.showError('문서 목록을 불러오는 데 실패했습니다.');
        }
    }

    renderDocuments(documents) {
        this.documentsTableBody.innerHTML = documents.map(doc => `
            <tr data-id="${doc.id}">
                <td>${this.escapeHtml(doc.filename)}</td>
                <td>${doc.format.toUpperCase()}</td>
                <td>${this.formatFileSize(doc.file_size)}</td>
                <td>${doc.chunk_count}</td>
                <td>${this.formatDate(doc.created_at)}</td>
                <td>
                    <button class="delete-button" onclick="documentManager.showDeleteModal('${doc.id}', '${this.escapeHtml(doc.filename)}')">
                        삭제
                    </button>
                </td>
            </tr>
        `).join('');
    }

    showDeleteModal(id, filename) {
        this.documentToDelete = id;
        this.deleteFilename.textContent = filename;
        this.deleteModal.style.display = 'flex';
    }

    hideDeleteModal() {
        this.deleteModal.style.display = 'none';
        this.documentToDelete = null;
    }

    async confirmDelete() {
        if (!this.documentToDelete) return;

        const id = this.documentToDelete;
        this.hideDeleteModal();

        try {
            const response = await fetch(`/api/documents/${id}`, {
                method: 'DELETE',
            });

            if (!response.ok) {
                throw new Error('Failed to delete document');
            }

            // 문서 목록 새로고침
            this.loadDocuments();

        } catch (error) {
            console.error('Delete error:', error);
            this.showError('문서 삭제에 실패했습니다.');
        }
    }

    showError(message) {
        // 에러 메시지 표시
        const existingError = document.querySelector('.error-message');
        if (existingError) {
            existingError.remove();
        }

        const errorEl = document.createElement('div');
        errorEl.className = 'error-message';
        errorEl.textContent = message;

        const uploadSection = document.querySelector('.upload-section');
        uploadSection.appendChild(errorEl);

        setTimeout(() => {
            errorEl.remove();
        }, 5000);
    }

    formatFileSize(bytes) {
        if (bytes < 1024) return bytes + ' B';
        if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB';
        return (bytes / (1024 * 1024)).toFixed(1) + ' MB';
    }

    formatDate(dateString) {
        const date = new Date(dateString);
        return date.toLocaleDateString('ko-KR', {
            year: 'numeric',
            month: '2-digit',
            day: '2-digit',
        });
    }

    escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }
}

// 전역에서 접근 가능하도록 인스턴스 저장
let chatApp;
let documentManager;
