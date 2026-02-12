document.addEventListener('DOMContentLoaded', function() {
    const uploadArea = document.getElementById('uploadArea');
    const csvInput = document.getElementById('csvInput');
    const fileInfo = document.getElementById('fileInfo');
    const fileName = document.getElementById('fileName');
    const fileSize = document.getElementById('fileSize');
    const datasetName = document.getElementById('datasetName');
    const separator = document.getElementById('separator');
    const uploadBtn = document.getElementById('uploadBtn');
    const uploadProgress = document.getElementById('uploadProgress');
    const progressFill = document.getElementById('progressFill');
    const progressText = document.getElementById('progressText');
    const uploadResult = document.getElementById('uploadResult');
    const resultContent = document.getElementById('resultContent');
    const dataPreview = document.getElementById('dataPreview');
    const previewTable = document.getElementById('previewTable');
    
    let selectedFile = null;
    
    uploadArea.addEventListener('click', () => csvInput.click());
    
    uploadArea.addEventListener('dragover', (e) => {
        e.preventDefault();
        uploadArea.classList.add('dragover');
    });
    
    uploadArea.addEventListener('dragleave', () => {
        uploadArea.classList.remove('dragover');
    });
    
    uploadArea.addEventListener('drop', (e) => {
        e.preventDefault();
        uploadArea.classList.remove('dragover');
        
        const file = e.dataTransfer.files[0];
        if (file && (file.name.endsWith('.csv') || file.name.endsWith('.tsv'))) {
            handleFileSelect(file);
        } else {
            alert('请上传 CSV 或 TSV 文件');
        }
    });
    
    csvInput.addEventListener('change', (e) => {
        const file = e.target.files[0];
        if (file) {
            handleFileSelect(file);
        }
    });
    
    datasetName.addEventListener('input', checkUploadReady);
    
    uploadBtn.addEventListener('click', uploadData);
    
    function handleFileSelect(file) {
        selectedFile = file;
        fileName.textContent = file.name;
        fileSize.textContent = formatFileSize(file.size);
        fileInfo.style.display = 'block';
        checkUploadReady();
        
        previewFile(file);
    }
    
    function checkUploadReady() {
        uploadBtn.disabled = !selectedFile || !datasetName.value.trim();
    }
    
    function formatFileSize(bytes) {
        if (bytes < 1024) return bytes + ' B';
        if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(2) + ' KB';
        return (bytes / (1024 * 1024)).toFixed(2) + ' MB';
    }
    
    function previewFile(file) {
        const reader = new FileReader();
        reader.onload = (e) => {
            const content = e.target.result;
            const lines = content.split('\n').slice(0, 6);
            
            let html = '<table class="preview-table"><thead><tr>';
            
            const headers = lines[0].split(separator.value === '\\t' ? '\t' : separator.value);
            headers.forEach(header => {
                html += `<th>${header}</th>`;
            });
            html += '</tr></thead><tbody>';
            
            for (let i = 1; i < lines.length && i < 6; i++) {
                if (lines[i].trim()) {
                    const values = lines[i].split(separator.value === '\\t' ? '\t' : separator.value);
                    html += '<tr>';
                    values.forEach(value => {
                        html += `<td>${value}</td>`;
                    });
                    html += '</tr>';
                }
            }
            
            html += '</tbody></table>';
            previewTable.innerHTML = html;
            dataPreview.style.display = 'block';
        };
        reader.readAsText(file);
    }
    
    async function uploadData() {
        if (!selectedFile || !datasetName.value.trim()) {
            alert('请选择文件并输入数据集名称');
            return;
        }
        
        uploadBtn.disabled = true;
        uploadBtn.textContent = '上传中...';
        uploadProgress.style.display = 'block';
        uploadResult.style.display = 'none';
        
        try {
            const formData = new FormData();
            formData.append('file', selectedFile);
            formData.append('dataset_name', datasetName.value.trim());
            formData.append('separator', separator.value);
            
            const xhr = new XMLHttpRequest();
            
            xhr.upload.addEventListener('progress', (e) => {
                if (e.lengthComputable) {
                    const percent = Math.round((e.loaded / e.total) * 100);
                    progressFill.style.width = percent + '%';
                    progressText.textContent = percent + '%';
                }
            });
            
            xhr.addEventListener('load', () => {
                if (xhr.status === 200) {
                    const response = JSON.parse(xhr.responseText);
                    if (response.success) {
                        showUploadResult(true, response);
                    } else {
                        showUploadResult(false, response);
                    }
                } else {
                    showUploadResult(false, { error: '上传失败' });
                }
                uploadBtn.disabled = false;
                uploadBtn.textContent = '上传数据';
            });
            
            xhr.addEventListener('error', () => {
                showUploadResult(false, { error: '网络错误' });
                uploadBtn.disabled = false;
                uploadBtn.textContent = '上传数据';
            });
            
            xhr.open('POST', '/api/upload/');
            xhr.send(formData);
            
        } catch (error) {
            console.error('上传失败:', error);
            showUploadResult(false, { error: '上传失败: ' + error.message });
            uploadBtn.disabled = false;
            uploadBtn.textContent = '上传数据';
        }
    }
    
    function showUploadResult(success, data) {
        uploadResult.style.display = 'block';
        
        if (success) {
            resultContent.innerHTML = `
                <div class="success-message">
                    <span class="success-icon">✅</span>
                    <h4>上传成功！</h4>
                </div>
                <div class="result-details">
                    <p><strong>数据集名称:</strong> ${data.dataset_name}</p>
                    <p><strong>记录数:</strong> ${data.record_count}</p>
                    <p><strong>用户数:</strong> ${data.user_count}</p>
                    <p><strong>题目数:</strong> ${data.item_count}</p>
                    <p><strong>知识点数:</strong> ${data.skill_count}</p>
                </div>
                <button onclick="location.href='/recommendation/'" class="primary-button">开始使用</button>
            `;
        } else {
            resultContent.innerHTML = `
                <div class="error-message">
                    <span class="error-icon">❌</span>
                    <h4>上传失败</h4>
                    <p>${data.error || '未知错误'}</p>
                </div>
                <button onclick="location.reload()" class="secondary-button">重新上传</button>
            `;
        }
    }
});