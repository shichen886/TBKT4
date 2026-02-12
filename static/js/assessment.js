document.addEventListener('DOMContentLoaded', function() {
    const uploadArea = document.getElementById('uploadArea');
    const imageInput = document.getElementById('imageInput');
    const previewImage = document.getElementById('previewImage');
    const uploadPlaceholder = document.querySelector('.upload-placeholder');
    
    const ocrEnabled = document.getElementById('ocrEnabled');
    const ocrEngineSection = document.getElementById('ocrEngineSection');
    const ocrEngine = document.getElementById('ocrEngine');
    const recognizeBtn = document.getElementById('recognizeBtn');
    
    const resultText = document.getElementById('resultText');
    const editBtn = document.getElementById('editBtn');
    const manualInput = document.getElementById('manualInput');
    const analyzeBtn = document.getElementById('analyzeBtn');
    const saveBtn = document.getElementById('saveBtn');
    const skillId = document.getElementById('skillId');
    const difficulty = document.getElementById('difficulty');
    const datasetName = document.getElementById('datasetName');
    
    console.log('saveBtn元素:', saveBtn);
    console.log('saveBtn初始disabled状态:', saveBtn.disabled);
    
    const analysisResults = document.getElementById('analysisResults');
    const analysisContent = document.getElementById('analysisContent');
    
    let uploadedImage = null;
    
    uploadArea.addEventListener('click', () => imageInput.click());
    
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
        if (file && file.type.startsWith('image/')) {
            handleImageUpload(file);
        }
    });
    
    imageInput.addEventListener('change', (e) => {
        const file = e.target.files[0];
        if (file) {
            handleImageUpload(file);
        }
    });
    
    ocrEnabled.addEventListener('change', function() {
        ocrEngineSection.style.display = this.checked ? 'block' : 'none';
        recognizeBtn.disabled = !this.checked || !uploadedImage;
    });
    
    recognizeBtn.addEventListener('click', performOCR);
    editBtn.addEventListener('click', () => {
        resultText.readOnly = false;
        resultText.focus();
    });
    analyzeBtn.addEventListener('click', analyzeQuestion);
    saveBtn.addEventListener('click', saveQuestion);
    
    // 监听手动输入，当有内容时启用保存按钮
    manualInput.addEventListener('input', function() {
        if (this.value.trim() && uploadedImage) {
            saveBtn.disabled = false;
        } else {
            saveBtn.disabled = true;
        }
    });
    
    function handleImageUpload(file) {
        const reader = new FileReader();
        reader.onload = (e) => {
            uploadedImage = e.target.result;
            previewImage.src = uploadedImage;
            previewImage.style.display = 'block';
            uploadPlaceholder.style.display = 'none';
            recognizeBtn.disabled = !ocrEnabled.checked;
            saveBtn.disabled = true;
        };
        reader.readAsDataURL(file);
    }
    
    async function performOCR() {
        if (!uploadedImage) {
            alert('请先上传图片');
            return;
        }
        
        recognizeBtn.disabled = true;
        recognizeBtn.textContent = '识别中...';
        
        try {
            const formData = new FormData();
            const blob = dataURLtoBlob(uploadedImage);
            console.log('图片Blob:', blob);
            console.log('图片类型:', blob.type);
            console.log('图片大小:', blob.size);
            formData.append('image', blob);
            formData.append('engine', ocrEngine.value);
            
            console.log('发送OCR请求到: /api/ocr/');
            console.log('OCR引擎:', ocrEngine.value);
            
            const response = await fetch('/api/ocr/', {
                method: 'POST',
                body: formData
            });
            
            console.log('响应状态:', response.status);
            const data = await response.json();
            console.log('响应数据:', data);
            
            if (data.success) {
                resultText.value = data.text;
                editBtn.disabled = false;
                saveBtn.disabled = false;
                console.log('识别成功，文本长度:', data.text.length);
                console.log('saveBtn.disabled:', saveBtn.disabled);
                console.log('saveBtn元素:', saveBtn);
                alert('✅ 识别成功！');
            } else {
                console.error('识别失败:', data.error);
                alert('❌ 识别失败：' + (data.error || '未知错误'));
            }
        } catch (error) {
            console.error('OCR识别失败:', error);
            alert('❌ 识别失败，请检查网络连接');
        } finally {
            recognizeBtn.disabled = false;
            recognizeBtn.textContent = '开始识别';
        }
    }
    
    function analyzeQuestion() {
        const text = manualInput.value || resultText.value;
        
        if (!text.trim()) {
            alert('请输入或识别题目内容');
            return;
        }
        
        analysisContent.innerHTML = `
            <div class="analysis-item">
                <h4>题目内容</h4>
                <p>${text}</p>
            </div>
            <div class="analysis-item">
                <h4>知识点分析</h4>
                <p>正在分析题目涉及的知识点...</p>
            </div>
            <div class="analysis-item">
                <h4>难度评估</h4>
                <div class="difficulty-bar">
                    <div class="difficulty-fill" style="width: 60%"></div>
                </div>
                <p>中等难度</p>
            </div>
            <div class="analysis-item">
                <h4>建议</h4>
                <p>建议先复习相关知识点，再进行练习。</p>
            </div>
        `;
        
        analysisResults.style.display = 'block';
    }
    
    async function saveQuestion() {
        if (!uploadedImage) {
            alert('请先上传图片');
            return;
        }
        
        const content = resultText.value || manualInput.value;
        
        if (!content.trim()) {
            alert('请输入或识别题目内容');
            return;
        }
        
        saveBtn.disabled = true;
        saveBtn.textContent = '保存中...';
        
        try {
            const formData = new FormData();
            const blob = dataURLtoBlob(uploadedImage);
            formData.append('image', blob);
            formData.append('content', content);
            formData.append('skill_id', skillId.value);
            formData.append('difficulty', difficulty.value);
            formData.append('dataset_name', datasetName.value);
            
            console.log('发送保存题目请求到: /api/save-question/');
            console.log('题目内容:', content);
            console.log('知识点ID:', skillId.value);
            console.log('难度级别:', difficulty.value);
            console.log('数据集名称:', datasetName.value);
            
            const response = await fetch('/api/save-question/', {
                method: 'POST',
                body: formData
            });
            
            console.log('响应状态:', response.status);
            const data = await response.json();
            console.log('响应数据:', data);
            
            if (data.success) {
                alert('✅ 题目已保存！');
                console.log('保存成功:', data);
            } else {
                console.error('保存失败:', data.error);
                alert('❌ 保存失败：' + (data.error || '未知错误'));
            }
        } catch (error) {
            console.error('保存题目失败:', error);
            alert('❌ 保存失败，请检查网络连接');
        } finally {
            saveBtn.disabled = false;
            saveBtn.textContent = '保存题目';
        }
    }
    
    function dataURLtoBlob(dataURL) {
        const arr = dataURL.split(',');
        const mime = arr[0].match(/:(.*?);/)[1];
        const bstr = atob(arr[1]);
        let n = bstr.length;
        const u8arr = new Uint8Array(n);
        while (n--) {
            u8arr[n] = bstr.charCodeAt(n);
        }
        return new Blob([u8arr], { type: mime });
    }
});