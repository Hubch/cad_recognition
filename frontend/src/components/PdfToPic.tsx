import * as pdfjsLib from 'pdfjs-dist';

// 配置pdfjs-lib以使用Web Worker
pdfjsLib.GlobalWorkerOptions.workerSrc = `https://cdnjs.cloudflare.com/ajax/libs/pdf.js/${pdfjsLib.version}/pdf.worker.entry.js`;

/**
 * 将PDF文件的第一页转换为图片的Base64 URL
 * @param {File} file - 用户上传的PDF文件
 * @returns {Promise<string>} - 图片的Base64 URL
 */
async function pdfToDataURL(file:File) {
    const typedArray = await readAsArrayBuffer(file) as ArrayBuffer;
    const loadingTask = pdfjsLib.getDocument({ data: typedArray });
    const pdf = await loadingTask.promise;
    const firstPage = await pdf.getPage(1);
    const scale = 1.0; // 可以调整比例
    const viewport = firstPage.getViewport({ scale });
    const canvas = document.createElement('canvas');
    const context = canvas.getContext('2d');
    canvas.height = viewport.height;
    canvas.width = viewport.width;

    if (!context) {
        throw new Error('Unable to get canvas context');
    }
    
    const renderContext = {
        canvasContext: context,
        viewport: viewport,
    };

    await firstPage.render(renderContext).promise;

    return canvas.toDataURL('image/png');
}

/**
 * 读取文件为ArrayBuffer
 * @param {File} file - 文件对象
 * @returns {Promise<ArrayBuffer>} - ArrayBuffer
 */
function readAsArrayBuffer(file:File) {
    return new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.onload = () => resolve(reader.result as ArrayBuffer);
        reader.onerror = (error) => reject(error);
        reader.readAsArrayBuffer(file);
    });
}

export { pdfToDataURL, readAsArrayBuffer };