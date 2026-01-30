(function () {
    'use strict';

    // ============================================================
    // Configuration
    // ============================================================
    const PROCESSING_WIDTH = 300;
    const DB_NAME = 'GymCardScannerDB';
    const STORE_NAME = 'cards';
    const DB_VERSION = 1;

    // ============================================================
    // State
    // ============================================================
    let db = null;
    let cameraStream = null;
    let facingMode = 'environment';
    let capturedCanvas = null;
    let cropCorners = []; // [{x,y},...] in image pixel coords (TL, TR, BR, BL)
    let croppedCanvas = null;
    let activeCorner = -1;
    let currentCardId = null;

    // ============================================================
    // DOM References
    // ============================================================
    const $ = (id) => document.getElementById(id);

    // ============================================================
    // Database (IndexedDB)
    // ============================================================
    const CardDB = {
        open() {
            return new Promise((resolve, reject) => {
                const req = indexedDB.open(DB_NAME, DB_VERSION);
                req.onupgradeneeded = (e) => {
                    const store = e.target.result.createObjectStore(STORE_NAME, {
                        keyPath: 'id',
                        autoIncrement: true,
                    });
                    store.createIndex('name', 'name', { unique: false });
                    store.createIndex('timestamp', 'timestamp', { unique: false });
                };
                req.onsuccess = () => {
                    db = req.result;
                    resolve(db);
                };
                req.onerror = () => reject(req.error);
            });
        },

        save(card) {
            return new Promise((resolve, reject) => {
                const tx = db.transaction(STORE_NAME, 'readwrite');
                const store = tx.objectStore(STORE_NAME);
                const req = store.add(card);
                req.onsuccess = () => resolve(req.result);
                req.onerror = () => reject(req.error);
            });
        },

        getAll() {
            return new Promise((resolve, reject) => {
                const tx = db.transaction(STORE_NAME, 'readonly');
                const store = tx.objectStore(STORE_NAME);
                const req = store.index('timestamp').openCursor(null, 'prev');
                const results = [];
                req.onsuccess = () => {
                    const cursor = req.result;
                    if (cursor) {
                        results.push(cursor.value);
                        cursor.continue();
                    } else {
                        resolve(results);
                    }
                };
                req.onerror = () => reject(req.error);
            });
        },

        get(id) {
            return new Promise((resolve, reject) => {
                const tx = db.transaction(STORE_NAME, 'readonly');
                const req = tx.objectStore(STORE_NAME).get(id);
                req.onsuccess = () => resolve(req.result);
                req.onerror = () => reject(req.error);
            });
        },

        delete(id) {
            return new Promise((resolve, reject) => {
                const tx = db.transaction(STORE_NAME, 'readwrite');
                const req = tx.objectStore(STORE_NAME).delete(id);
                req.onsuccess = () => resolve();
                req.onerror = () => reject(req.error);
            });
        },

        search(query) {
            const q = query.toLowerCase().trim();
            return this.getAll().then((cards) =>
                cards.filter(
                    (c) =>
                        c.name.toLowerCase().includes(q) ||
                        (c.notes && c.notes.toLowerCase().includes(q))
                )
            );
        },
    };

    // ============================================================
    // Image Processing - Edge Detection
    // ============================================================
    const EdgeDetect = {
        // Convert ImageData to grayscale Float32Array
        grayscale(imgData) {
            const d = imgData.data;
            const gray = new Float32Array(imgData.width * imgData.height);
            for (let i = 0; i < gray.length; i++) {
                const j = i * 4;
                gray[i] = 0.299 * d[j] + 0.587 * d[j + 1] + 0.114 * d[j + 2];
            }
            return gray;
        },

        // Gaussian blur (5x5 kernel, sigma ~1.4)
        gaussianBlur(src, w, h) {
            const kernel = [
                1, 4, 7, 4, 1, 4, 16, 26, 16, 4, 7, 26, 41, 26, 7, 4, 16, 26,
                16, 4, 1, 4, 7, 4, 1,
            ];
            const kSum = 273;
            const dst = new Float32Array(w * h);
            for (let y = 2; y < h - 2; y++) {
                for (let x = 2; x < w - 2; x++) {
                    let sum = 0;
                    for (let ky = -2; ky <= 2; ky++) {
                        for (let kx = -2; kx <= 2; kx++) {
                            sum +=
                                src[(y + ky) * w + (x + kx)] *
                                kernel[(ky + 2) * 5 + (kx + 2)];
                        }
                    }
                    dst[y * w + x] = sum / kSum;
                }
            }
            return dst;
        },

        // Sobel edge detection returning gradient magnitude
        sobel(src, w, h) {
            const mag = new Float32Array(w * h);
            for (let y = 1; y < h - 1; y++) {
                for (let x = 1; x < w - 1; x++) {
                    const idx = y * w + x;
                    const gx =
                        -src[(y - 1) * w + (x - 1)] -
                        2 * src[y * w + (x - 1)] -
                        src[(y + 1) * w + (x - 1)] +
                        src[(y - 1) * w + (x + 1)] +
                        2 * src[y * w + (x + 1)] +
                        src[(y + 1) * w + (x + 1)];
                    const gy =
                        -src[(y - 1) * w + (x - 1)] -
                        2 * src[(y - 1) * w + x] -
                        src[(y - 1) * w + (x + 1)] +
                        src[(y + 1) * w + (x - 1)] +
                        2 * src[(y + 1) * w + x] +
                        src[(y + 1) * w + (x + 1)];
                    mag[idx] = Math.sqrt(gx * gx + gy * gy);
                }
            }
            return mag;
        },

        // Adaptive threshold using mean of neighborhood
        threshold(src, w, h) {
            // Compute global Otsu threshold
            const hist = new Int32Array(256);
            for (let i = 0; i < src.length; i++) {
                hist[Math.min(255, Math.max(0, Math.round(src[i])))]++;
            }
            const total = src.length;
            let sumAll = 0;
            for (let i = 0; i < 256; i++) sumAll += i * hist[i];

            let sumBg = 0,
                wBg = 0,
                maxVar = 0,
                bestT = 0;
            for (let t = 0; t < 256; t++) {
                wBg += hist[t];
                if (wBg === 0) continue;
                const wFg = total - wBg;
                if (wFg === 0) break;
                sumBg += t * hist[t];
                const meanBg = sumBg / wBg;
                const meanFg = (sumAll - sumBg) / wFg;
                const variance = wBg * wFg * (meanBg - meanFg) * (meanBg - meanFg);
                if (variance > maxVar) {
                    maxVar = variance;
                    bestT = t;
                }
            }

            const out = new Uint8Array(w * h);
            for (let i = 0; i < src.length; i++) {
                out[i] = src[i] > bestT ? 255 : 0;
            }
            return out;
        },

        // Find the largest rectangular contour from edge map
        findCardCorners(edgeMap, w, h) {
            // Dilate edges to connect nearby ones
            const dilated = new Uint8Array(w * h);
            for (let y = 1; y < h - 1; y++) {
                for (let x = 1; x < w - 1; x++) {
                    if (
                        edgeMap[y * w + x] ||
                        edgeMap[(y - 1) * w + x] ||
                        edgeMap[(y + 1) * w + x] ||
                        edgeMap[y * w + (x - 1)] ||
                        edgeMap[y * w + (x + 1)]
                    ) {
                        dilated[y * w + x] = 255;
                    }
                }
            }

            // Find boundary points from each edge
            const margin = Math.round(w * 0.02);
            const topPoints = [];
            const bottomPoints = [];
            const leftPoints = [];
            const rightPoints = [];

            // Scan columns for top/bottom boundaries
            for (let x = margin; x < w - margin; x += 2) {
                for (let y = margin; y < h - margin; y++) {
                    if (dilated[y * w + x]) {
                        topPoints.push({ x, y });
                        break;
                    }
                }
                for (let y = h - margin - 1; y >= margin; y--) {
                    if (dilated[y * w + x]) {
                        bottomPoints.push({ x, y });
                        break;
                    }
                }
            }

            // Scan rows for left/right boundaries
            for (let y = margin; y < h - margin; y += 2) {
                for (let x = margin; x < w - margin; x++) {
                    if (dilated[y * w + x]) {
                        leftPoints.push({ x, y });
                        break;
                    }
                }
                for (let x = w - margin - 1; x >= margin; x--) {
                    if (dilated[y * w + x]) {
                        rightPoints.push({ x, y });
                        break;
                    }
                }
            }

            if (
                topPoints.length < 3 ||
                bottomPoints.length < 3 ||
                leftPoints.length < 3 ||
                rightPoints.length < 3
            ) {
                return null;
            }

            // Fit lines using median approach (robust to outliers)
            const fitLine = (points, horizontal) => {
                if (horizontal) {
                    points.sort((a, b) => a.y - b.y);
                    const medY = points[Math.floor(points.length / 2)].y;
                    return { horizontal: true, val: medY };
                } else {
                    points.sort((a, b) => a.x - b.x);
                    const medX = points[Math.floor(points.length / 2)].x;
                    return { horizontal: false, val: medX };
                }
            };

            // Use RANSAC-like line fitting for better results
            const ransacLine = (points, isVertical) => {
                const coord = isVertical ? 'x' : 'y';
                const values = points.map((p) => p[coord]);
                values.sort((a, b) => a - b);
                // Use interquartile range to reject outliers
                const q1 = values[Math.floor(values.length * 0.25)];
                const q3 = values[Math.floor(values.length * 0.75)];
                const iqr = q3 - q1;
                const filtered = values.filter(
                    (v) => v >= q1 - iqr * 1.5 && v <= q3 + iqr * 1.5
                );
                if (filtered.length === 0) return values[Math.floor(values.length / 2)];
                return filtered[Math.floor(filtered.length / 2)];
            };

            const topY = ransacLine(topPoints, false);
            const bottomY = ransacLine(bottomPoints, false);
            const leftX = ransacLine(leftPoints, true);
            const rightX = ransacLine(rightPoints, true);

            // Validate the detected rectangle
            const rectW = rightX - leftX;
            const rectH = bottomY - topY;
            if (rectW < w * 0.15 || rectH < h * 0.15) return null;
            if (rectW > w * 0.98 || rectH > h * 0.98) return null;

            return [
                { x: leftX, y: topY }, // TL
                { x: rightX, y: topY }, // TR
                { x: rightX, y: bottomY }, // BR
                { x: leftX, y: bottomY }, // BL
            ];
        },

        // Main detection pipeline
        detect(sourceCanvas) {
            const sw = sourceCanvas.width;
            const sh = sourceCanvas.height;
            const scale = PROCESSING_WIDTH / sw;
            const pw = PROCESSING_WIDTH;
            const ph = Math.round(sh * scale);

            // Downsample
            const procCanvas = document.createElement('canvas');
            procCanvas.width = pw;
            procCanvas.height = ph;
            const ctx = procCanvas.getContext('2d');
            ctx.drawImage(sourceCanvas, 0, 0, pw, ph);
            const imgData = ctx.getImageData(0, 0, pw, ph);

            // Process
            const gray = this.grayscale(imgData);
            const blurred = this.gaussianBlur(gray, pw, ph);
            const edges = this.sobel(blurred, pw, ph);
            const binary = this.threshold(edges, pw, ph);
            const corners = this.findCardCorners(binary, pw, ph);

            if (corners) {
                return corners.map((p) => ({
                    x: Math.round(p.x / scale),
                    y: Math.round(p.y / scale),
                }));
            }

            // Default: centered card rectangle (credit card ratio 1.586:1)
            const cardRatio = 1.586;
            let cardW, cardH;
            if (sw / sh > cardRatio) {
                cardH = sh * 0.7;
                cardW = cardH * cardRatio;
            } else {
                cardW = sw * 0.7;
                cardH = cardW / cardRatio;
            }
            const cx = sw / 2,
                cy = sh / 2;
            return [
                { x: Math.round(cx - cardW / 2), y: Math.round(cy - cardH / 2) },
                { x: Math.round(cx + cardW / 2), y: Math.round(cy - cardH / 2) },
                { x: Math.round(cx + cardW / 2), y: Math.round(cy + cardH / 2) },
                { x: Math.round(cx - cardW / 2), y: Math.round(cy + cardH / 2) },
            ];
        },
    };

    // ============================================================
    // Perspective Transform
    // ============================================================
    const PerspectiveWarp = {
        // Solve 8x8 linear system using Gaussian elimination
        solve8x8(A, b) {
            const n = 8;
            const aug = [];
            for (let i = 0; i < n; i++) {
                aug[i] = [...A[i], b[i]];
            }

            for (let col = 0; col < n; col++) {
                let maxRow = col;
                let maxVal = Math.abs(aug[col][col]);
                for (let row = col + 1; row < n; row++) {
                    if (Math.abs(aug[row][col]) > maxVal) {
                        maxVal = Math.abs(aug[row][col]);
                        maxRow = row;
                    }
                }
                [aug[col], aug[maxRow]] = [aug[maxRow], aug[col]];

                if (Math.abs(aug[col][col]) < 1e-10) return null;

                for (let row = col + 1; row < n; row++) {
                    const factor = aug[row][col] / aug[col][col];
                    for (let j = col; j <= n; j++) {
                        aug[row][j] -= factor * aug[col][j];
                    }
                }
            }

            const x = new Array(n);
            for (let i = n - 1; i >= 0; i--) {
                x[i] = aug[i][n];
                for (let j = i + 1; j < n; j++) {
                    x[i] -= aug[i][j] * x[j];
                }
                x[i] /= aug[i][i];
            }
            return x;
        },

        // Compute homography: src corners → dst corners
        computeHomography(src, dst) {
            const A = [];
            const b = [];
            for (let i = 0; i < 4; i++) {
                const sx = src[i].x,
                    sy = src[i].y;
                const dx = dst[i].x,
                    dy = dst[i].y;
                A.push([sx, sy, 1, 0, 0, 0, -dx * sx, -dx * sy]);
                b.push(dx);
                A.push([0, 0, 0, sx, sy, 1, -dy * sx, -dy * sy]);
                b.push(dy);
            }
            const h = this.solve8x8(A, b);
            if (!h) return null;
            return [h[0], h[1], h[2], h[3], h[4], h[5], h[6], h[7], 1];
        },

        // Apply homography to a point
        transformPoint(H, x, y) {
            const w = H[6] * x + H[7] * y + H[8];
            return {
                x: (H[0] * x + H[1] * y + H[2]) / w,
                y: (H[3] * x + H[4] * y + H[5]) / w,
            };
        },

        // Warp source canvas using perspective transform
        warp(srcCanvas, srcCorners) {
            // Compute output dimensions from the corners
            const widthTop = Math.hypot(
                srcCorners[1].x - srcCorners[0].x,
                srcCorners[1].y - srcCorners[0].y
            );
            const widthBot = Math.hypot(
                srcCorners[2].x - srcCorners[3].x,
                srcCorners[2].y - srcCorners[3].y
            );
            const heightLeft = Math.hypot(
                srcCorners[3].x - srcCorners[0].x,
                srcCorners[3].y - srcCorners[0].y
            );
            const heightRight = Math.hypot(
                srcCorners[2].x - srcCorners[1].x,
                srcCorners[2].y - srcCorners[1].y
            );

            const outW = Math.round(Math.max(widthTop, widthBot));
            const outH = Math.round(Math.max(heightLeft, heightRight));

            const dstCorners = [
                { x: 0, y: 0 },
                { x: outW, y: 0 },
                { x: outW, y: outH },
                { x: 0, y: outH },
            ];

            // Compute inverse homography (dst → src) for backward mapping
            const H = this.computeHomography(dstCorners, srcCorners);
            if (!H) return srcCanvas;

            const outCanvas = document.createElement('canvas');
            outCanvas.width = outW;
            outCanvas.height = outH;
            const outCtx = outCanvas.getContext('2d');
            const outImg = outCtx.createImageData(outW, outH);

            const srcCtx = srcCanvas.getContext('2d');
            const srcImg = srcCtx.getImageData(
                0, 0, srcCanvas.width, srcCanvas.height
            );
            const srcData = srcImg.data;
            const srcW = srcCanvas.width;
            const srcH = srcCanvas.height;
            const outData = outImg.data;

            for (let dy = 0; dy < outH; dy++) {
                for (let dx = 0; dx < outW; dx++) {
                    const sp = this.transformPoint(H, dx, dy);
                    const sx = Math.round(sp.x);
                    const sy = Math.round(sp.y);

                    const outIdx = (dy * outW + dx) * 4;
                    if (sx >= 0 && sx < srcW && sy >= 0 && sy < srcH) {
                        const srcIdx = (sy * srcW + sx) * 4;
                        outData[outIdx] = srcData[srcIdx];
                        outData[outIdx + 1] = srcData[srcIdx + 1];
                        outData[outIdx + 2] = srcData[srcIdx + 2];
                        outData[outIdx + 3] = 255;
                    }
                }
            }

            outCtx.putImageData(outImg, 0, 0);
            return outCanvas;
        },
    };

    // ============================================================
    // Camera
    // ============================================================
    const Camera = {
        async start() {
            const video = $('camera-feed');
            try {
                if (cameraStream) {
                    cameraStream.getTracks().forEach((t) => t.stop());
                }
                cameraStream = await navigator.mediaDevices.getUserMedia({
                    video: {
                        facingMode: facingMode,
                        width: { ideal: 1920 },
                        height: { ideal: 1080 },
                    },
                    audio: false,
                });
                video.srcObject = cameraStream;
                await video.play();
            } catch (err) {
                showToast('Camera access denied. Please allow camera permissions.');
                console.error('Camera error:', err);
            }
        },

        stop() {
            if (cameraStream) {
                cameraStream.getTracks().forEach((t) => t.stop());
                cameraStream = null;
            }
            const video = $('camera-feed');
            video.srcObject = null;
        },

        capture() {
            const video = $('camera-feed');
            const canvas = document.createElement('canvas');
            canvas.width = video.videoWidth;
            canvas.height = video.videoHeight;
            const ctx = canvas.getContext('2d');
            ctx.drawImage(video, 0, 0);
            return canvas;
        },

        switchCamera() {
            facingMode = facingMode === 'environment' ? 'user' : 'environment';
            this.start();
        },
    };

    // ============================================================
    // Canvas ↔ Image Blob Conversion
    // ============================================================
    function canvasToBlob(canvas, type = 'image/jpeg', quality = 0.85) {
        return new Promise((resolve) => {
            canvas.toBlob(resolve, type, quality);
        });
    }

    function blobToImageCanvas(blob) {
        return new Promise((resolve) => {
            const img = new Image();
            img.onload = () => {
                const c = document.createElement('canvas');
                c.width = img.width;
                c.height = img.height;
                c.getContext('2d').drawImage(img, 0, 0);
                URL.revokeObjectURL(img.src);
                resolve(c);
            };
            img.src = URL.createObjectURL(blob);
        });
    }

    // ============================================================
    // Corner Handle Interaction
    // ============================================================
    const CropUI = {
        containerRect: null,
        canvasRect: null,
        imageW: 0,
        imageH: 0,

        init(imageCanvas) {
            this.imageW = imageCanvas.width;
            this.imageH = imageCanvas.height;

            const cropCanvas = $('crop-canvas');
            cropCanvas.width = imageCanvas.width;
            cropCanvas.height = imageCanvas.height;
            cropCanvas.getContext('2d').drawImage(imageCanvas, 0, 0);

            // Detect corners
            cropCorners = EdgeDetect.detect(imageCanvas);

            // Wait a frame for layout, then position handles
            requestAnimationFrame(() => {
                this.updateLayout();
                this.positionHandles();
                this.drawOutline();
            });

            this.bindEvents();
        },

        updateLayout() {
            this.containerRect = $('crop-container').getBoundingClientRect();
            this.canvasRect = $('crop-canvas').getBoundingClientRect();
        },

        // Convert image pixel coords to container CSS coords
        imgToCSS(px, py) {
            const scaleX = this.canvasRect.width / this.imageW;
            const scaleY = this.canvasRect.height / this.imageH;
            const offsetX = this.canvasRect.left - this.containerRect.left;
            const offsetY = this.canvasRect.top - this.containerRect.top;
            return {
                x: px * scaleX + offsetX,
                y: py * scaleY + offsetY,
            };
        },

        // Convert container CSS coords to image pixel coords
        cssToImg(cx, cy) {
            const scaleX = this.canvasRect.width / this.imageW;
            const scaleY = this.canvasRect.height / this.imageH;
            const offsetX = this.canvasRect.left - this.containerRect.left;
            const offsetY = this.canvasRect.top - this.containerRect.top;
            return {
                x: Math.round(
                    Math.max(0, Math.min(this.imageW, (cx - offsetX) / scaleX))
                ),
                y: Math.round(
                    Math.max(0, Math.min(this.imageH, (cy - offsetY) / scaleY))
                ),
            };
        },

        positionHandles() {
            const handles = document.querySelectorAll('.corner-handle');
            handles.forEach((handle, i) => {
                const css = this.imgToCSS(cropCorners[i].x, cropCorners[i].y);
                handle.style.left = css.x + 'px';
                handle.style.top = css.y + 'px';
            });
        },

        drawOutline() {
            const svg = $('crop-outline');
            const points = cropCorners
                .map((c) => {
                    const css = this.imgToCSS(c.x, c.y);
                    return `${css.x},${css.y}`;
                })
                .join(' ');
            svg.innerHTML = `
                <polygon points="${points}"
                    fill="rgba(26,115,232,0.15)"
                    stroke="#1a73e8"
                    stroke-width="2"
                    stroke-dasharray="6,4"/>
            `;
        },

        bindEvents() {
            const handles = document.querySelectorAll('.corner-handle');
            const onStart = (i, e) => {
                e.preventDefault();
                activeCorner = i;
            };

            handles.forEach((handle, i) => {
                handle.addEventListener('mousedown', (e) => onStart(i, e));
                handle.addEventListener('touchstart', (e) => onStart(i, e), {
                    passive: false,
                });
            });

            const getPos = (e) => {
                const point = e.touches ? e.touches[0] : e;
                return {
                    x: point.clientX - this.containerRect.left,
                    y: point.clientY - this.containerRect.top,
                };
            };

            const onMove = (e) => {
                if (activeCorner < 0) return;
                e.preventDefault();
                const pos = getPos(e);
                const img = this.cssToImg(pos.x, pos.y);
                cropCorners[activeCorner] = img;
                this.positionHandles();
                this.drawOutline();
            };

            const onEnd = () => {
                activeCorner = -1;
            };

            document.addEventListener('mousemove', onMove);
            document.addEventListener('mouseup', onEnd);
            document.addEventListener('touchmove', onMove, { passive: false });
            document.addEventListener('touchend', onEnd);
        },
    };

    // ============================================================
    // Toast Notification
    // ============================================================
    let toastTimer = null;
    function showToast(message) {
        const toast = $('toast');
        toast.textContent = message;
        toast.classList.remove('hidden');
        clearTimeout(toastTimer);
        toastTimer = setTimeout(() => toast.classList.add('hidden'), 2500);
    }

    // ============================================================
    // View Navigation
    // ============================================================
    function navigateTo(viewName, title) {
        document.querySelectorAll('.view').forEach((v) => v.classList.remove('active'));
        $(('view-' + viewName)).classList.add('active');
        $('header-title').textContent = title || 'Gym Card Scanner';

        const showBack = viewName !== 'scanner' && viewName !== 'gallery';
        $('btn-back').classList.toggle('hidden', !showBack);

        if (viewName === 'scanner') {
            Camera.start();
        } else {
            Camera.stop();
        }
    }

    // ============================================================
    // Gallery Rendering
    // ============================================================
    async function renderGallery(cards) {
        const grid = $('gallery-grid');
        const empty = $('gallery-empty');
        const emptyMsg = $('empty-message');

        if (!cards) cards = await CardDB.getAll();

        grid.innerHTML = '';

        if (cards.length === 0) {
            grid.classList.add('hidden');
            empty.classList.remove('hidden');
            const query = $('search-input').value.trim();
            emptyMsg.innerHTML = query
                ? `No cards found for "${query}".`
                : 'No cards saved yet.<br>Scan your first card!';
            return;
        }

        grid.classList.remove('hidden');
        empty.classList.add('hidden');

        for (const card of cards) {
            const item = document.createElement('div');
            item.className = 'card-item';
            item.dataset.id = card.id;

            const canvas = document.createElement('canvas');
            const info = document.createElement('div');
            info.className = 'card-info';
            info.innerHTML = `
                <div class="card-name">${escapeHTML(card.name)}</div>
                <div class="card-date">${formatDate(card.timestamp)}</div>
            `;

            item.appendChild(canvas);
            item.appendChild(info);
            grid.appendChild(item);

            // Load thumbnail
            if (card.thumbnail) {
                const img = new Image();
                img.onload = () => {
                    canvas.width = img.width;
                    canvas.height = img.height;
                    canvas.getContext('2d').drawImage(img, 0, 0);
                    URL.revokeObjectURL(img.src);
                };
                img.src = URL.createObjectURL(card.thumbnail);
            }

            item.addEventListener('click', () => showDetail(card.id));
        }
    }

    // ============================================================
    // Detail View
    // ============================================================
    async function showDetail(id) {
        currentCardId = id;
        const card = await CardDB.get(id);
        if (!card) return;

        navigateTo('detail', card.name);

        $('detail-name').textContent = card.name;
        $('detail-notes').textContent = card.notes || '';
        $('detail-notes').classList.toggle('hidden', !card.notes);
        $('detail-date').textContent = 'Scanned: ' + formatDate(card.timestamp);

        // Render full image
        if (card.image) {
            const imgCanvas = await blobToImageCanvas(card.image);
            const detailCanvas = $('detail-canvas');
            detailCanvas.width = imgCanvas.width;
            detailCanvas.height = imgCanvas.height;
            detailCanvas.getContext('2d').drawImage(imgCanvas, 0, 0);
        }
    }

    // ============================================================
    // Utility Functions
    // ============================================================
    function escapeHTML(str) {
        const div = document.createElement('div');
        div.textContent = str;
        return div.innerHTML;
    }

    function formatDate(ts) {
        const d = new Date(ts);
        return d.toLocaleDateString(undefined, {
            year: 'numeric',
            month: 'short',
            day: 'numeric',
            hour: '2-digit',
            minute: '2-digit',
        });
    }

    function createThumbnail(canvas, maxWidth = 300) {
        const scale = maxWidth / canvas.width;
        const thumbCanvas = document.createElement('canvas');
        thumbCanvas.width = maxWidth;
        thumbCanvas.height = Math.round(canvas.height * scale);
        thumbCanvas
            .getContext('2d')
            .drawImage(canvas, 0, 0, thumbCanvas.width, thumbCanvas.height);
        return thumbCanvas;
    }

    // ============================================================
    // Event Handlers & Initialization
    // ============================================================
    async function init() {
        // Open database
        await CardDB.open();

        // Register service worker
        if ('serviceWorker' in navigator) {
            navigator.serviceWorker
                .register('/sw.js')
                .catch((err) => console.error('SW registration failed:', err));
        }

        // --- Scanner View ---
        $('btn-capture').addEventListener('click', () => {
            capturedCanvas = Camera.capture();
            navigateTo('crop', 'Adjust Crop');
            CropUI.init(capturedCanvas);
        });

        $('btn-gallery').addEventListener('click', () => {
            navigateTo('gallery', 'Saved Cards');
            renderGallery();
        });

        $('btn-switch-camera').addEventListener('click', () => {
            Camera.switchCamera();
        });

        // --- Crop View ---
        $('btn-retake').addEventListener('click', () => {
            navigateTo('scanner', 'Gym Card Scanner');
        });

        $('btn-crop-confirm').addEventListener('click', () => {
            croppedCanvas = PerspectiveWarp.warp(capturedCanvas, cropCorners);
            navigateTo('save', 'Save Card');

            const resultCanvas = $('result-canvas');
            resultCanvas.width = croppedCanvas.width;
            resultCanvas.height = croppedCanvas.height;
            resultCanvas.getContext('2d').drawImage(croppedCanvas, 0, 0);
            $('card-name').value = '';
            $('card-notes').value = '';
            $('card-name').focus();
        });

        // --- Save View ---
        $('save-form').addEventListener('submit', async (e) => {
            e.preventDefault();
            const name = $('card-name').value.trim();
            if (!name) return;

            const notes = $('card-notes').value.trim();
            const imageBlob = await canvasToBlob(croppedCanvas);
            const thumbCanvas = createThumbnail(croppedCanvas);
            const thumbBlob = await canvasToBlob(thumbCanvas, 'image/jpeg', 0.7);

            await CardDB.save({
                name: name,
                notes: notes || '',
                image: imageBlob,
                thumbnail: thumbBlob,
                timestamp: Date.now(),
            });

            showToast('Card saved!');
            navigateTo('gallery', 'Saved Cards');
            renderGallery();
        });

        // --- Gallery View ---
        let searchDebounce = null;
        $('search-input').addEventListener('input', () => {
            const query = $('search-input').value.trim();
            $('btn-clear-search').classList.toggle('hidden', !query);
            clearTimeout(searchDebounce);
            searchDebounce = setTimeout(async () => {
                if (query) {
                    const results = await CardDB.search(query);
                    renderGallery(results);
                } else {
                    renderGallery();
                }
            }, 200);
        });

        $('btn-clear-search').addEventListener('click', () => {
            $('search-input').value = '';
            $('btn-clear-search').classList.add('hidden');
            renderGallery();
        });

        $('btn-scan-new').addEventListener('click', () => {
            navigateTo('scanner', 'Gym Card Scanner');
        });

        // --- Detail View ---
        $('btn-delete-card').addEventListener('click', async () => {
            if (currentCardId !== null && confirm('Delete this card?')) {
                await CardDB.delete(currentCardId);
                showToast('Card deleted');
                navigateTo('gallery', 'Saved Cards');
                renderGallery();
            }
        });

        // --- Back Button ---
        $('btn-back').addEventListener('click', () => {
            if (
                document.querySelector('#view-crop.active') ||
                document.querySelector('#view-save.active')
            ) {
                navigateTo('scanner', 'Gym Card Scanner');
            } else if (document.querySelector('#view-detail.active')) {
                navigateTo('gallery', 'Saved Cards');
                renderGallery();
            }
        });

        // --- Start camera ---
        navigateTo('scanner', 'Gym Card Scanner');
    }

    // Start the app when DOM is ready
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', init);
    } else {
        init();
    }
})();
