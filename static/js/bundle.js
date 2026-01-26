async function initLive2D() {
    const canvas = document.getElementById('my-canvas');
    if (!canvas) return;

    // WAJIB: Gunakan PIXI (Bukan PTXT)
    const app = new PIXI.Application({
        view: canvas,
        autoStart: true,
        backgroundAlpha: 0,
        resizeTo: window
    });

    try {
        // Sesuaikan path ke folder runtime Anda
        const modelUrl = '/static/mao_pro_en/runtime/mao_pro.model3.json';
        
        // Memuat model menggunakan plugin index.min.js
        const model = await PIXI.live2d.Live2DModel.from(modelUrl);

        app.stage.addChild(model);

        // Atur posisi dan ukuran karakter
        model.scale.set(0.12); 
        model.x = window.innerWidth / 2;
        model.y = window.innerHeight / 2;
        model.anchor.set(0.5, 0.5);

        console.log("Mao Pro Berhasil Dimuat!");
    } catch (e) {
        console.error("Gagal memuat model:", e);
    }
}

// Menjalankan fungsi setelah semua library terload
window.addEventListener('load', initLive2D);