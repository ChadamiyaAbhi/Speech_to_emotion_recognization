document.addEventListener("DOMContentLoaded", () => {
    let recordBtn = document.getElementById("recordBtn");
    let stopBtn = document.getElementById("stopBtn");
    let uploadBtn = document.getElementById("uploadBtn");
    let audioFileInput = document.getElementById("audioFile");
    let resultText = document.getElementById("result");
    let mediaRecorder;
    let audioChunks = [];

    // Recording Audio
    recordBtn.addEventListener("click", async () => {
        let stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        mediaRecorder = new MediaRecorder(stream);
        mediaRecorder.start();
        recordBtn.disabled = true;
        stopBtn.disabled = false;
        audioChunks = [];

        mediaRecorder.ondataavailable = event => {
            audioChunks.push(event.data);
        };
    });

    stopBtn.addEventListener("click", () => {
        mediaRecorder.stop();
        stopBtn.disabled = true;
        recordBtn.disabled = false;

        mediaRecorder.onstop = async () => {
            let audioBlob = new Blob(audioChunks, { type: "audio/wav" });
            let formData = new FormData();
            formData.append("file", audioBlob, "recorded_audio.wav");

            let response = await fetch("/upload", {
                method: "POST",
                body: formData
            });

            let data = await response.json();
            resultText.innerText = data.emotion ? `Emotion: ${data.emotion}` : "Failed to process audio";
        };
    });

    // Uploading Audio File
    uploadBtn.addEventListener("click", async () => {
        let file = audioFileInput.files[0];

        if (!file) {
            resultText.innerText = "No file selected!";
            return;
        }

        let formData = new FormData();
        formData.append("file", file);

        let response = await fetch("/upload", {
            method: "POST",
            body: formData
        });

        let data = await response.json();
        resultText.innerText = data.emotion ? `Emotion: ${data.emotion}` : "Failed to process audio";
    });
});
