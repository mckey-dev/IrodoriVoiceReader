// ================================================================================
// popup.js
//
// 操作ウィンドウの UI。抽出、読み上げ開始、再生制御、進捗表示を行う。
// ================================================================================

const status = document.getElementById("status");
const result = document.getElementById("result");
const progress = document.getElementById("progress");
const currentSentence = document.getElementById("currentSentence");
const extractButton = document.getElementById("extractButton");
const readButton = document.getElementById("readButton");
const readSelectionButton = document.getElementById("readSelectionButton");
const pauseButton = document.getElementById("pauseButton");
const resumeButton = document.getElementById("resumeButton");
const stopButton = document.getElementById("stopButton");
const speedRange = document.getElementById("speedRange");
const speedValue = document.getElementById("speedValue");
const volumeRange = document.getElementById("volumeRange");
const volumeValue = document.getElementById("volumeValue");
const sentenceGapRange = document.getElementById("sentenceGapRange");
const sentenceGapValue = document.getElementById("sentenceGapValue");
const highlightSentenceInput = document.getElementById("highlightSentenceInput");
const autoReadChatInput = document.getElementById("autoReadChatInput");
const captionInput = document.getElementById("captionInput");
const numStepsInput = document.getElementById("numStepsInput");
const speedInput = document.getElementById("speedInput");
const cfgScaleTextInput = document.getElementById("cfgScaleTextInput");
const cfgScaleCaptionInput = document.getElementById("cfgScaleCaptionInput");
const cfgScaleSpeakerInput = document.getElementById("cfgScaleSpeakerInput");
const tScheduleModeInput = document.getElementById("tScheduleModeInput");
const swayCoeffInput = document.getElementById("swayCoeffInput");
const watermarkInput = document.getElementById("watermarkInput");
const modelSelect = document.getElementById("modelSelect");
const modelNote = document.getElementById("modelNote");
const MODEL_SERVER_URL = "http://127.0.0.1:8000";
let modelBusy = false;
let loadedModelId = "";

// ================================================================================
// getCurrentTab
// 読み上げ対象のタブを取得する。操作ウィンドウではなく、最後に使った通常窓を使う。
// ================================================================================
async function getCurrentTab() {
    let windowId;

    try {
        const last = await chrome.windows.getLastFocused({
            windowTypes: ["normal"]
        });
        windowId = last && last.id;
    } catch (error) {
        windowId = undefined;
    }

    const query = {
        active: true
    };

    if (windowId != null) {
        query.windowId = windowId;
    } else {
        query.currentWindow = true;
    }

    const [tab] = await chrome.tabs.query(query);

    if (!tab || !tab.id) {
        throw new Error("現在のタブを取得できませんでした。");
    }

    return tab;
}

// ================================================================================
// sendCommand
// background へ操作を送り、失敗時は例外を投げる。
// ================================================================================
async function sendCommand(action, extra = {}) {
    const response = await chrome.runtime.sendMessage({
        action,
        ...extra
    });

    if (!response || response.success === false) {
        throw new Error(response?.error || "操作に失敗しました。");
    }

    return response;
}

// ================================================================================
// applyPlaybackState
// 取得した再生状態を、ポップアップ表示へ反映する。
// ================================================================================
function applyPlaybackState(playbackState) {
    if (!playbackState) {
        return;
    }

    // 全文を復元する
    if (typeof playbackState.currentText === "string") {
        result.textContent = playbackState.currentText;
    }

    // 進捗
    const current = Number(playbackState.currentSentenceIndex);
    const total = Number(playbackState.currentSentenceTotal);

    if (Number.isFinite(current) && Number.isFinite(total)) {
        if (total > 0) {
            progress.textContent = `${current} / ${total} 文`;
        } else {
            progress.textContent = "0 / 0 文";
        }
    }

    // 現在の文
    if (typeof playbackState.currentSentenceText === "string") {
        if (playbackState.currentSentenceText) {
            currentSentence.textContent = playbackState.currentSentenceText;
        } else if (playbackState.stopped) {
            currentSentence.textContent = "現在の文章はありません";
        }
    }

    // 速度
    const speed = Number(playbackState.playbackSpeed);

    if (Number.isFinite(speed)) {
        speedRange.value = String(speed);
        speedValue.textContent = speed.toFixed(2) + "x";
    }

    // 音量
    const volume = Number(playbackState.playbackVolume);

    if (Number.isFinite(volume)) {
        const volumePercent = Math.round(volume * 100);
        volumeRange.value = String(volumePercent);
        volumeValue.textContent = volumePercent + "%";
    }

    const sentenceGap = Number(playbackState.sentenceGapMs);

    if (Number.isFinite(sentenceGap)) {
        sentenceGapRange.value = String(sentenceGap);
        sentenceGapValue.textContent = sentenceGap + "ms";
    }

    applyGenerationOptions(playbackState.generationOptions);

    // 再生状態
    if (playbackState.lastError || playbackState.playbackState === "error") {
        status.textContent = "読み上げエラー:\n" + (
            playbackState.lastError || "読み上げに失敗しました。"
        );
        return;
    }

    if (playbackState.stopped) {
        if (total > 0 && current >= total) {
            status.textContent = "読み上げ完了";
        } else {
            status.textContent = "停止しました";
        }

        return;
    }

    if (playbackState.paused) {
        status.textContent = "一時停止中";
        return;
    }

    if (playbackState.playbackState === "finished") {
        status.textContent = "読み上げ完了";
        return;
    }

    if (playbackState.playing) {
        status.textContent = "読み上げ中";
        return;
    }

    // 再生していない
    if (total === 0) {
        status.textContent = "待機中";
    }
}

// ================================================================================
// applyGenerationOptions
// 生成オプションをポップアップ入力へ反映する。
// ================================================================================
function applyGenerationOptions(options) {
    if (!options) {
        return;
    }

    if (typeof options.caption === "string") {
        captionInput.value = options.caption;
    }

    if (Number.isFinite(Number(options.num_steps))) {
        numStepsInput.value = String(options.num_steps);
    }

    if (Number.isFinite(Number(options.speed))) {
        speedInput.value = String(options.speed);
    }

    if (Number.isFinite(Number(options.cfg_scale_text))) {
        cfgScaleTextInput.value = String(options.cfg_scale_text);
    }

    if (Number.isFinite(Number(options.cfg_scale_caption))) {
        cfgScaleCaptionInput.value = String(options.cfg_scale_caption);
    }

    if (Number.isFinite(Number(options.cfg_scale_speaker))) {
        cfgScaleSpeakerInput.value = String(options.cfg_scale_speaker);
    }

    if (typeof options.t_schedule_mode === "string" && options.t_schedule_mode) {
        tScheduleModeInput.value = options.t_schedule_mode === "sway" ? "sway" : "linear";
    }

    if (Number.isFinite(Number(options.sway_coeff))) {
        swayCoeffInput.value = String(options.sway_coeff);
    }

    watermarkInput.checked = Boolean(options.watermark);
}

// ================================================================================
// collectGenerationOptions
// ポップアップ入力から生成オプションを集める。
// ================================================================================
function collectGenerationOptions() {
    return {
        caption: captionInput.value,
        speed: Number(speedInput.value),
        num_steps: Number(numStepsInput.value),
        cfg_scale_text: Number(cfgScaleTextInput.value),
        cfg_scale_caption: Number(cfgScaleCaptionInput.value),
        cfg_scale_speaker: Number(cfgScaleSpeakerInput.value),
        t_schedule_mode: tScheduleModeInput.value,
        sway_coeff: Number(swayCoeffInput.value),
        watermark: watermarkInput.checked
    };
}

// ================================================================================
// applySettingsToForm
// settings.json の内容をポップアップ入力へ反映する。
// ================================================================================
function applySettingsToForm(settings) {
    if (!settings) {
        return;
    }

    applyGenerationOptions(settings);

    const speed = Number(settings.playbackSpeed);

    if (Number.isFinite(speed)) {
        speedRange.value = String(speed);
        speedValue.textContent = speed.toFixed(2) + "x";
    }

    const volume = Number(settings.playbackVolume);

    if (Number.isFinite(volume)) {
        const volumePercent = Math.round(volume * 100);
        volumeRange.value = String(volumePercent);
        volumeValue.textContent = volumePercent + "%";
    }

    const sentenceGap = Number(settings.sentenceGapMs);

    if (Number.isFinite(sentenceGap)) {
        sentenceGapRange.value = String(sentenceGap);
        sentenceGapValue.textContent = sentenceGap + "ms";
    }

    highlightSentenceInput.checked = Boolean(settings.highlightSentence);
    autoReadChatInput.checked = Boolean(settings.autoReadChat);
}

// ================================================================================
// sendGenerationOptions
// 生成オプションを settings.json と offscreen へ送る。
// ================================================================================
async function sendGenerationOptions() {
    const value = collectGenerationOptions();

    try {
        await sendCommand("changeGenerationOptions", {
            value: value
        });
    } catch (error) {
        console.error("Generation options error:", error);
        status.textContent = "生成設定エラー:\n" + error.message;
    }
}

// ================================================================================
// restorePlaybackState
// ポップアップを開いたとき、進行中の読み上げ状態を復元する。
// ================================================================================
async function restorePlaybackState() {
    try {
        const settingsResponse = await sendCommand("reloadSettings");
        applySettingsToForm(settingsResponse && settingsResponse.settings);
    } catch (error) {
        console.warn("Failed to restore settings.json:", error);
    }

    try {
        const response = await sendCommand("getPlaybackState");

        if (response && response.state) {
            applyPlaybackState(response.state);
        }
    } catch (error) {
        console.log(
            "Playback state restore is not available yet:",
            error
        );
    }

    await refreshVoices();
    await refreshModels();
}

// ================================================================================
// extractArticle
// 現在のタブから記事本文を抽出する。
// ================================================================================
async function extractArticle() {
    const tab = await getCurrentTab();

    await chrome.scripting.executeScript({
        target: { tabId: tab.id },
        files: ["extract_text.js", "content.js"]
    });

    const response = await chrome.tabs.sendMessage(tab.id, {
        action: "extractText"
    });

    if (!response || !response.success) {
        throw new Error("本文を抽出できませんでした。");
    }

    return response.text;
}

// ================================================================================
// extractButton クリック
// 本文抽出ボタンの処理。
// ================================================================================
extractButton.addEventListener("click", async () => {
    status.textContent = "本文を抽出しています……";
    result.textContent = "";

    try {
        const text = await extractArticle();
        result.textContent = text;
        status.textContent = `本文を抽出しました（${text.length}文字）`;
    } catch (error) {
        console.error(error);
        status.textContent = "抽出エラー:\n" + error.message;
    }
});

// ================================================================================
// readButton クリック
// ページ全体の読み上げを開始する。
// ================================================================================
readButton.addEventListener("click", async () => {
    status.textContent = "読み上げを開始しています……";

    try {
        const tab = await getCurrentTab();
        const response = await chrome.runtime.sendMessage({
            action: "startPageReading",
            tabId: tab.id
        });

        if (!response || !response.success) {
            throw new Error(
                response?.error || "読み上げを開始できませんでした。"
            );
        }

        status.textContent = "読み上げ中";

        if (typeof response.text === "string") {
            result.textContent = response.text;
        }
    } catch (error) {
        console.error(error);
        status.textContent = "読み上げエラー:\n" + error.message;
    }
});

// ================================================================================
// readSelectionButton クリック
// 選択テキストだけの読み上げを開始する。
// ================================================================================
readSelectionButton.addEventListener("click", async () => {
    status.textContent = "選択テキストを読み上げています……";

    try {
        const tab = await getCurrentTab();
        const response = await chrome.runtime.sendMessage({
            action: "startSelectionReading",
            tabId: tab.id
        });

        if (!response || !response.success) {
            throw new Error(
                response?.error || "選択テキストを読み上げできませんでした。"
            );
        }

        status.textContent = "読み上げ中";

        if (typeof response.text === "string") {
            result.textContent = response.text;
        }
    } catch (error) {
        console.error(error);
        status.textContent = "読み上げエラー:\n" + error.message;
    }
});

// ================================================================================
// pauseButton クリック
// 読み上げを一時停止する。
// ================================================================================
pauseButton.addEventListener("click", async () => {
    try {
        await sendCommand("pausePlayback");
        status.textContent = "一時停止中";
    } catch (error) {
        console.error(error);
        status.textContent = "一時停止エラー:\n" + error.message;
    }
});

// ================================================================================
// resumeButton クリック
// 一時停止中の読み上げを再開する。
// ================================================================================
resumeButton.addEventListener("click", async () => {
    try {
        await sendCommand("resumePlayback");
        status.textContent = "読み上げ中";
    } catch (error) {
        console.error(error);
        status.textContent = "再開エラー:\n" + error.message;
    }
});

// ================================================================================
// stopButton クリック
// 読み上げを停止し、表示を初期状態へ戻す。
// ================================================================================
stopButton.addEventListener("click", async () => {
    try {
        await sendCommand("stopPlayback");
        status.textContent = "停止しました";
        progress.textContent = "0 / 0 文";
        currentSentence.textContent = "現在の文章はありません";
        result.textContent = "";
    } catch (error) {
        console.error(error);
        status.textContent = "停止エラー:\n" + error.message;
    }
});

// ================================================================================
// speedRange 入力
// 再生速度を変更する。
// ================================================================================
speedRange.addEventListener("input", async () => {
    const value = Number(speedRange.value);
    speedValue.textContent = value.toFixed(2) + "x";

    try {
        await sendCommand("changeSpeed", { value: value });
    } catch (error) {
        console.error("Speed error:", error);
        status.textContent = "速度変更エラー:\n" + error.message;
    }
});

// ================================================================================
// volumeRange 入力
// 再生音量を変更する。
// ================================================================================
volumeRange.addEventListener("input", async () => {
    const value = Number(volumeRange.value);
    volumeValue.textContent = value + "%";

    try {
        await sendCommand("changeVolume", { value: value / 100 });
    } catch (error) {
        console.error("Volume error:", error);
        status.textContent = "音量変更エラー:\n" + error.message;
    }
});

// ================================================================================
// sentenceGapRange 入力
// 次の文を読み始めるまでの待ち時間を変更する。
// ================================================================================
sentenceGapRange.addEventListener("input", async () => {
    const value = Number(sentenceGapRange.value);
    sentenceGapValue.textContent = value + "ms";

    try {
        await sendCommand("changeSentenceGap", { value: value });
    } catch (error) {
        console.error("Sentence gap error:", error);
        status.textContent = "待ち時間変更エラー:\n" + error.message;
    }
});

// ================================================================================
// ハイライト / チャット自動読み上げ
// ================================================================================
async function sendReaderFlags() {
    try {
        await sendCommand("changeReaderFlags", {
            highlightSentence: highlightSentenceInput.checked,
            autoReadChat: autoReadChatInput.checked
        });
    } catch (error) {
        console.error("Reader flags error:", error);
        status.textContent = "設定エラー:\n" + error.message;
    }
}

highlightSentenceInput.addEventListener("change", () => {
    sendReaderFlags();
});

autoReadChatInput.addEventListener("change", () => {
    sendReaderFlags();
});

// ================================================================================
// 生成設定の入力
// 変更した生成オプションを offscreen へ送る。
// ================================================================================
[
    captionInput,
    speedInput,
    numStepsInput,
    cfgScaleTextInput,
    cfgScaleCaptionInput,
    cfgScaleSpeakerInput,
    tScheduleModeInput,
    swayCoeffInput,
    watermarkInput
].forEach(input => {
    input.addEventListener("change", () => {
        sendGenerationOptions();
    });
});

// ================================================================================
// onMessage
// Offscreen からの読み上げ進捗を操作ウィンドウへ反映する。
// ================================================================================
chrome.runtime.onMessage.addListener((message) => {
    if (message.action !== "readingProgress") {
        return;
    }

    const current = Number(message.current);
    const total = Number(message.total);

    if (!Number.isFinite(current) || !Number.isFinite(total)) {
        return;
    }

    progress.textContent = `${current} / ${total} 文`;

    if (typeof message.sentence === "string") {
        currentSentence.textContent = message.sentence;
    }

    if (message.state === "error") {
        status.textContent = "読み上げエラー:\n" + (
            message.error || "読み上げに失敗しました。"
        );
    } else if (message.state === "finished") {
        status.textContent = "読み上げ完了";
    } else if (message.state === "paused") {
        status.textContent = "一時停止中";
    } else if (message.state === "playing") {
        status.textContent = "読み上げ中";
    } else if (message.state === "starting") {
        status.textContent = "読み上げを開始しています……";
    } else if (message.state === "stopped") {
        status.textContent = "停止しました";
        progress.textContent = "0 / 0 文";
        currentSentence.textContent = "現在の文章はありません";
        result.textContent = "";
    }
});

// ================================================================================
// modelErrorMessage
// モデル API の失敗を、表示用の文にする。
// ================================================================================
function modelErrorMessage(error, fallback) {
    const message = error && error.message ? error.message : "";

    if (/fetch/i.test(message)) {
        return "サーバーに接続できません";
    }

    if (message) {
        return message;
    }

    return fallback;
}

// ================================================================================
// showModelNote
// 選択中のモデルの説明を出す。
// ================================================================================
function showModelNote() {
    const option = modelSelect.options[modelSelect.selectedIndex];
    modelNote.textContent = option && option.dataset.note
        ? option.dataset.note
        : "";
}

// ================================================================================
// refreshModels
// サーバーのモデル一覧と、いま読み込んでいるモデルを反映する。
// ================================================================================
async function refreshModels() {
    if (modelBusy) {
        return;
    }

    try {
        const response = await fetch(MODEL_SERVER_URL + "/models");
        const data = await response.json();

        if (!response.ok) {
            throw new Error(
                (data && data.detail) || "モデル一覧を取得できませんでした。"
            );
        }

        const models = Array.isArray(data.models) ? data.models : [];
        modelSelect.replaceChildren();

        models.forEach((model) => {
            const option = document.createElement("option");
            option.value = model.id;
            option.textContent = model.label || model.name || model.id;
            option.dataset.note = model.note || "";
            modelSelect.appendChild(option);
        });

        loadedModelId = data.active || "";

        if (loadedModelId) {
            modelSelect.value = loadedModelId;
        }

        showModelNote();
        modelSelect.disabled = models.length === 0;
    } catch (error) {
        console.warn("Failed to load models:", error);
        modelNote.textContent = modelErrorMessage(
            error,
            "モデル一覧を取得できませんでした。"
        );
    }
}

// ================================================================================
// modelSelect 変更
// 選んだチェックポイントを読み込む。未取得ならサーバーがダウンロードする。
// ================================================================================
modelSelect.addEventListener("change", async () => {
    const modelId = modelSelect.value;
    showModelNote();

    if (!modelId || modelId === loadedModelId || modelBusy) {
        return;
    }

    const previousId = loadedModelId;
    modelBusy = true;
    modelSelect.disabled = true;
    status.textContent = "モデルを読み込んでいます……\n初回はダウンロードに時間がかかります。";

    try {
        const response = await fetch(
            MODEL_SERVER_URL + "/models/" + encodeURIComponent(modelId),
            { method: "PUT" }
        );
        let data = null;

        try {
            data = await response.json();
        } catch (error) {
            data = null;
        }

        if (!response.ok) {
            const detail = data && data.detail;
            throw new Error(
                typeof detail === "string" && detail
                    ? detail
                    : "モデルの切り替えに失敗しました。"
            );
        }

        loadedModelId = (data && data.active) || modelId;
        modelSelect.value = loadedModelId;
        showModelNote();

        try {
            await sendCommand("reloadSettings");
        } catch (error) {
            console.warn("Failed to reload settings after model switch:", error);
        }

        const label = modelSelect.options[modelSelect.selectedIndex];
        status.textContent = "モデルを切り替えました"
            + (label ? "（" + label.textContent + "）" : "");
    } catch (error) {
        console.error(error);
        modelSelect.value = previousId;
        showModelNote();
        status.textContent = "モデル切り替えエラー:\n" + modelErrorMessage(
            error,
            "モデルの切り替えに失敗しました。"
        );
    } finally {
        modelBusy = false;
        modelSelect.disabled = false;
    }
});

// ================================================================================
// 初期化
// ポップアップ表示時に、現在の再生状態を復元する。
// ================================================================================
restorePlaybackState();

// ================================================================================
