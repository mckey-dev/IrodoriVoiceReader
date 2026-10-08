// ================================================================================
// extension_settings.js
//
// settings.json の読み書きと、ページへのフラグ配信。
// ================================================================================

const SERVER_URL = "http://127.0.0.1:8000";

let cachedSettings = null;

// ================================================================================
// getCachedSettings
// メモリ上の設定を返す。未読み込みなら null。
// ================================================================================
export function getCachedSettings() {
    return cachedSettings;
}

// ================================================================================
// setCachedSettings
// メモリ上の設定を差し替える。
// ================================================================================
export function setCachedSettings(settings) {
    cachedSettings = settings || null;
    return cachedSettings;
}

// ================================================================================
// defaultExtensionSettings
// 拡張設定の既定値を返す。
// ================================================================================
export function defaultExtensionSettings() {
    return {
        model: "v4.1-anime",
        playbackSpeed: 1.0,
        playbackVolume: 1.0,
        sentenceGapMs: 500,
        caption: "",
        num_steps: 40,
        speed: 1.0,
        cfg_scale_text: 3.0,
        cfg_scale_caption: 3.0,
        cfg_scale_speaker: 5.0,
        t_schedule_mode: "linear",
        sway_coeff: -1.0,
        watermark: false,
        highlightSentence: false,
        autoReadChat: false
    };
}

// ================================================================================
// normalizeScheduleMode
// linear / sway 以外は linear に戻す。
// ================================================================================
function normalizeScheduleMode(value) {
    return String(value || "").trim().toLowerCase() === "sway"
        ? "sway"
        : "linear";
}

// ================================================================================
// normalizeExtensionSettings
// 設定値を既定値と合成し、範囲を整える。
// ================================================================================
export function normalizeExtensionSettings(value) {
    const settings = defaultExtensionSettings();

    if (!value || typeof value !== "object") {
        return settings;
    }

    if (typeof value.model === "string" && value.model.trim()) {
        settings.model = value.model.trim();
    }

    if (typeof value.caption === "string") {
        settings.caption = value.caption;
    }

    const numSteps = Number(value.num_steps);

    if (Number.isFinite(numSteps) && numSteps >= 1) {
        settings.num_steps = Math.round(numSteps);
    }

    const speed = Number(value.speed);

    if (Number.isFinite(speed) && speed > 0) {
        settings.speed = speed;
    }

    const cfgText = Number(value.cfg_scale_text);

    if (Number.isFinite(cfgText) && cfgText >= 0) {
        settings.cfg_scale_text = cfgText;
    }

    const cfgCaption = Number(value.cfg_scale_caption);

    if (Number.isFinite(cfgCaption) && cfgCaption >= 0) {
        settings.cfg_scale_caption = cfgCaption;
    }

    const cfgSpeaker = Number(value.cfg_scale_speaker);

    if (Number.isFinite(cfgSpeaker) && cfgSpeaker >= 0) {
        settings.cfg_scale_speaker = cfgSpeaker;
    }

    settings.t_schedule_mode = normalizeScheduleMode(value.t_schedule_mode);

    const swayCoeff = Number(value.sway_coeff);

    if (Number.isFinite(swayCoeff)) {
        settings.sway_coeff = swayCoeff;
    }

    if (typeof value.watermark === "boolean") {
        settings.watermark = value.watermark;
    }

    if (typeof value.highlightSentence === "boolean") {
        settings.highlightSentence = value.highlightSentence;
    }

    if (typeof value.autoReadChat === "boolean") {
        settings.autoReadChat = value.autoReadChat;
    }

    const playbackSpeed = Number(value.playbackSpeed);

    if (Number.isFinite(playbackSpeed)) {
        settings.playbackSpeed = Math.max(0.5, Math.min(2.0, playbackSpeed));
    }

    const playbackVolume = Number(value.playbackVolume);

    if (Number.isFinite(playbackVolume)) {
        settings.playbackVolume = Math.max(0, Math.min(1, playbackVolume));
    }

    const sentenceGapMs = Number(value.sentenceGapMs);

    if (Number.isFinite(sentenceGapMs)) {
        settings.sentenceGapMs = Math.max(0, Math.min(2000, Math.round(sentenceGapMs)));
    }

    return settings;
}

// ================================================================================
// loadExtensionSettings
// サーバーの settings.json を読む。失敗時は拡張内のファイルを使う。
// ================================================================================
export async function loadExtensionSettings() {
    try {
        const response = await fetch(SERVER_URL + "/extension/settings");

        if (response.ok) {
            const data = await response.json();
            return normalizeExtensionSettings(data && data.settings);
        }
    } catch (error) {
        console.warn("Failed to load settings from server:", error);
    }

    try {
        const response = await fetch(chrome.runtime.getURL("settings.json"));

        if (response.ok) {
            return normalizeExtensionSettings(await response.json());
        }
    } catch (error) {
        console.warn("Failed to load bundled settings.json:", error);
    }

    return defaultExtensionSettings();
}

// ================================================================================
// writeExtensionSettings
// サーバー経由で settings.json に保存する。
// ================================================================================
export async function writeExtensionSettings(settings) {
    const response = await fetch(SERVER_URL + "/extension/settings", {
        method: "PUT",
        headers: {
            "Content-Type": "application/json"
        },
        body: JSON.stringify(settings)
    });

    if (!response.ok) {
        throw new Error("設定ファイルの保存に失敗しました。");
    }

    const data = await response.json();
    return normalizeExtensionSettings(data && data.settings);
}

// ================================================================================
// persistExtensionSettings
// 設定の一部を更新し、settings.json へ保存する。
// ================================================================================
export async function persistExtensionSettings(patch) {
    const current = cachedSettings || await loadExtensionSettings();

    cachedSettings = normalizeExtensionSettings({
        ...current,
        ...patch
    });

    try {
        cachedSettings = await writeExtensionSettings(cachedSettings);
    } catch (error) {
        console.warn("Failed to save settings.json:", error);
    }

    return cachedSettings;
}

// ================================================================================
// getReaderFlags
// ページ側へ渡すハイライト／チャット自動のフラグを返す。
// ================================================================================
export async function getReaderFlags() {
    const settings = cachedSettings || await loadExtensionSettings();
    cachedSettings = settings;

    return {
        highlightSentence: Boolean(settings.highlightSentence),
        autoReadChat: Boolean(settings.autoReadChat)
    };
}

// ================================================================================
// broadcastReaderFlags
// 開いているページへ、ハイライトとチャット自動の設定を伝える。
// ================================================================================
export async function broadcastReaderFlags(settings) {
    const highlightSentence = Boolean(settings && settings.highlightSentence);
    const autoReadChat = Boolean(settings && settings.autoReadChat);
    let tabs = [];

    try {
        tabs = await chrome.tabs.query({});
    } catch (error) {
        return;
    }

    tabs.forEach((tab) => {
        if (!tab.id) {
            return;
        }

        chrome.tabs.sendMessage(tab.id, {
            action: "setHighlightSentence",
            value: highlightSentence
        }).catch(() => {});

        chrome.tabs.sendMessage(tab.id, {
            action: "setAutoReadChat",
            value: autoReadChat
        }).catch(() => {});
    });
}

// ================================================================================
