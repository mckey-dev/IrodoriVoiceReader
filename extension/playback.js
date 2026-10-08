// ================================================================================
// playback.js
//
// 文単位の連続読み上げを制御する。
// 次文の TTS 先読み、文間の待ち、一時停止、再開、停止、速度・音量変更を行う。
// 文の手動移動は行わない。
// ================================================================================

import {
    state,
    sleep,
    isSessionActive,
    cancelAllRequests,
    createNewSession
} from "./state.js";

import { splitSentences } from "./sentence.js";
import { generateAudio } from "./tts.js";

import {
    playAudio,
    stopCurrentAudio,
    pauseCurrentAudio,
    resumeCurrentAudio,
    updateCurrentAudioSpeed,
    updateCurrentAudioVolume
} from "./audio.js";

import { sendProgress } from "./progress.js";
import { debugError, debugLog } from "./debug.js";

// ================================================================================
// markReadingFailed
// 失敗を state と進捗に残し、再生中のままに見せない。
// ================================================================================
function markReadingFailed(sessionId, current, total, sentence, error) {
    const message = error || "読み上げに失敗しました。";

    if (sessionId === state.readingSession) {
        state.lastError = message;
        state.stopped = true;
        state.paused = false;
    }

    sendProgress(current, total, "error", sentence, {
        error: message
    });
}

// ================================================================================
// waitSentenceGap
// 前の文の再生終了から、次の文を始めるまで待つ。
// 待ちの開始時の設定値を使う。一時停止中はカウントを止める。
// ================================================================================
async function waitSentenceGap(sessionId) {
    const gap = Math.max(0, Math.min(2000, Math.round(Number(state.sentenceGapMs) || 0)));

    if (gap <= 0) {
        return;
    }

    let elapsed = 0;

    while (elapsed < gap) {
        if (!isSessionActive(sessionId)) {
            return;
        }

        while (
            state.paused &&
            !state.stopped &&
            sessionId === state.readingSession
        ) {
            await sleep(100);
        }

        if (!isSessionActive(sessionId)) {
            return;
        }

        const slice = Math.min(50, gap - elapsed);
        await sleep(slice);

        if (!isSessionActive(sessionId)) {
            return;
        }

        if (state.paused) {
            continue;
        }

        elapsed += slice;
    }
}

// ================================================================================
// startReading
// 全文を文単位で読み上げる。再生中に次文の TTS を先読みする。
// ================================================================================
export async function startReading(text) {

    // 古いセッションを停止する
    state.stopped = true;
    state.paused = false;

    // 古いセッションを無効にする
    const sessionId = createNewSession();

    stopCurrentAudio();
    cancelAllRequests();

    // テキストを準備する
    state.currentText = String(text || "");
    state.currentSentences = splitSentences(state.currentText);

    const sentences = state.currentSentences;

    state.currentSentenceTotal = sentences.length;
    state.currentSentenceIndex = 0;
    state.currentSentenceText = "";

    // テキストが空
    if (sentences.length === 0) {
        state.lastError = "読み上げる文章がありません。";
        throw new Error(state.lastError);
    }

    // 新しい再生を開始する
    state.stopped = false;
    state.paused = false;
    state.lastError = "";

    debugLog("playback", "new session", {
        sessionId: sessionId,
        chars: state.currentText.length,
        sentences: sentences.length,
        lengths: sentences.map(item => item.length)
    });

    // 開始時の進捗
    state.currentSentenceIndex = 1;
    state.currentSentenceTotal = sentences.length;
    state.currentSentenceText = sentences[0];

    sendProgress(1, sentences.length, "starting", sentences[0]);

    // 最初の TTS リクエスト。文間の待ちは2文目から。
    let nextAudioPromise = generateAudio(sentences[0], sessionId);
    let sentenceGapPromise = Promise.resolve();
    let failed = false;

    // 再生のメインループ
    for (let i = 0; i < sentences.length; i++) {

        // セッションを確認する
        if (!isSessionActive(sessionId)) {
            debugLog("playback", "stop loop: session inactive", {
                sessionId: sessionId,
                index: i + 1
            });
            break;
        }

        // 一時停止中は待機する
        while (
            state.paused &&
            !state.stopped &&
            sessionId === state.readingSession
        ) {
            await sleep(100);
        }

        if (!isSessionActive(sessionId)) {
            break;
        }

        // 現在の TTS と、前の文との間隔を待つ。
        // 先読みは前の文の再生前に始まっているので、待ちのあいだも変換は続く。
        let blob;

        try {
            const waited = await Promise.all([
                nextAudioPromise,
                sentenceGapPromise
            ]);
            blob = waited[0];
        } catch (error) {
            debugError("playback", "tts failed", {
                sessionId: sessionId,
                index: i + 1,
                chars: sentences[i].length,
                error: error && error.message ? error.message : String(error)
            });

            if (!isSessionActive(sessionId)) {
                break;
            }

            markReadingFailed(
                sessionId,
                i + 1,
                sentences.length,
                sentences[i],
                error && error.message ? error.message : String(error)
            );

            throw error;
        }

        if (!isSessionActive(sessionId)) {
            debugLog("playback", "stop loop: session inactive after tts", {
                sessionId: sessionId,
                index: i + 1
            });
            break;
        }

        if (!blob) {
            debugLog("playback", "stop loop: empty audio", {
                sessionId: sessionId,
                index: i + 1,
                chars: sentences[i].length
            });
            markReadingFailed(
                sessionId,
                i + 1,
                sentences.length,
                sentences[i],
                "音声を取得できませんでした。"
            );
            failed = true;
            break;
        }

        // 次の文を先読みする
        //
        // 現在の文を再生している間に、
        // 次の文のTTSを生成する。
        if (i + 1 < sentences.length) {
            nextAudioPromise = generateAudio(sentences[i + 1], sessionId);
            debugLog("playback", "prefetch", {
                sessionId: sessionId,
                index: i + 2,
                total: sentences.length
            });
        }

        // 現在の文を更新する
        //
        // i は 0 始まり
        // currentSentenceIndex は 1 始まり
        state.currentSentenceIndex = i + 1;
        state.currentSentenceTotal = sentences.length;
        state.currentSentenceText = sentences[i];

        debugLog("playback", "playing", {
            sessionId: sessionId,
            index: state.currentSentenceIndex,
            total: state.currentSentenceTotal,
            chars: sentences[i].length,
            bytes: blob.size
        });

        sendProgress(
            state.currentSentenceIndex,
            state.currentSentenceTotal,
            "playing",
            state.currentSentenceText
        );

        // 現在の音声を再生する
        try {
            await playAudio(blob, sessionId);
        } catch (error) {
            debugError("playback", "play failed", {
                sessionId: sessionId,
                index: i + 1,
                error: error && error.message ? error.message : String(error)
            });

            if (!isSessionActive(sessionId)) {
                break;
            }

            markReadingFailed(
                sessionId,
                i + 1,
                sentences.length,
                sentences[i],
                error && error.message ? error.message : String(error)
            );

            throw error;
        }

        if (i + 1 < sentences.length && isSessionActive(sessionId)) {
            sentenceGapPromise = waitSentenceGap(sessionId);
        }
    }

    // 失敗時は完了扱いにしない
    if (failed) {
        return;
    }

    // 完了
    if (!state.stopped && sessionId === state.readingSession) {
        state.currentSentenceIndex = sentences.length;
        state.currentSentenceTotal = sentences.length;
        state.currentSentenceText = "";

        debugLog("playback", "finished", {
            sessionId: sessionId,
            total: sentences.length
        });

        sendProgress(sentences.length, sentences.length, "finished", "");
    }
}

// ================================================================================
// pauseReading
// 読み上げを一時停止する。
// ================================================================================
export function pauseReading() {
    if (state.stopped) {
        return;
    }

    state.paused = true;
    pauseCurrentAudio();

    sendProgress(
        state.currentSentenceIndex,
        state.currentSentenceTotal,
        "paused",
        state.currentSentenceText
    );

    debugLog("playback", "paused", {
        index: state.currentSentenceIndex,
        total: state.currentSentenceTotal
    });
}

// ================================================================================
// resumeReading
// 一時停止中の読み上げを再開する。
// ================================================================================
export async function resumeReading() {
    if (state.stopped) {
        debugLog("playback", "resume ignored: stopped");
        return;
    }

    state.paused = false;
    await resumeCurrentAudio();

    sendProgress(
        state.currentSentenceIndex,
        state.currentSentenceTotal,
        "playing",
        state.currentSentenceText
    );

    debugLog("playback", "resumed", {
        index: state.currentSentenceIndex,
        total: state.currentSentenceTotal
    });
}

// ================================================================================
// stopReading
// 読み上げを停止し、進捗とセッションをリセットする。
// ================================================================================
export function stopReading() {
    debugLog("playback", "stopping");

    // 先に現在のセッションを無効にする
    state.stopped = true;
    state.paused = false;
    state.lastError = "";
    createNewSession();

    // 音声と TTS を停止する
    stopCurrentAudio();
    cancelAllRequests();

    // 進捗をリセットする
    state.currentSentenceIndex = 0;
    state.currentSentenceTotal = 0;
    state.currentSentenceText = "";

    sendProgress(0, 0, "stopped", "");
}

// ================================================================================
// setSpeed
// 再生速度を 0.5〜2.0 の範囲で設定する。
// ================================================================================
export function setSpeed(value) {
    const speed = Number(value);

    if (!Number.isFinite(speed)) {
        return;
    }

    state.playbackSpeed = Math.max(0.5, Math.min(2.0, speed));
    updateCurrentAudioSpeed();

    debugLog("playback", "speed", {
        value: state.playbackSpeed
    });
}

// ================================================================================
// getGenerationOptions
// 現在の生成オプションを、サーバーへ送る形で返す。
// ================================================================================
export function getGenerationOptions() {
    return {
        caption: state.caption,
        num_steps: state.numSteps,
        speed: state.speed,
        cfg_scale_text: state.cfgScaleText,
        cfg_scale_caption: state.cfgScaleCaption,
        cfg_scale_speaker: state.cfgScaleSpeaker,
        t_schedule_mode: state.tScheduleMode,
        sway_coeff: state.swayCoeff,
        watermark: state.watermark
    };
}

// ================================================================================
// applyGenerationOptions
// 生成オプションを state へ反映する。保存はしない。
// ================================================================================
function applyGenerationOptions(options = {}) {
    if (typeof options.caption === "string") {
        state.caption = options.caption;
    }

    const numSteps = Number(options.num_steps);

    if (Number.isFinite(numSteps) && numSteps >= 1) {
        state.numSteps = Math.round(numSteps);
    }

    const speed = Number(options.speed);

    if (Number.isFinite(speed) && speed > 0) {
        state.speed = speed;
    }

    const cfgText = Number(options.cfg_scale_text);

    if (Number.isFinite(cfgText) && cfgText >= 0) {
        state.cfgScaleText = cfgText;
    }

    const cfgCaption = Number(options.cfg_scale_caption);

    if (Number.isFinite(cfgCaption) && cfgCaption >= 0) {
        state.cfgScaleCaption = cfgCaption;
    }

    const cfgSpeaker = Number(options.cfg_scale_speaker);

    if (Number.isFinite(cfgSpeaker) && cfgSpeaker >= 0) {
        state.cfgScaleSpeaker = cfgSpeaker;
    }

    if (typeof options.t_schedule_mode === "string" && options.t_schedule_mode.trim()) {
        state.tScheduleMode = options.t_schedule_mode.trim().toLowerCase() === "sway"
            ? "sway"
            : "linear";
    }

    const swayCoeff = Number(options.sway_coeff);

    if (Number.isFinite(swayCoeff)) {
        state.swayCoeff = swayCoeff;
    }

    if (typeof options.watermark === "boolean") {
        state.watermark = options.watermark;
    }
}

// ================================================================================
// setGenerationOptions
// 生成オプションを更新する。保存は settings.json へ background が行う。
// ================================================================================
export function setGenerationOptions(options = {}) {
    applyGenerationOptions(options);

    const saved = getGenerationOptions();
    debugLog("playback", "generation options", saved);
    return saved;
}

// ================================================================================
// setVolume
// 再生音量を 0.0〜1.0 の範囲で設定する。
// ================================================================================
export function setVolume(value) {
    const volume = Number(value);

    if (!Number.isFinite(volume)) {
        return;
    }

    state.playbackVolume = Math.max(0, Math.min(1, volume));
    updateCurrentAudioVolume();

    debugLog("playback", "volume", {
        value: state.playbackVolume
    });
}

// ================================================================================
// setSentenceGap
// 次の文を読み始めるまでの待ち時間を、0〜2000ms で設定する。
// ================================================================================
export function setSentenceGap(value) {
    const gap = Number(value);

    if (!Number.isFinite(gap)) {
        return;
    }

    state.sentenceGapMs = Math.max(0, Math.min(2000, Math.round(gap)));

    debugLog("playback", "sentence gap", {
        value: state.sentenceGapMs
    });
}

// ================================================================================
