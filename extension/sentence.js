// ================================================================================
// sentence.js
//
// 読み上げ対象テキストを、TTS 向けの単位に分割する。
//
// 1. 改行で行に分ける。空行は捨て、行をまたいで結合しない
// 2. 100文字未満の行は、中に 。 があってもそのまま1単位
// 3. 100文字以上の行は 。 、 . , の直後で区切る（区切り文字は直前の単位に残す）
// 4. . の直後が数字、, の前後が数字のときは切らない。！ ？ では切らない
// 5. 区切ったあとも100文字を超える塊は、先頭100文字に "…以下略" を付け、続きは捨てる
// 6. 各読み上げ単位の末尾に "。" が無ければ付与する（末尾が . のときは付与しない）
// ================================================================================

import { debugLog } from "./debug.js";

export const MAX_SENTENCE_LENGTH = 100;
const OMISSION_MARK = "…以下略";

// ================================================================================
// normalizeSource
// 改行をそろえ、前後の空白を除く。
// ================================================================================
function normalizeSource(text) {
    return String(text || "")
        .replace(/\r\n/g, "\n")
        .replace(/\r/g, "\n")
        .replace(/\u2028|\u2029/g, "\n")
        .trim();
}

// ================================================================================
// isDigit
// 半角数字かどうか。
// ================================================================================
function isDigit(character) {
    return character >= "0" && character <= "9";
}

// ================================================================================
// isPunctuationBreak
// index 位置の文字の直後で、長い行を区切るか。
// ================================================================================
function isPunctuationBreak(text, index) {
    const character = text[index];

    if (character === "。" || character === "、") {
        return true;
    }

    if (character === ".") {
        return !isDigit(text[index + 1]);
    }

    if (character === ",") {
        return !(isDigit(text[index - 1]) && isDigit(text[index + 1]));
    }

    return false;
}

// ================================================================================
// splitAtPunctuation
// 。 、 . , の直後で分ける。区切り文字は直前の単位に残す。
// ================================================================================
function splitAtPunctuation(line) {
    const parts = [];
    let start = 0;

    for (let index = 0; index < line.length; index += 1) {
        if (!isPunctuationBreak(line, index)) {
            continue;
        }

        const piece = line.slice(start, index + 1).trim();

        if (piece) {
            parts.push(piece);
        }

        start = index + 1;
    }

    const tail = line.slice(start).trim();

    if (tail) {
        parts.push(tail);
    }

    return parts;
}

// ================================================================================
// truncateOverflow
// 100文字を超える塊は、先頭100文字に "…以下略" を付けて打ち切る。
// ================================================================================
function truncateOverflow(text) {
    if (text.length <= MAX_SENTENCE_LENGTH) {
        return text;
    }

    return text.slice(0, MAX_SENTENCE_LENGTH) + OMISSION_MARK;
}

// ================================================================================
// splitLine
// 1行を読み上げ単位へ分ける。100文字未満はそのまま。
// ================================================================================
function splitLine(line) {
    if (line.length < MAX_SENTENCE_LENGTH) {
        return [line];
    }

    return splitAtPunctuation(line).map(truncateOverflow);
}

// ================================================================================
// ensureTrailingPeriod
// 末尾に 。 が無ければ付ける。. で終わる英文単位はそのまま。
// ================================================================================
function ensureTrailingPeriod(text) {
    const trimmed = String(text || "").trim();

    if (!trimmed) {
        return trimmed;
    }

    if (trimmed.endsWith("。") || trimmed.endsWith(".")) {
        return trimmed;
    }

    return trimmed + "。";
}

// ================================================================================
// splitSentences
// 本文を、読み上げ単位に分ける。
// ================================================================================
export function splitSentences(text) {
    const source = normalizeSource(text);

    if (!source) {
        return [];
    }

    const sentences = source
        .split("\n")
        .map(line => line.trim())
        .filter(line => line.length > 0)
        .flatMap(splitLine)
        .map(ensureTrailingPeriod);

    debugLog("sentence", "split", {
        chars: source.length,
        sentences: sentences.length,
        lengths: sentences.map(item => item.length),
        omitted: sentences.filter(item => item.includes(OMISSION_MARK)).length,
        preview: sentences.map(item => item.slice(0, 80))
    });

    return sentences;
}

// ================================================================================
