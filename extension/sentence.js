// ================================================================================
// sentence.js
//
// 読み上げ対象テキストを、TTS 向けの単位に分割する。
//
// 1. 改行で行に分ける。空行は捨て、行をまたいで結合しない
// 2. 50文字未満の行は、句読点があってもそのまま1単位
// 3. 50文字以上の行は、先頭50文字以内で区切る（区切り文字は直前の単位に残す）
//    - 第一条件: 最後の 。 または ". "（ピリオド+半角スペース）
//    - 第一条件が無いときだけ第二条件: 最後の 、 ！ ？ または ", "（カンマ+半角スペース）
//    - ", " は 1,000 のような数字内のカンマでは使わない
// 4. 区切ったあとも50文字を超える塊は、先頭50文字に "…以下略" を付け、続きは捨てる
// 5. 各読み上げ単位の末尾に "。" が無ければ付与する（末尾が . のときは付与しない）
// ================================================================================

import { debugLog } from "./debug.js";

export const MAX_SENTENCE_LENGTH = 50;
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
// findLastPrimaryBreakEnd
// 窓 [0, windowEnd) 内の第一区切りのうち、いちばん後ろの文字 index（含む）。
// ================================================================================
function findLastPrimaryBreakEnd(text, windowEnd) {
    let lastEnd = -1;

    for (let index = 0; index < windowEnd; index += 1) {
        if (text[index] === "。") {
            lastEnd = index;
        }

        if (
            text[index] === "." &&
            index + 1 < windowEnd &&
            text[index + 1] === " "
        ) {
            lastEnd = index + 1;
        }
    }

    return lastEnd;
}

// ================================================================================
// findLastSecondaryBreakEnd
// 窓内の第二区切りのうち、いちばん後ろの文字 index（含む）。
// ================================================================================
function findLastSecondaryBreakEnd(text, windowEnd) {
    let lastEnd = -1;

    for (let index = 0; index < windowEnd; index += 1) {
        const character = text[index];

        if (character === "、" || character === "！" || character === "？") {
            lastEnd = index;
        }

        if (
            character === "," &&
            index + 1 < windowEnd &&
            text[index + 1] === " "
        ) {
            const afterSpace = text[index + 2];

            if (!(isDigit(text[index - 1]) && isDigit(afterSpace))) {
                lastEnd = index + 1;
            }
        }
    }

    return lastEnd;
}

// ================================================================================
// findLastBreakEndInWindow
// 先頭 limit 文字の窓で切る位置（含む index）。無ければ -1。
// ================================================================================
function findLastBreakEndInWindow(text, limit) {
    const windowEnd = Math.min(limit, text.length);
    const primaryEnd = findLastPrimaryBreakEnd(text, windowEnd);

    if (primaryEnd >= 0) {
        return primaryEnd;
    }

    return findLastSecondaryBreakEnd(text, windowEnd);
}

// ================================================================================
// splitLongLine
// 50文字以上の行を、窓内の区切りで繰り返し分ける。
// ================================================================================
function splitLongLine(line) {
    const parts = [];
    let rest = line;

    while (rest.length >= MAX_SENTENCE_LENGTH) {
        const lastBreak = findLastBreakEndInWindow(
            rest,
            MAX_SENTENCE_LENGTH
        );

        if (lastBreak < 0) {
            parts.push(truncateOverflow(rest));
            rest = "";
            break;
        }

        const piece = rest.slice(0, lastBreak + 1).trimStart();

        if (piece) {
            parts.push(piece);
        }

        rest = rest.slice(lastBreak + 1).trimStart();
    }

    if (rest) {
        parts.push(rest);
    }

    return parts.map(truncateOverflow);
}

// ================================================================================
// truncateOverflow
// 50文字を超える塊は、先頭50文字に "…以下略" を付けて打ち切る。
// ================================================================================
function truncateOverflow(text) {
    if (text.length <= MAX_SENTENCE_LENGTH) {
        return text;
    }

    return text.slice(0, MAX_SENTENCE_LENGTH) + OMISSION_MARK;
}

// ================================================================================
// splitLine
// 1行を読み上げ単位へ分ける。50文字未満はそのまま。
// ================================================================================
function splitLine(line) {
    if (line.length < MAX_SENTENCE_LENGTH) {
        return [line];
    }

    return splitLongLine(line);
}

// ================================================================================
// ensureTrailingPeriod
// 末尾に 。 が無ければ付ける。. で終わる英文単位はそのまま。
// ================================================================================
function ensureTrailingPeriod(text) {
    const value = String(text || "").trimStart();

    if (!value) {
        return value;
    }

    const end = value.trimEnd();

    if (end.endsWith("。") || end.endsWith(".")) {
        return value;
    }

    return value + "。";
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
