// ================================================================================
// test_sentence_split.mjs
//
// extension/sentence.js の分割を、Node 上で確認する。
// ================================================================================

import {
    MAX_SENTENCE_LENGTH,
    splitSentences
} from "../extension/sentence.js";

function assert(condition, message) {
    if (!condition) {
        throw new Error(message);
    }
}

function assertEqual(actual, expected, message) {
    const actualText = JSON.stringify(actual);
    const expectedText = JSON.stringify(expected);

    if (actualText !== expectedText) {
        throw new Error(
            `${message}\n  expected: ${expectedText}\n  actual:   ${actualText}`
        );
    }
}

assertEqual(MAX_SENTENCE_LENGTH, 100, "max length");

assertEqual(splitSentences(""), [], "empty text");
assertEqual(splitSentences("   "), [], "whitespace only");

assertEqual(
    splitSentences("こんにちは。今日はいい天気です。"),
    ["こんにちは。今日はいい天気です。"],
    "short line stays one unit even with 。"
);

assertEqual(
    splitSentences("一行目\n二行目"),
    ["一行目。", "二行目。"],
    "newline is a sentence boundary and lines are not merged"
);

assertEqual(
    splitSentences("文です。\n次の行"),
    ["文です。", "次の行。"],
    "period then newline stays two units"
);

assertEqual(
    splitSentences("でしょ～？\nあの時ね"),
    ["でしょ～？。", "あの時ね。"],
    "each line gets its own trailing 。"
);

assertEqual(
    splitSentences("一行目\r\n\r\n二行目"),
    ["一行目。", "二行目。"],
    "blank lines are discarded and lines stay separate"
);

assertEqual(
    splitSentences("今日は、いい天気です。"),
    ["今日は、いい天気です。"],
    "japanese comma is not a break under 100 chars"
);

assertEqual(
    splitSentences("Hello, world."),
    ["Hello, world."],
    "english period already present; no extra 。"
);

assertEqual(
    splitSentences("Version 3.14 is ready."),
    ["Version 3.14 is ready."],
    "decimal is not a sentence end"
);

assertEqual(
    splitSentences("There are 1,000 left."),
    ["There are 1,000 left."],
    "thousands comma stays with the number"
);

assertEqual(
    splitSentences("すごい！本当に？続きです。"),
    ["すごい！本当に？続きです。"],
    "exclamation and question are not breaks"
);

assertEqual(
    splitSentences("末尾に句点がない文"),
    ["末尾に句点がない文。"],
    "trailing 。 is added when missing"
);

const long = "あ".repeat(250);
assertEqual(
    splitSentences(long),
    ["あ".repeat(100) + "…以下略。"],
    "over 100 chars without punctuation is truncated with …以下略"
);

const sample100 = "あ".repeat(100);
assertEqual(
    splitSentences(sample100),
    [sample100 + "。"],
    "exactly 100 chars stays whole and gets trailing 。"
);

const sample99 = "あ".repeat(99);
assertEqual(
    splitSentences(sample99),
    [sample99 + "。"],
    "99 chars stays one unit"
);

const shortLines = "あ".repeat(20) + "\n" + "い".repeat(20) + "\n" + "う".repeat(20);
assertEqual(
    splitSentences(shortLines),
    [
        "あ".repeat(20) + "。",
        "い".repeat(20) + "。",
        "う".repeat(20) + "。"
    ],
    "short lines are not merged across newlines"
);

const splitOnPeriod = "あ".repeat(40) + "。" + "い".repeat(80);
assertEqual(
    splitSentences(splitOnPeriod),
    ["あ".repeat(40) + "。", "い".repeat(80) + "。"],
    "a line of 100 or more splits after 。"
);

const splitOnComma = "あ".repeat(60) + "、" + "い".repeat(50);
assertEqual(
    splitSentences(splitOnComma),
    ["あ".repeat(60) + "、。", "い".repeat(50) + "。"],
    "a long line splits after 、"
);

const splitOnEnglishComma = "a".repeat(80) + ", " + "b".repeat(30);
assertEqual(
    splitSentences(splitOnEnglishComma),
    ["a".repeat(80) + ",。", "b".repeat(30) + "。"],
    "a long line splits after a comma that is not inside a number"
);

const keepThousands = "あ".repeat(95) + "1,000です。";
assertEqual(
    splitSentences(keepThousands),
    ["あ".repeat(95) + "1,000…以下略。"],
    "a comma inside 1,000 is not a break"
);

const keepDecimal = "あ".repeat(96) + "3.14です。";
assertEqual(
    splitSentences(keepDecimal),
    ["あ".repeat(96) + "3.14…以下略。"],
    "a decimal point is not a break"
);

const overflowAfterBreak = "あ".repeat(40) + "。" + "い".repeat(150);
assertEqual(
    splitSentences(overflowAfterBreak),
    [
        "あ".repeat(40) + "。",
        "い".repeat(100) + "…以下略。"
    ],
    "a piece still over 100 chars is truncated and the rest is dropped"
);

const dialogue = [
    "あ、そうそう！　こないだ私の両親に会ったでしょ～？",
    "あの時ね、いつ作るんだ～って……聞かれちゃった",
    "……なにってぇ……アハハ、こらぁ！　かまととぶっちゃってぇ♡",
    "横、失礼しちゃうよん……よっこらせい！",
    "お酒臭いぃ？　こら～！　そんなこと言っちゃだめだぁ。チューしちゃうもんね"
].join("\n\n");
const dialogueParts = splitSentences(dialogue);

assert(
    dialogueParts.every(item => item.endsWith("。")),
    "dialogue units end with 。"
);
assert(
    dialogueParts.some(item => item.includes("チューしちゃうもんね")),
    "last phrase stays with its line when the line is under 100"
);
assertEqual(dialogueParts.length, 5, "dialogue lines stay one unit each");

console.log("test_sentence_split: ok");
console.log({
    max: MAX_SENTENCE_LENGTH,
    long: splitSentences(long)[0].length,
    dialogueParts: dialogueParts.map(item => item.length),
    dialoguePreview: dialogueParts
});
