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

const OMISSION = "…以下略";

assertEqual(MAX_SENTENCE_LENGTH, 50, "max length");

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
    "japanese comma is not a break under max length"
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
    "exclamation and question are not breaks on short lines"
);

assertEqual(
    splitSentences("末尾に句点がない文"),
    ["末尾に句点がない文。"],
    "trailing 。 is added when missing"
);

const long = "あ".repeat(250);
assertEqual(
    splitSentences(long),
    ["あ".repeat(MAX_SENTENCE_LENGTH) + OMISSION + "。"],
    "over max chars without punctuation is truncated with …以下略"
);

const sampleAtMax = "あ".repeat(MAX_SENTENCE_LENGTH);
assertEqual(
    splitSentences(sampleAtMax),
    [sampleAtMax + "。"],
    "exactly max chars stays whole and gets trailing 。"
);

const sampleUnderMax = "あ".repeat(MAX_SENTENCE_LENGTH - 1);
assertEqual(
    splitSentences(sampleUnderMax),
    [sampleUnderMax + "。"],
    "under max chars stays one unit"
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

const splitOnPeriod = "あ".repeat(20) + "。" + "い".repeat(40);
assertEqual(
    splitSentences(splitOnPeriod),
    ["あ".repeat(20) + "。", "い".repeat(40) + "。"],
    "a long line splits after 。"
);

const splitOnComma = "あ".repeat(30) + "、" + "い".repeat(30);
assertEqual(
    splitSentences(splitOnComma),
    ["あ".repeat(30) + "、。", "い".repeat(30) + "。"],
    "a long line splits after 、"
);

const splitOnEnglishComma = "a".repeat(30) + ", " + "b".repeat(25);
assertEqual(
    splitSentences(splitOnEnglishComma),
    ["a".repeat(30) + ", 。", "b".repeat(25) + "。"],
    "a long line splits after ,  (comma and space)"
);

const keepThousands = "あ".repeat(45) + "1,000です。";
assertEqual(
    splitSentences(keepThousands),
    ["あ".repeat(45) + "1,000" + OMISSION + "。"],
    "a comma inside 1,000 is not a break"
);

const keepDecimal = "あ".repeat(46) + "3.14です。";
assertEqual(
    splitSentences(keepDecimal),
    ["あ".repeat(46) + "3.14" + OMISSION + "。"],
    "a decimal point is not a break"
);

const overflowAfterBreak = "あ".repeat(20) + "。" + "い".repeat(150);
assertEqual(
    splitSentences(overflowAfterBreak),
    [
        "あ".repeat(20) + "。",
        "い".repeat(MAX_SENTENCE_LENGTH) + OMISSION + "。"
    ],
    "a piece still over max chars is truncated and the rest is dropped"
);

const splitOnExclamation = "あ".repeat(40) + "！" + "い".repeat(30);
assertEqual(
    splitSentences(splitOnExclamation),
    ["あ".repeat(40) + "！。", "い".repeat(30) + "。"],
    "a long line splits after ！ when no primary break in the window"
);

const primaryOverSecondary =
    "あ".repeat(10) + "、" + "い".repeat(10) + "。" + "う".repeat(30);
assertEqual(
    splitSentences(primaryOverSecondary),
    [
        "あ".repeat(10) + "、" + "い".repeat(10) + "。",
        "う".repeat(30) + "。"
    ],
    "primary 。 wins over secondary 、 in the same window"
);

const splitOnEnglishPeriodSpace = "a".repeat(40) + ". " + "b".repeat(20);
assertEqual(
    splitSentences(splitOnEnglishPeriodSpace),
    ["a".repeat(40) + ". ", "b".repeat(20) + "。"],
    "a long line splits after .  (period and space); space is kept"
);

const rezeroSynopsis =
    "突如、コンビニ帰りに異世界へ召喚されたひきこもり学生の菜月昴。知識も技術も武力もコミュ能力もない、ないない尽くしの凡人が、チートボーナスを与えられることもなく放り込まれた異世界で必死こいて生き抜く。彼に与えられたたった一個の祝福は、『死んだら巻き戻ります』という痛みを伴う『死に戻り』のみ！　頼れるもののいない異世界で、いったい彼は何度死に、なにを掴み取るのか。　　※血も死体も出る予定ですが、そんな派手なことにはなりません。";
const rezeroParts = splitSentences(rezeroSynopsis);

assert(
    rezeroParts[0].length > 10,
    "re:zero synopsis first unit is not only the opening comma phrase"
);
assert(
    !rezeroParts[0].startsWith("突如、。"),
    "re:zero synopsis does not start with a tiny 突如、 unit"
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
    "last phrase stays with its line when the line is under max length"
);
assertEqual(dialogueParts.length, 5, "dialogue lines stay one unit each");

console.log("test_sentence_split: ok");
console.log({
    max: MAX_SENTENCE_LENGTH,
    long: splitSentences(long)[0].length,
    dialogueParts: dialogueParts.map(item => item.length),
    dialoguePreview: dialogueParts
});
