import { data } from "../shared/data.js";
import { service } from "../shared/couchdb_service.js";
import axios from "axios";
export const correctYrl = async (req, res) => {
    let sentenceYrl = req.body.sentence_yrl;
    console.log("LOG");
    // since the spell_checking service don't check the first word, add a leading filling word (duplicate first word)
    // remove anything that are not word characters before the first word
    sentenceYrl = sentenceYrl.replace(/^[^\p{L}]+/u, "");
    sentenceYrl = sentenceYrl.replace(/\s+/g, " ");
    // split in words or hyphened words
    const words = sentenceYrl.split(/[- ]+/);
    sentenceYrl = words[0] + " " + words.join(" ");
    const ortography = req.body.ortography;
    const sentenceYrlArray = sentenceYrl.split(" ");
    let selectedNextWords = req.body.selected_next_words.split(",");
    let selectedDictWords = req.body.selected_dict_words.split(",");
    selectedNextWords = [...new Set(selectedNextWords)]; // remove duplicates
    selectedDictWords = [...new Set(selectedDictWords)]; // remove duplicates
    selectedNextWords = selectedNextWords.filter((element) => sentenceYrlArray.includes(element)); // intersection of two arrays
    selectedDictWords = selectedDictWords.filter((element) => sentenceYrlArray.includes(element)); // intersection of two arrays
    if (data.languages["Nheengatu"]["ortographies"][ortography]["spell_checker"] ===
        undefined) {
        res.status(400);
        res.send();
        return;
    }
    if (typeof data.languages["Nheengatu"]["ortographies"][ortography]["spell_checker"] !== "string") {
        res.status(400);
        res.send();
        return;
    }
    const spellCheckerUrl = data.languages["Nheengatu"]["ortographies"][ortography]["spell_checker"];
    const response = await service.getDocument({
        db: "assistente-nheengatu",
        docId: "config:d4b14a8a065f5cab5496fe4fe6daa975",
    });
    let document = response.result;
    const listOfWordsToExclude = document["languages"]["Nheengatu"]["ortographies"][ortography]["spell_checker_svc"]["blacklist"];
    const filter = document["languages"]["Nheengatu"]["ortographies"][ortography]["spell_checker_svc"]["filter_cfg"];
    try {
        try {
            const axiosResponse = await axios.post(spellCheckerUrl, {
                sentence: sentenceYrl,
                orthography: ortography.toLowerCase(),
            });
            let correctedSentenceYrl = axiosResponse.data.corrected_words;
            let correctedSentenceYrlArray = correctedSentenceYrl.slice(1);
            // remove strange characters from correctedSentenceYrl
            correctedSentenceYrlArray = correctedSentenceYrlArray.map((array) => {
                return array.map((word) => {
                    console.log(word);
                    return word.replace(/[^\p{L}\s]/gu, "");
                });
            });
            // list of list of words
            let correctedSentenceYrlArrayFixed = [];
            correctedSentenceYrlArray.map((array, index) => {
                if (array[0].trim().toLowerCase() === words[index].trim().toLowerCase()) {
                    correctedSentenceYrlArrayFixed.push([]);
                }
                else {
                    correctedSentenceYrlArrayFixed.push(array);
                }
            });
            correctedSentenceYrlArrayFixed = correctedSentenceYrlArrayFixed.map((array) => array.map((word) => {
                if (listOfWordsToExclude.includes(word.toLowerCase())) {
                    console.log("INCUI?");
                    console.log(filter);
                    if (filter === "do_nothing") {
                        return word;
                    }
                    if (filter === "remove_word") {
                        return "";
                    }
                    if (filter === "remove_sentence") {
                        return "";
                    }
                    if (filter === "redact_word") {
                        return "*******";
                    }
                    else {
                        return word;
                    }
                }
                return word;
            }));
            correctedSentenceYrlArrayFixed = correctedSentenceYrlArrayFixed.map((array) => array.filter((word) => word.trim() !== ""));
            const response = {
                corrected_sentence: correctedSentenceYrlArrayFixed,
            };
            res.json(response);
        }
        catch (err) {
            console.error("Error:", err);
        }
        // register translation request in database
        const docId = Date.now();
        const dateTime = new Date().toLocaleString("pt-BR", {
            timeZone: "Brazil/East",
        });
        const doc = {
            _id: "spell_check_yrl:" + req.body.username + "_" + docId,
            created: dateTime,
            username: req.body.username,
            token: req.body.token,
            ortography: ortography,
            disable_dic: req.body.disable_dic,
            disable_next: req.body.disable_next,
            disable_word_meaning: req.body.disable_word_meaning,
            selected_next_words: selectedNextWords,
            selected_dict_words: selectedDictWords,
        };
        try {
            service
                .postDocument({
                db: "assistente-nheengatu",
                document: doc,
            })
                .then((response) => {
                console.log("Cloudant response:" + JSON.stringify(response.result));
            });
        }
        catch (err) {
            console.error("Error:", err);
        }
    }
    catch (err) {
        console.error("Error:", err);
    }
};
//# sourceMappingURL=correct_yrl.js.map