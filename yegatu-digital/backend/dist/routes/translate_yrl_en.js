import { data } from "../shared/data.js";
import { service } from "../shared/couchdb_service.js";
import axios from "axios";
export const translateYrlEn = async (req, res) => {
    let sentenceYrl = req.body.sentence_yrl;
    const ortography = req.body.ortography;
    if (!sentenceYrl || !ortography) {
        res.status(400).send();
        return;
    }
    if (sentenceYrl && sentenceYrl.trim().slice(-1) === ".") {
        sentenceYrl = sentenceYrl.trim().slice(0, -1);
    }
    let model;
    if (req.body.model) {
        model = req.body.model;
    }
    else {
        model = data.translationModel;
    }
    const sentenceYrlArray = sentenceYrl.split(" ");
    let selectedNextWords = req.body.selected_next_words.split(",");
    let selectedDictWords = req.body.selected_dict_words.split(",");
    selectedNextWords = [...new Set(selectedNextWords)]; // remove duplicates
    selectedDictWords = [...new Set(selectedDictWords)]; // remove duplicates
    selectedNextWords = selectedNextWords.filter((element) => sentenceYrlArray.includes(element)); // intersection of two arrays
    selectedDictWords = selectedDictWords.filter((element) => sentenceYrlArray.includes(element)); //
    if (data.languages["Nheengatu"]["ortographies"][ortography]["translator"] ===
        undefined) {
        res.status(400);
        res.send();
        return;
    }
    if (typeof data.languages["Nheengatu"]["ortographies"][ortography]["translator"] !== "string") {
        res.status(400);
        res.send();
        return;
    }
    const translatorUrl = data.languages["Nheengatu"]["ortographies"][ortography]["translator"];
    try {
        try {
            console.log(translatorUrl);
            console.log(sentenceYrl);
            console.log(model);
            const responseAxios = await axios.post(translatorUrl, {
                src_lang: "yrl",
                tgt_lang: "eng",
                sentence: sentenceYrl,
                model: "NLLB",
            });
            console.log(responseAxios.data);
            let sentenceEn = responseAxios.data.translated_sentence;
            const response = {
                sentence_en: sentenceEn,
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
            _id: "translation:" + req.body.username + "_" + docId,
            created: dateTime,
            username: req.body.username,
            token: req.body.token,
            source_ortography: ortography,
            source_language: "nheengatu",
            target_language: "english",
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
//# sourceMappingURL=translate_yrl_en.js.map