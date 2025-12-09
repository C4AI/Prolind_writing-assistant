import { data } from "../shared/data.js";
import { service } from "../shared/couchdb_service.js";
import axios from "axios";
import { blacklist } from "../shared/blacklist.js";
export const translateEn = async (req, res) => {
    let sentenceEn = req.body.sentence_en;
    const ortography = req.body.ortography;
    let model;
    if (!sentenceEn || !ortography) {
        res.status(400);
        res.send();
        return;
    }
    console.log("Informações");
    console.log(sentenceEn);
    console.log(ortography);
    console.log(data.languages);
    console.log(data.languages["Nheengatu"]);
    console.log(data.languages["Nheengatu"]["ortographies"]);
    console.log(data.languages["Nheengatu"]["ortographies"][ortography]);
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
    console.log("Informações");
    const translatorUrl = data.languages["Nheengatu"]["ortographies"][ortography]["translator"];
    if (sentenceEn && sentenceEn.trim().slice(-1) === ".") {
        sentenceEn = sentenceEn.trim().slice(0, -1);
    }
    const response = await service.getDocument({
        db: "assistente-nheengatu",
        docId: "config:d4b14a8a065f5cab5496fe4fe6daa975",
    });
    let document = response.result;
    const listOfWordsToExclude = document["languages"]["Nheengatu"]["ortographies"][ortography]["translator_svc"]["blacklist"];
    const filter = document["languages"]["Nheengatu"]["ortographies"][ortography]["translator_svc"]["filter_cfg"];
    if (req.body.model) {
        model = req.body.model;
    }
    else {
        model = data.translationModel;
    }
    try {
        try {
            console.log(translatorUrl);
            const responseAxios = await axios.post(translatorUrl, {
                src_lang: "eng",
                tgt_lang: "yrl",
                sentence: sentenceEn,
                model: model,
            });
            console.log(responseAxios.data);
            let sentenceYrl = responseAxios.data.translated_sentence;
            sentenceYrl = blacklist(listOfWordsToExclude, sentenceYrl, filter);
            sentenceYrl = sentenceYrl.trim();
            const response = {
                sentence_yrl: sentenceYrl,
            };
            res.json(response);
        }
        catch (err) {
            console.error("Error:", err);
        }
        // register translation request in database
        const doc_id = Date.now();
        const date_time = new Date().toLocaleString("pt-BR", {
            timeZone: "Brazil/East",
        });
        const doc = {
            _id: "translation:" + req.body.username + "_" + doc_id,
            created: date_time,
            username: req.body.username,
            token: req.body.token,
            source_language: "english",
            target_language: "nheengatu",
            target_ortography: ortography,
            disable_dic: req.body.disable_dic,
            disable_next: req.body.disable_next,
            disable_word_meaning: req.body.disable_word_meaning,
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
            console.log("Error:", err);
        }
    }
    catch (err) {
        console.log("Error:", err);
    }
};
//# sourceMappingURL=translate_en.js.map