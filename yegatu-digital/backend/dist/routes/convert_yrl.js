import { service } from "../shared/couchdb_service.js";
import axios from "axios";
export const convertYrl = async (req, res) => {
    const sentence = req.body.sentence;
    if (!sentence) {
        res.status(400);
        res.send();
        return;
    }
    try {
        try {
            const responseAxios = await axios.post("https://conversion-arn.y6dbcklf96p.us-south.codeengine.appdomain.cloud/api", {
                sentence: sentence,
            });
            const convertedSentece = responseAxios.data.converted_sentence;
            const response = {
                converted_sentence: convertedSentece,
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
            source_ortography: "CLG",
            target_ortography: "YRW",
            source_language: "nheengatu",
            target_language: "english",
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
            console.error("Error:", err);
        }
    }
    catch (err) {
        console.error("Error:", err);
    }
};
//# sourceMappingURL=convert_yrl.js.map