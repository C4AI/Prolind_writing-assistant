import { service } from "../shared/couchdb_service.js";
import { data } from "../shared/data.js";
export const languageUrls = (req, res) => {
    const language = req.query.language;
    try {
        if (!language) {
            res.status(400);
            res.send();
            return;
        }
        service
            .getDocument({
            db: "assistente-nheengatu",
            docId: "config:d4b14a8a065f5cab5496fe4fe6daa975",
        })
            .then((response) => {
            if (!response.result["languages"][language] ||
                !response.result["languages"][language]["services"]) {
                res.status(400);
                res.send();
                return;
            }
            const translatorUrl = response.result["languages"][language]["services"]["translator"];
            const nextWordUrl = response.result["languages"][language]["services"]["next_word"];
            const spellCheckerUrl = response.result["languages"][language]["services"]["spell_checker"];
            data.translatorUrl = translatorUrl;
            data.nextWordUrl = nextWordUrl;
            data.spellCheckerUrl = spellCheckerUrl;
            res.status(200);
            res.send();
        });
    }
    catch (err) {
        console.error("Error:", err);
    }
};
//# sourceMappingURL=language_urls.js.map