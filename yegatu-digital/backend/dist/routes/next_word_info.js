import { data } from "../shared/data.js";
import { service } from "../shared/couchdb_service.js";
export const nextWordInfo = async (req, res) => {
    const orthography = req.query.orthography;
    const response = await service.getDocument({
        db: "assistente-nheengatu",
        docId: "config:d4b14a8a065f5cab5496fe4fe6daa975",
    });
    let document = response.result;
    if (!orthography) {
        res.status(201);
        res.send();
        return;
    }
    let listOfWordsToExclude;
    let filter;
    if (document["languages"]["Nheengatu"]["ortographies"][orthography] &&
        document["languages"]["Nheengatu"]["ortographies"][orthography]) {
        listOfWordsToExclude =
            document["languages"]["Nheengatu"]["ortographies"][orthography]["next_word_svc"]["blacklist"];
        filter =
            document["languages"]["Nheengatu"]["ortographies"][orthography]["next_word_svc"]["filter_cfg"];
    }
    else {
        res.status(201);
        res.send();
        return;
    }
    try {
        const response = {
            next_word_url: data.nextWordUrl,
            next_word_mode: data.nextWordMode,
            next_word_fetch_url: data.fetchUrl,
            next_word_timeout: data.nextWordTimeout,
            list_of_words_to_exclude: listOfWordsToExclude,
            filter: filter,
        };
        res.status(200);
        res.json(response);
    }
    catch (err) {
        console.error("Error:", err);
    }
};
//# sourceMappingURL=next_word_info.js.map