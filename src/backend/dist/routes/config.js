import { service } from "../shared/couchdb_service.js";
import { data } from "../shared/data.js";
export const config = (req, res) => {
    try {
        service
            .getDocument({
            db: "assistente-nheengatu",
            docId: "config:d4b14a8a065f5cab5496fe4fe6daa975",
        })
            .then((response) => {
            const nextWordMode = response.result["next_word_mode"];
            const userAuthTimeout = response.result["user_auth_timeout"];
            const fetchUrl = response.result["next_word_fetch_url"];
            const authTokenKey = response.result["auth_token_key"];
            const users = response.result["users"];
            const charMap = response.result["char_map"];
            const languages = response.result["languages"];
            const spellCheckerTimeout = response.result["spell_checker_timeout"];
            const nextWordTimeout = response.result["next_word_timeout"];
            const translatorTimeout = response.result["translator_timeout"];
            const enableModelChange = response.result["enable_model_change"];
            // build a char conversion map
            var charConvMap = [];
            for (var dest_char in charMap) {
                const sourceCharList = charMap[dest_char];
                for (var i = 0; i < sourceCharList.length; i++) {
                    let map = [sourceCharList[i], dest_char];
                    charConvMap.push(map);
                    map = [sourceCharList[i].toUpperCase(), dest_char.toUpperCase()];
                    charConvMap.push(map);
                }
            }
            data.fetchUrl = fetchUrl;
            data.nextWordMode = nextWordMode;
            data.nextWordTimeout = nextWordTimeout;
            data.userAuthTimeout = userAuthTimeout;
            data.tokenKey = authTokenKey; // random value
            data.users = users;
            data.charConvMap = charConvMap;
            data.languages = languages;
            data.spellCheckerTimeout = spellCheckerTimeout;
            data.translatorTimeout = translatorTimeout;
            data.enableModelChange = enableModelChange;
        });
        res.json();
    }
    catch (err) {
        console.log("Error:", err);
    }
};
//# sourceMappingURL=config.js.map