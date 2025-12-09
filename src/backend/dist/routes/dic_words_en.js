import { data } from "../shared/data.js";
import { loadDictionaries } from "../shared/load_data.js";
export const dicWordsEn = async (req, res) => {
    let language;
    let orthography;
    if (req.body.language) {
        language = req.body.language;
    }
    if (req.body.orthography) {
        orthography = req.body.orthography;
    }
    if (orthography && language) {
        if (!data.languages ||
            !data.languages[language] ||
            !data.languages[language].ortographies ||
            !data.languages[language].ortographies[orthography] ||
            !data.languages[language].ortographies[orthography].dictionary ||
            !data.languages[language].ortographies[orthography].dictionary_en) {
            res.json({});
        }
        const dic = data.languages[language].ortographies[orthography].dictionary;
        const dicEn = data.languages[language].ortographies[orthography].dictionary_en;
        await loadDictionaries(dic, dicEn);
    }
    try {
        const response = {
            dic_words: data.dicListObjEn,
            char_conv_map: data.charConvMap,
        };
        data.dicListObjEn = {};
        res.status(200);
        res.json(response);
    }
    catch (err) {
        console.error("Error:", err);
    }
};
//# sourceMappingURL=dic_words_en.js.map