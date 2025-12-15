import express from "express";
import logger from "morgan";
import cors from "cors";
import bodyParser from "body-parser";
import createErrors from "http-errors";
import { auth } from "./routes/auth.js";
import { translatePt } from "./routes/translate_pt.js";
import { translateYrl } from "./routes/translate_yrl.js";
import { nextWordInfo } from "./routes/next_word_info.js";
import { dicWords } from "./routes/dic_words.js";
import { authEn } from "./routes/auth_en.js";
import { translateEn } from "./routes/translate_en.js";
import { translateYrlEn } from "./routes/translate_yrl_en.js";
import { dicWordsEn } from "./routes/dic_words_en.js";
import { tokenMiddleware, verifyToken } from "./routes/verify_token.js";
import { tokenMiddlewareEn, verifyTokenEn } from "./routes/verify_token_en.js";
import { correctYrl } from "./routes/correct_yrl.js";
import { languages } from "./routes/languages.js";
import { config } from "./routes/config.js";
import { languageUrls } from "./routes/language_urls.js";
import { changeLanguage } from "./routes/change_language.js";
import { changeLanguageEn } from "./routes/change_language_en.js";
import { timeout } from "./routes/timeout.js";
import { convertYrl } from "./routes/convert_yrl.js";
import { service } from "./shared/couchdb_service.js";
import { data } from "./shared/data.js";
import { addFeedback } from "./routes/feedback/add_feedback.js";
import { getFeedback } from "./routes/feedback/get_feedback.js";
import { deleteFeedback } from "./routes/feedback/delete_feedback.js";
import { listFeedbacks } from "./routes/feedback/list_feedbacks.js";
import { addTranslationData } from "./routes/add_translation_data.js";
import { languagesTopology } from "./routes/languages_topology.js";
import { getFeedbackConfig } from "./routes/feedback/get_feedback_config.js";
if (process.env.NODE_ENV !== "test") {
    try {
        service
            .getDocument({
            db: "assistente-nheengatu",
            docId: "config:d4b14a8a065f5cab5496fe4fe6daa975",
        })
            .then((response) => {
            console.log(response.result);
            const nextWordMode = response.result["next_word_mode"];
            const nextWordTimeout = response.result["next_word_timeout"];
            const userAuthTimeout = response.result["user_auth_timeout"];
            const fetchUrl = response.result["next_word_fetch_url"];
            const authTokenKey = response.result["auth_token_key"];
            const users = response.result["users"];
            const charMap = response.result["char_map"];
            const languages = response.result["languages"];
            const spellCheckerTimeout = response.result["spell_checker_timeout"];
            const translatorTimeout = response.result["translator_timeout"];
            const enableModelChange = response.result["enable_model_change"];
            // build a char conversion map
            const charConvMap = [];
            for (const destChar in charMap) {
                const sourceCharList = charMap[destChar];
                for (let i = 0; i < sourceCharList.length; i++) {
                    let map = [sourceCharList[i], destChar];
                    charConvMap.push(map);
                    map = [sourceCharList[i].toUpperCase(), destChar.toUpperCase()];
                    charConvMap.push(map);
                }
            }
            const translatorUrl = response.result["languages"]["Nheengatu"]["services"]["translator"];
            const nextWordUrl = response.result["languages"]["Nheengatu"]["services"]["next_word"];
            const spellCheckerUrl = response.result["languages"]["Nheengatu"]["services"]["spell_checker"];
            const translationModel = response.result["languages"]["Nheengatu"]["translation_model"];
            data.translatorUrl = translatorUrl;
            data.nextWordUrl = nextWordUrl;
            data.spellCheckerUrl = spellCheckerUrl;
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
            data.translationModel = translationModel;
            console.log(data.languages["Nheengatu"]);
        });
    }
    catch (err) {
        console.error("Error:", err);
    }
}
export const app = express();
app.use(cors()); // debugging cors error (after npm install cors) uninstall if does not work
export const port = 3000;
export const host = "0.0.0.0";
if (process.env.NODE_ENV !== "test") {
    app.listen(port);
}
app.use(logger("dev"));
app.use(express.json());
app.use(express.urlencoded({ extended: false }));
app.use(bodyParser.urlencoded({ extended: false }));
app.get("/", (req, res) => {
    res.send("Indigenous API");
});
app.post("/auth", auth);
app.post("/auth_en", authEn);
app.post("/config", config);
app.post("/translate_pt", tokenMiddleware, translatePt);
app.post("/translate_en", tokenMiddlewareEn, translateEn);
app.post("/translate_yrl", tokenMiddleware, translateYrl);
app.post("/translate_yrl_en", tokenMiddlewareEn, translateYrlEn);
app.post("/correct_yrl", tokenMiddleware, correctYrl);
app.post("/correct_yrl_en", tokenMiddlewareEn, correctYrl);
app.post("/convert_yrl", tokenMiddleware, convertYrl);
app.post("/change_language", tokenMiddleware, changeLanguage);
app.post("/change_language_en", tokenMiddlewareEn, changeLanguageEn);
app.post("/add_translation_data", tokenMiddleware, addTranslationData);
app.get("/language_urls", tokenMiddleware, languageUrls);
app.get("/languages", tokenMiddleware, languages);
app.get("/next_word_info", nextWordInfo);
app.post("/dic_words", tokenMiddleware, dicWords);
app.post("/dic_words_en", tokenMiddlewareEn, dicWordsEn);
app.post("/add_feedback", tokenMiddleware, addFeedback);
app.post("/delete_feedback", tokenMiddleware, deleteFeedback);
app.post("/list_feedbacks", tokenMiddleware, listFeedbacks);
app.get("/get_feedback", tokenMiddleware, getFeedback);
app.get("/get_feedback_config", tokenMiddleware, getFeedbackConfig);
app.get("/timeout", timeout);
app.get("/verify_token", verifyToken);
app.get("/verify_token_en", verifyTokenEn);
app.get("/languages_topology", languagesTopology);
// catch 404 and forward to error handler
app.use(function (req, res, next) {
    next(createErrors(404));
});
//# sourceMappingURL=index.js.map