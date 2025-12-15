import { v4 as uuidv4 } from "uuid";
import { service } from "../../shared/couchdb_service.js";
export const addFeedback = async (req, res) => {
    console.log("add_feedback: called");
    let sourceApp;
    let timestamp;
    let language;
    let ortography;
    let languageModel;
    let overallFeedback;
    let translationFeedback;
    let spellingFeedback;
    let predictionFeedback;
    let user;
    let translationLogEntries;
    let spellingLogEntries;
    let predictionLogEntries;
    let configFlags;
    if (req.body.source_app === undefined ||
        req.body.timestamp === undefined ||
        req.body.ortography === undefined ||
        req.body.language === undefined ||
        req.body.overall_feedback === undefined ||
        req.body.user === undefined ||
        req.body.translation_log_entries === undefined ||
        req.body.spelling_log_entries === undefined ||
        req.body.prediction_log_entries === undefined ||
        req.body.config_flags === undefined ||
        req.body.translation_feedback === undefined ||
        req.body.spelling_feedback === undefined ||
        req.body.prediction_feedback === undefined) {
        res.status(400);
        res.send();
        console.log("add_feedback: returned code 400");
        return;
    }
    sourceApp = req.body.source_app;
    timestamp = req.body.timestamp;
    language = req.body.language;
    ortography = req.body.ortography;
    languageModel = req.body.language_model;
    overallFeedback = req.body.overall_feedback;
    translationFeedback = req.body.translation_feedback;
    spellingFeedback = req.body.spelling_feedback;
    predictionFeedback = req.body.prediction_feedback;
    user = req.body.user;
    translationLogEntries = req.body.translation_log_entries;
    spellingLogEntries = req.body.spelling_log_entries;
    predictionLogEntries = req.body.prediction_log_entries;
    configFlags = req.body.config_flags;
    const feedbackId = uuidv4();
    translationLogEntries.forEach((entry) => {
        entry.translation_log_id = uuidv4();
    });
    spellingLogEntries.forEach((entry) => {
        entry.spelling_log_id = uuidv4();
    });
    predictionLogEntries.forEach((entry) => {
        entry.prediction_log_id = uuidv4();
    });
    try {
        const doc = {
            _id: "feedback:" + feedbackId,
            source_app: sourceApp,
            timestamp: timestamp,
            language: language,
            ortography: ortography,
            language_model: languageModel,
            overall_feedback: overallFeedback,
            translation_feedback: translationFeedback,
            spelling_feedback: spellingFeedback,
            prediction_feedback: predictionFeedback,
            user: user,
            translation_log_entries: translationLogEntries,
            spelling_log_entries: spellingLogEntries,
            prediction_log_entries: predictionLogEntries,
            config_flags: configFlags,
        };
        console.log("add_feedback: doc:" + doc);
        try {
            service
                .postDocument({
                db: "assistente-nheengatu",
                document: doc,
            })
                .then((response) => {
                console.log("add_feedback: Cloudant response:" + JSON.stringify(response.result));
            });
            res.json({ feedback_id: feedbackId });
        }
        catch (err) {
            console.log("add_feedback: Error:", err);
        }
    }
    catch (err) {
        console.error("add_feedback: Error:", err);
    }
};
//# sourceMappingURL=add_feedback.js.map