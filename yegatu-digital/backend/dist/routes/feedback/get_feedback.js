import { service } from "../../shared/couchdb_service.js";
export const getFeedback = async (req, res) => {
    let feedback_id;
    if (!req.query.feedback_id) {
        res.status(400);
        res.send();
        return;
    }
    feedback_id = req.query.feedback_id;
    try {
        try {
            service
                .getDocument({
                db: "assistente-nheengatu",
                docId: "feedback:" + feedback_id,
            })
                .then((response) => {
                res.send({
                    feedback_id: feedback_id,
                    language: response.result.language,
                    ortography: response.result.ortography,
                    language_model: response.result.language_model,
                    overall_feedback: response.result.overall_feedback,
                    translation_feedback: response.result.translation_feedback,
                    spelling_feedback: response.result.spelling_feedback,
                    prediction_feedback: response.result.prediction_feedback,
                    user: response.result.user,
                    translation_log_entries: response.result.translation_log_entries,
                    spelling_log_entries: response.result.spelling_log_entries,
                    prediction_log_entries: response.result.prediction_log_entries,
                    config_flags: response.result.config_flags,
                    source_app: response.result.source_app,
                    timestamp: response.result.timestamp,
                });
            });
        }
        catch (err) {
            console.log("Error:", err);
        }
    }
    catch (err) {
        console.error("Error:", err);
    }
};
//# sourceMappingURL=get_feedback.js.map