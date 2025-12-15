import { service } from "../../shared/couchdb_service.js";
export const getFeedbackConfig = async (req, res) => {
    try {
        const response = await service.getDocument({
            db: "assistente-nheengatu",
            docId: "config:d4b14a8a065f5cab5496fe4fe6daa975",
        });
        const username = req.body.username;
        let document = response.result;
        let enableFeedbackForUser = false;
        const enableFeedbackGlobal = document["enable_users_feedback"];
        if (document["users"] !== undefined &&
            document["users"][username] !== undefined &&
            document["users"][username]["enable_user_feedback"] !== undefined) {
            enableFeedbackForUser =
                document["users"][username]["enable_user_feedback"];
        }
        const feedbackCountThreshold = document["users_feedback_count_threshold"];
        const feedbackTimeThreshold = document["users_feedback_time_threshold"];
        res.json({
            enable_feedback: enableFeedbackForUser && enableFeedbackGlobal,
            feedback_count_threshold: feedbackCountThreshold,
            feedback_time_threshold: feedbackTimeThreshold,
        });
    }
    catch (err) {
        console.error("Error:", err);
        res.status(500).json({ error: "Internal server error" });
    }
};
//# sourceMappingURL=get_feedback_config.js.map