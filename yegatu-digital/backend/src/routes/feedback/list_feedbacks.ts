import { Request, Response } from "express";
import { service } from "../../shared/couchdb_service.js";

export const listFeedbacks = async (req: Request, res: Response) => {
  let startTimestamp;
  let endTimestamp;
  let includeLogs;
  if (
    req.body.start_timestamp === undefined ||
    req.body.end_timestamp === undefined ||
    req.body.include_logs === undefined
  ) {
    res.status(400);
    res.send();
    return;
  }
  startTimestamp = req.body.start_timestamp;
  endTimestamp = req.body.end_timestamp;
  includeLogs = req.body.include_logs;

  try {
    const selector: any = {
      _id: { $regex: "^feedback:" },
    };

    if (startTimestamp !== -1 && endTimestamp !== -1) {
      selector.timestamp = {
        $gte: Number(startTimestamp),
        $lte: Number(endTimestamp),
      };
    } else if (startTimestamp !== -1) {
      selector.timestamp = { $gte: Number(startTimestamp) };
    } else if (endTimestamp !== -1) {
      selector.timestamp = { $lte: Number(endTimestamp) };
    }

    const response = await service.postFind({
      db: "assistente-nheengatu",
      selector,
    });

    if (!response.result || !Array.isArray(response.result.docs)) {
      res.status(400).json({ error: "Nenhum feedback encontrado" });
      return;
    }

    const processedFeedbacks = response.result.docs.map((feedback: any) => {
      return includeLogs
        ? {
            feedback_id: feedback._id.split(":")[1],
            source_app: feedback.source_app,
            timestamp: feedback.timestamp,
            language: feedback.language,
            ortography: feedback.ortography,
            language_model: feedback.language_model,
            overall_feedback: feedback.overall_feedback,
            translation_feedback: feedback.translation_feedback,
            spelling_feedback: feedback.spelling_feedback,
            prediction_feedback: feedback.prediction_feedback,
            translation_log_entries: feedback.translation_log_entries,
            spelling_log_entries: feedback.spelling_log_entries,
            prediction_log_entries: feedback.prediction_log_entries,
            user: feedback.user,
            config_flags: feedback.config_flags,
          }
        : {
            feedback_id: feedback._id.split(":")[1],
            source_app: feedback.source_app,
            timestamp: feedback.timestamp,
            language: feedback.language,
            ortography: feedback.ortography,
            language_model: feedback.language_model,
            overall_feedback: feedback.overall_feedback,
            translation_feedback: feedback.translation_feedback,
            spelling_feedback: feedback.spelling_feedback,
            prediction_feedback: feedback.prediction_feedback,
            user: feedback.user,
            config_flags: feedback.config_flags,
          };
    });

    processedFeedbacks.sort((a: any, b: any) => {
      return a.timestamp - b.timestamp;
    });

    res.json(processedFeedbacks);
  } catch (err) {
    res.status(500);
    res.send();
    console.error("Error:", err);
  }
};
