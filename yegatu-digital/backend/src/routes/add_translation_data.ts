import { Request, Response } from "express";
import { v4 as uuidv4 } from "uuid";
import { service } from "../shared/couchdb_service.js";

export const addTranslationData = async (req: Request, res: Response) => {
  let source;
  let timestamp;
  let user;
  let sourceSentence;
  let targetSentence;
  let sourceLanguage;
  let targetLanguage;

  if (
    req.body.source === undefined ||
    req.body.timestamp === undefined ||
    req.body.user === undefined ||
    req.body.source_sentence === undefined ||
    req.body.target_sentence === undefined ||
    req.body.source_language === undefined ||
    req.body.target_language === undefined
  ) {
    res.status(400);
    res.send();
    return;
  }

  source = req.body.source;
  timestamp = req.body.timestamp;
  user = req.body.user;
  sourceSentence = req.body.source_sentence;
  targetSentence = req.body.target_sentence;
  sourceLanguage = req.body.source_language;
  targetLanguage = req.body.target_language;

  const translationDataId = uuidv4();

  try {
    const doc = {
      _id: "translation_data:" + translationDataId,
      source: source,
      timestamp: timestamp,
      user: user,
      source_sentence: sourceSentence,
      target_sentence: targetSentence,
      source_language: sourceLanguage,
      target_language: targetLanguage,
    };
    console.log(doc);
    try {
      service
        .postDocument({
          db: "assistente-nheengatu",
          document: doc,
        })
        .then((response) => {
          console.log("Cloudant response:" + JSON.stringify(response.result));
        });

      res.json({ translation_data_id: translationDataId });
    } catch (err) {
      console.log("Error:", err);
    }
  } catch (err) {
    console.error("Error:", err);
  }
};
