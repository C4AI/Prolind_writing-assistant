import { Request, Response } from "express";
import { data } from "../shared/data.js";
import { service } from "../shared/couchdb_service.js";
import axios from "axios";
import { blacklist } from "../shared/blacklist.js";
import { constants } from "node:crypto";

export const translatePt = async (req: Request, res: Response) => {
  console.log("translatePt", req.body);
  let sentencePt = req.body.sentence_pt;
  const ortography = req.body.ortography;

  if (!sentencePt || !ortography) {
    res.status(400);
    res.send();
    return;
  }

  if (sentencePt && sentencePt.trim().slice(-1) === ".") {
    sentencePt = sentencePt.trim().slice(0, -1);
  }

  if (
    data.languages["Nheengatu"]["ortographies"][ortography]["translator"] ===
    undefined
  ) {
    res.status(400);
    res.send();
    return;
  }

  if (
    typeof data.languages["Nheengatu"]["ortographies"][ortography][
      "translator"
    ] !== "string"
  ) {
    res.status(400);
    res.send();
    return;
  }

  console.log(data.languages["Nheengatu"]);

  const translatorUrl: string =
    data.languages["Nheengatu"]["ortographies"][ortography]["translator"];

   const response = await service.getDocument({
    db: "assistente-nheengatu",
    docId: "config:d4b14a8a065f5cab5496fe4fe6daa975",
  });

  let document = response.result;

  const listOfWordsToExclude =
    document["languages"]["Nheengatu"]["ortographies"][ortography][
      "translator_svc"
    ]["blacklist"];
  const filter =
    document["languages"]["Nheengatu"]["ortographies"][ortography][
      "translator_svc"
    ]["filter_cfg"];

  let model;
  if (req.body.model) {
    model = req.body.model;
  } else {
    model = data.translationModel;
  }

  try {
    try {
      const responseAxios = await axios.post(translatorUrl, {
        src_lang: "por",
        tgt_lang: "yrl",
        orthography: ortography.toLowerCase(),
        sentence: sentencePt,
        model: model,
      });
      let sentenceYrl = responseAxios.data.translated_sentence;

      sentenceYrl = blacklist(listOfWordsToExclude, sentenceYrl, filter);
      sentenceYrl = sentenceYrl.trim();

      const response = {
        sentence_yrl: sentenceYrl,
      };

      res.json(response);
    } catch (err) {
      console.error("Error:", err);
    }
    // register translation request in database
    const docId = Date.now();
    const dateTime = new Date().toLocaleString("pt-BR", {
      timeZone: "Brazil/East",
    });
    const doc = {
      _id: "translation:" + req.body.username + "_" + docId,
      created: dateTime,
      username: req.body.username,
      token: req.body.token,
      target_ortography: ortography,
      target_language: "nheengatu",
      source_language: "portuguese",
      disable_dic: req.body.disable_dic,
      disable_next: req.body.disable_next,
      disable_word_meaning: req.body.disable_word_meaning,
    };
    try {
      service
        .postDocument({
          db: "assistente-nheengatu",
          document: doc,
        })
        .then((response) => {
          console.log("Cloudant response:" + JSON.stringify(response.result));
        });
    } catch (err) {
      console.error("Error:", err);
    }
  } catch (err) {
    console.error("Error:", err);
  }
};
