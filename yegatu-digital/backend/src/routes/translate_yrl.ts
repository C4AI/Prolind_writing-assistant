import { Request, Response } from "express";
import { data } from "../shared/data.js";
import { service } from "../shared/couchdb_service.js";
import axios from "axios";

export const translateYrl = async (req: Request, res: Response) => {
  let sentenceYrl = req.body.sentence_yrl;
  const ortography = req.body.ortography;
  let model;

  if (!sentenceYrl || !ortography) {
    res.status(400).send();
    return;
  }

  if (sentenceYrl && sentenceYrl.trim().slice(-1) === ".") {
    sentenceYrl = sentenceYrl.trim().slice(0, -1);
  }

  if (req.body.model) {
    model = req.body.model;
  } else {
    model = data.translationModel;
  }

  console.log(model);
  const sentenceYrlArray = sentenceYrl.split(" ");
  let selectedNextWords = req.body.selected_next_words.split(",");
  let selectedDictWords = req.body.selected_dict_words.split(",");
  selectedNextWords = [...new Set(selectedNextWords)]; // remove duplicates
  selectedDictWords = [...new Set(selectedDictWords)]; // remove duplicates
  selectedNextWords = selectedNextWords.filter((element: string) =>
    sentenceYrlArray.includes(element)
  ); // intersection of two arrays
  selectedDictWords = selectedDictWords.filter((element: string) =>
    sentenceYrlArray.includes(element)
  ); //

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

  const translatorUrl: string =
    data.languages["Nheengatu"]["ortographies"][ortography]["translator"];

  try {
    try {
      const responseAxios = await axios.post(translatorUrl, {
        src_lang: "yrl",
        tgt_lang: "por",
        orthography: ortography.toLowerCase(),
        sentence: sentenceYrl,
        model: model,
      });
      let sentencePt = responseAxios.data.translated_sentence;

      const response = {
        sentence_pt: sentencePt,
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
      source_ortography: ortography,
      source_language: "nheengatu",
      target_language: "portuguese",
      disable_dic: req.body.disable_dic,
      disable_next: req.body.disable_next,
      disable_word_meaning: req.body.disable_word_meaning,
      selected_next_words: selectedNextWords,
      selected_dict_words: selectedDictWords,
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
