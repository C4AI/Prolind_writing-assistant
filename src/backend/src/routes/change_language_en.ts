import { Request, Response } from "express";
import { data } from "../shared/data.js";
import jwt from "jsonwebtoken";
import { service } from "../shared/couchdb_service.js";

export const changeLanguageEn = async (req: Request, res: Response) => {
  let language: string = req.body.language;
  let orthography: string = req.body.ortography;
  const username = req.body.username;

  if (!language || !orthography) {
    res.status(400);
    res.send();
    return;
  }

  // standardization of the language and orthography
  language = language.charAt(0).toUpperCase() + String(language).slice(1);
  orthography = orthography.toLowerCase();

  if (
    !data.languages[language] ||
    !data.languages[language]["ortographies"] ||
    !data.languages[language]["ortographies"][orthography]
  ) {
    res.status(400);
    res.statusMessage = "Invalid orthography or language";
    res.send();
  }

  try {
    const response = await service.getDocument({
      db: "assistente-nheengatu",
      docId: "config:d4b14a8a065f5cab5496fe4fe6daa975",
    });

    let document = response.result;

    document["users"][req.body.username]["language"] = language;
    document["users"][req.body.username]["orthography"] =
      orthography.toUpperCase();

    await service.putDocument({
      db: "assistente-nheengatu",
      docId: "config:d4b14a8a065f5cab5496fe4fe6daa975",
      document: document,
    });

    data.users[username].language = language;
    data.users[username].orthography = orthography.toUpperCase();
  } catch (err) {
    console.error("Error:", err);
  }
  res.send();
};
