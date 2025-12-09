import { data } from "../shared/data.js";
import { Request, Response } from "express";


export const languages = (req: Request, res: Response) => {
  const username = req.body.username;

  try {
    const response = {
      languages: data.languages,
      language: data.users[username].language,
      orthography: data.users[username].orthography,
      enable_model_change: data.enableModelChange,
    };
    res.status(200);
    res.json(response);
  } catch (err) {
    console.error("Error:", err);
  }
};
