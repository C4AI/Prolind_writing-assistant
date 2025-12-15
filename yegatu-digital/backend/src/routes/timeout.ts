import { Request, Response } from "express";
import { data } from "../shared/data.js";

export const timeout = (req: Request, res: Response) => {
  try {
    const response = {
      spell_checker_timeout: data.spellCheckerTimeout,
      translator_timeout: data.translatorTimeout,
    };
    res.status(200);
    res.json(response);
  } catch (err) {
    console.error("Error:", err);
  }
};
