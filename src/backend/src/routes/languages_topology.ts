import { Request, Response } from "express";
import { data } from "../shared/data.js";

type LanguageTopology = {
  name: string;
  enabled: boolean;
  ortographiesTopology?: OrtographyTopology[];
};

type OrtographyTopology = {
  name: string;
  translatorEnabled: boolean;
  dictionaryEnabled: boolean;
  nextWordEnabled: boolean;
  spellCheckerEnabled: boolean;
};

export const languagesTopology = (req: Request, res: Response) => {
  console.log(data.languages);
  const languagesData = data.languages;
  let languagesTopology: LanguageTopology[] = [];
  for (const [key, value] of Object.entries(languagesData)) {
    const typedValue = value as unknown as {
      enabled: boolean;
      ortographies?: any;
    };
    let orthographiesTopology = [];
    if (value.ortographies) {
      for (const [key2, value2] of Object.entries(value.ortographies)) {
        let orthographyTopology: OrtographyTopology = {
          dictionaryEnabled: false,
          name: key2,
          nextWordEnabled: false,
          spellCheckerEnabled: false,
          translatorEnabled: false,
        };
        if (value2.dictionary) {
          orthographyTopology.dictionaryEnabled = true;
        }
        if (value2.translator) {
          orthographyTopology.translatorEnabled = true;
        }
        if (value2.next_word) {
          orthographyTopology.nextWordEnabled = true;
        }
        if (value2.spell_checker) {
          orthographyTopology.spellCheckerEnabled = true;
        }
        orthographiesTopology.push(orthographyTopology);
      }
    }
    const LanguageTopology: LanguageTopology = {
      enabled: typedValue.enabled ?? false,
      name: key,
      ortographiesTopology: orthographiesTopology,
    };
    languagesTopology.push(LanguageTopology);
  }

  res.json(languagesTopology);
};
