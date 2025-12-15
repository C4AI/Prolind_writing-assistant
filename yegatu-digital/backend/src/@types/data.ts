import { User } from "./user.js";

export type Data = {
  dicList: string[];
  dicListEn: string[];
  dicListObj: {
    [key: string]: unknown;
  };
  dicListObjEn: {
    [key: string]: unknown;
  };
  languages: {
    [language: string]: {
      [properties: string]: {
        [orthography: string]: {
          [properties: string]: unknown;
        };
      };
    };
  };
  charConvMap: unknown[][];
  fetchUrl: string;
  nextWordUrl: string;
  translatorUrl: string;
  nextWordMode: number;
  nextWordTimeout: number;
  tokenKey: string;
  userAuthTimeout: number;
  spellCheckerUrl: string;
  users: {
    [user: string]: User;
  };
  spellCheckerTimeout: number;
  translatorTimeout: number;
  enableModelChange: boolean;
  translationModel: string;
};
