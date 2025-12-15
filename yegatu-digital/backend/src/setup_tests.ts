import { verify } from "crypto";
import { loadDictionaries } from "./shared/load_data.js";

jest.mock("@ibm-cloud/cloudant", () => {
  return {
    CloudantV1: jest.fn().mockImplementation(() => ({
      setServiceUrl: jest.fn(),
      postDocument: jest.fn(),
    })),
    IamAuthenticator: jest.fn(),
  };
});

export const dataMocked = {
  next_word_mode: 999,
  user_auth_timeout: 999,
  next_word_fetch_url: "url_mocked",
  auth_token_key: "key_mocked",
  users: { user_mocked: { language: "", orthography: "" } },
  char_map: { char_mocked: ["CHAR_MOCKED"] },
  languages: {
    language_mocked: {
      enabled: true,
      services: {},
      ortographies: {
        orthography_mocked: {
          spell_checker: "spell_checker_mocked",
          translator: "translator_mocked",
          next_word: "next_word_mocked",
        },
      },
    },
    Nheengatu: {
      enabled: true,
      services: {
        spell_checker: "spell_checker_mocked",
        translator: "translator_mocked",
        next_word: "next_word_mocked",
      },
      ortographies: {
        orthography_mocked: {
          spell_checker: "spell_checker_mocked",
          translator: "translator_mocked",
          next_word: "next_word_mocked",
        },
      },
    },
  },
  spell_checker_timeout: 999,
  next_word_timeout: 999,
  translator_timeout: 999,
  enable_model_change: false,
  dic_list_obj_en: {},
  dic_list_obj: {},
};

export const dataMockedResponse = {
  nextWordMode: 999,
  userAuthTimeout: 999,
  nextWordFetchUrl: "url_mocked",
  authTokenKey: "key_mocked",
  users: { user_mocked: { language: "", orthography: "" } },
  charConvMap: [],
  languages: {
    language_mocked: {
      enabled: true,
      services: {},
      ortographies: {
        orthography_mocked: {
          spell_checker: "spell_checker_mocked",
          translator: "translator_mocked",
          next_word: "next_word_mocked",
        },
      },
    },
    Nheengatu: {
      enabled: true,
      services: {
        spell_checker: "spell_checker_mocked",
        translator: "translator_mocked",
        next_word: "next_word_mocked",
      },
      ortographies: {
        orthography_mocked: {
          spell_checker: "spell_checker_mocked",
          translator: "translator_mocked",
          next_word: "next_word_mocked",
        },
      },
    },
  },
  spellCheckerTimeout: 999,
  nextWordTimeout: 999,
  translatorTimeout: 999,
  enableModelChange: false,
  dicListObjEn: {},
  dicListObj: {},
};

jest.mock("./shared/couchdb_service.js", () => ({
  service: {
    postDocument: jest.fn().mockResolvedValue(jest.fn()),
    getAttachment: jest
      .fn()
      .mockResolvedValue(jest.fn().mockResolvedValue(jest.fn())),
    getDocument: jest.fn().mockResolvedValue({
      result: dataMocked,
    }),
    putDocument: jest.fn(),
  },
}));

jest.mock("./shared/load_data", () => ({
  streamToString: jest.fn(),
  loadDictionaries: jest.fn(),
}));

jest.mock("./shared/data.js", () => ({
  data: {
    dicList: [],
    dicListEn: [],
    dicListObj: {},
    dicListObjEn: {},
    languages: {
      Language_mocked: {
        ortographies: {
          orthography_mocked: {},
        },
      },
      Nheengatu: {
        enabled: true,
        services: {
          spell_checker: "spell_checker_mocked",
          translator: "translator_mocked",
          next_word: "next_word_mocked",
        },
        ortographies: {
          orthography_mocked: {
            spell_checker: "spell_checker_mocked",
            translator: "translator_mocked",
            next_word: "next_word_mocked",
          },
        },
      },
    },
    charConvMap: [],
    fetchUrl: "",
    nextWordUrl: "",
    translatorUrl: "",
    nextWordMode: 0,
    nextWordTimeout: 0,
    userAuthTimeout: 0,
    spellCheckerUrl: "",
    tokenKey: "",
    users: {
      user_mocked: {
        password: "password_mocked",
        app_id: [1],
      },
    },
    spellCheckerTimeout: 0,
    translatorTimeout: 0,
    enableModelChange: false,
  },
}));
