export type BodyPt = {
  sentence_pt: string;
  ortography: string;
  disable_dic: boolean;
  disable_next: boolean;
  disable_word_meaning: boolean;
};

export type BodyYrl = {
  sentence_yrl: string;
  ortography: string;
  selected_next_words: string;
  selected_dict_words: string;
  disable_dic: boolean;
  disable_next: boolean;
  disable_word_meaning: boolean;
};

export type BodyEn = {
  sentence_en: string;
  ortography: string;
  disable_dic: boolean;
  disable_next: boolean;
  disable_word_meaning: boolean;
};

export type BodyYrlEn = {
  sentence_yrl: string;
  ortography: string;
  selected_next_words: string;
  selected_dict_words: string;
  disable_dic: boolean;
  disable_next: boolean;
  disable_word_meaning: boolean;
};

export type BodyYrlConvert = {
  sentence: string;
};

export type LanguageHeaders = {
  [key: string]: {
    [key: string]: {
      Authorization: string | null;
      "Content-Type": string;
      accept: string;
    };
  };
};

export type LanguageTopology = {
  name: string;
  enabled: boolean;
  ortographiesTopology?: OrtographyTopology[];
};

export type OrtographyTopology = {
  name: string;
  translatorEnabled: boolean;
  dictionaryEnabled: boolean;
  nextWordEnabled: boolean;
  spellCheckerEnabled: boolean;
};
