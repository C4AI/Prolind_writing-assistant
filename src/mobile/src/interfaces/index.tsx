export interface Language {
    enabled: boolean;
    ortographies?: object;
    services?: object;
  }
  
  export interface LanguagesData {
    [key: string]: Language;
  }
  