import { data } from "../shared/data.js";
export const languagesTopology = (req, res) => {
    console.log(data.languages);
    const languagesData = data.languages;
    let languagesTopology = [];
    for (const [key, value] of Object.entries(languagesData)) {
        const typedValue = value;
        let orthographiesTopology = [];
        if (value.ortographies) {
            for (const [key2, value2] of Object.entries(value.ortographies)) {
                let orthographyTopology = {
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
        const LanguageTopology = {
            enabled: typedValue.enabled ?? false,
            name: key,
            ortographiesTopology: orthographiesTopology,
        };
        languagesTopology.push(LanguageTopology);
    }
    res.json(languagesTopology);
};
//# sourceMappingURL=languages_topology.js.map