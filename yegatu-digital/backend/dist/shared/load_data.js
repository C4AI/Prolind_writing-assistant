import { service } from "./couchdb_service.js";
import { data } from "./data.js";
export const loadDictionaries = async (dic, dicEn) => {
    function streamToString(stream) {
        const chunks = [];
        return new Promise((resolve, reject) => {
            stream.on("data", (chunk) => chunks.push(Buffer.from(chunk)));
            stream.on("error", (err) => reject(err));
            stream.on("end", () => resolve(Buffer.concat(chunks).toString("utf8")));
        });
    }
    async function loadDictionary(attachmentName, isEnglish = false) {
        const response = await service.getAttachment({
            db: "assistente-nheengatu",
            docId: "config:96794e4c8d17a83e696bf000a854b221",
            attachmentName,
        });
        //console.log("load_data.loadDictionary(): attachmentName = " + attachmentName);
        //console.log("load_data.loadDictionary(): response = " + response);
        //console.log("load_data.loadDictionary(): JSON.stringify(response) = " + JSON.stringify(response));
        //const csvData = await streamToString(response.result);
        const csvData = response.result;
        const lines = csvData.split("\n");
        const dicList = [];
        const dicListObj = {};
        for (const line of lines) {
            const trimmedLine = line.trim();
            if (!trimmedLine)
                continue;
            let yrlWords = [];
            let translatedWords = [];
            if (trimmedLine.startsWith('"')) {
                const closeIndex = trimmedLine.indexOf('"', 1);
                yrlWords = trimmedLine.slice(1, closeIndex).split(",").map((w) => w.trim());
                const secondQuoteIndex = trimmedLine.indexOf('"', closeIndex + 1);
                if (secondQuoteIndex !== -1) {
                    translatedWords = [trimmedLine.slice(secondQuoteIndex + 1, trimmedLine.lastIndexOf('"')).trim()];
                }
                else {
                    translatedWords = [trimmedLine.slice(closeIndex + 2).trim()];
                }
            }
            else {
                const delimiterIndex = trimmedLine.indexOf(",");
                yrlWords = [trimmedLine.slice(0, delimiterIndex).trim()];
                translatedWords = [trimmedLine.slice(delimiterIndex + 1).trim()];
            }
            for (const yrlWord of yrlWords) {
                dicList.push(`${yrlWord}: \t${translatedWords}`);
                dicListObj[yrlWord] = translatedWords;
            }
        }
        if (isEnglish) {
            data.dicListEn = dicList;
            data.dicListObjEn = dicListObj;
        }
        else {
            data.dicList = dicList;
            data.dicListObj = dicListObj;
        }
    }
    await Promise.all([
        loadDictionary(dic),
        loadDictionary(dicEn, true)
    ]);
};
//# sourceMappingURL=load_data.js.map